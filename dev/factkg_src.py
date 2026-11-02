# %% [markdown]
# # Latent knowledge-base memory: checking claims with 16 memory tokens
#
# A small reproduction of **MEMORILLA: Latent Semantic Memory for LLMs** (Stanford, arXiv:2610.21375) on **FactKG**, the benchmark where the paper reports its largest gain (84.3% vs 80.4% for full context, with 108x fewer tokens).
#
# **The use case.** Support bots, sales assistants and compliance tools must check statements against a company's own knowledge base: *"Does the Pro plan include SSO and EU data residency?"* The answer often needs **several facts combined**. Retrieval gets worse as more facts are needed; the paper reports latent memory stays stable.
#
# **The task.** Each FactKG claim comes with its own knowledge-graph facts (up to 200 triples such as `Paul Boateng → successor → Dawn Butler`). The frozen LLM must say whether the claim is **True or False** (random = 50%).
#
# **What this notebook does**
# 1. Encodes every fact **one time** with a frozen encoder (the memory store).
# 2. Trains a small Perceiver-style **memory module**. For each claim it reads *all* the claim's facts and writes **K soft tokens** into the frozen LLM's input embedding space.
# 3. Compares on held-out test claims:
#
# | Method | What the LLM gets as memory |
# |---|---|
# | Closed-book | nothing (only what the LLM already knows) |
# | RAG top-k | the k facts most similar to the claim, as text |
# | Full context | all of the claim's facts, as text (~2K tokens) |
# | **Latent memory** | **K soft tokens** |
# | Latent memory, wrong facts | K soft tokens built from *another* claim's facts (sanity check) |
#
# 4. Makes charts for a post: accuracy vs tokens, accuracy vs how many facts the claim combines, accuracy vs knowledge-base size, and a GIF of the memory slots reading the facts.
#
# **Scoring.** The LLM compares the likelihood of answering `True` vs `False`.
#
# **Runtime.** Kaggle or Colab T4: about 3 h with the default config. Set `QUICK_RUN = True` for a ~25 min end-to-end check. **Kaggle:** turn on *Internet* and a *GPU T4* accelerator; outputs land in `/kaggle/working/latent_kb_out`.
#
# **Honest notes.** Scaled-down model pair (0.6B encoder + 0.6B LLM vs 4B + 8B in the paper) and a single training stage, so absolute numbers will differ from the paper.

# %%
!pip -q install -U "transformers>=4.56" "datasets>=3.0" huggingface_hub matplotlib pillow

# %% [markdown]
# ## 1. Config
# Change only this cell to scale the experiment up or down.

# %%
import os
QUICK_RUN = False      # True: small sizes, ~25 min end-to-end check

LLM_NAME = "Qwen/Qwen3-0.6B"              # frozen decoder (try "Qwen/Qwen3-1.7B" on L4/A100)
ENC_NAME = "Qwen/Qwen3-Embedding-0.6B"    # frozen encoder for the memory store and RAG
DATA_REPO = "MemoryAsModality/FactKG"

N_TRAIN = 30000         # training claims (of ~86K)
N_EVAL = 1000           # held-out test claims (of ~9K)

K_SLOTS = 16            # memory tokens given to the LLM (paper: 16)
N_BLOCKS = 2            # cross-attention read blocks (paper: 2)
N_HEADS = 8
RAG_KS = [5, 20]        # RAG baselines: top-5 and top-20 facts

MAX_Q_TOKENS = 128
MAX_ANS_TOKENS = 8
EPOCHS = 2
LR = 2e-4
TAU = 1.0               # temperature of the True/False loss
AUX_WEIGHT = 0.5        # training-only: make the LLM restate the most relevant fact from the memory tokens
PREFILL_CHUNK = 1024
SEED = 0

if QUICK_RUN:
    N_TRAIN, N_EVAL, EPOCHS = 4000, 300, 2

ON_KAGGLE = os.path.exists("/kaggle/working")
CACHE_DIR = "/tmp/lkb_cache" if ON_KAGGLE else "cache"
OUT_DIR = "/kaggle/working/latent_kb_out" if ON_KAGGLE else "latent_kb_out"

# %%
def load_hf_token():
    """Optional. Public models and data need no token; a token only avoids download rate limits."""
    if os.environ.get("HF_TOKEN"):
        return "environment"
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        return "Kaggle secret"
    except Exception:
        pass
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        return "Colab secret"
    except Exception:
        return None

print("HF token:", load_hf_token() or "none (fine for public models)", "| quick run:", QUICK_RUN, "| Kaggle:", ON_KAGGLE)

# %%
import json, math, random, time, re, textwrap
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from contextlib import nullcontext
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8   # T4 has no real bf16
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
BATCH = 16 if BF16 else 8          # claims per micro-batch
GRAD_ACCUM = 1 if BF16 else 2      # micro-batches per optimizer step

def amp():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if DEV == "cuda" else nullcontext()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", "| bf16:", BF16, "| batch:", BATCH, "x", GRAD_ACCUM)

# %% [markdown]
# ## 2. Load FactKG
# Each row is one claim with its own list of knowledge-graph facts (`documents`). `reasoning_type` says how many facts the claim combines (`num1`…`num4`) and whether it uses negation, substitution or existence.

# %%
def n_facts(rt):
    m = re.findall(r"num(\d)", rt)
    return max(int(x) for x in m) if m else (0 if "existence" in rt else -1)

def as_list(x):
    return [] if x is None else list(x)   # parquet lists arrive as numpy arrays

def to_claim(r):
    claim = r["question"].split("Claim:", 1)[-1].strip()
    docs = [str(d) for d in as_list(r["documents"])] or ["(no facts)"]
    correct = str(r["correct_answer"]).strip()
    return dict(claim=claim, docs=docs, label=correct, options=[correct, "False" if correct == "True" else "True"],
                reasoning=str(r["reasoning_type"]), n_facts=n_facts(str(r["reasoning_type"])), style=str(r["style"]),
                entities=[str(e) for e in as_list(r["entities"])])

train_df = load_dataset(DATA_REPO, split="train").to_pandas()
test_df = load_dataset(DATA_REPO, split="test").to_pandas()
train_c = [to_claim(r) for _, r in train_df.sample(n=min(N_TRAIN, len(train_df)), random_state=SEED).iterrows()]
test_c = [to_claim(r) for _, r in test_df.sample(n=min(N_EVAL, len(test_df)), random_state=SEED).iterrows()]
print(f"train claims: {len(train_c)} | eval claims: {len(test_c)} | True share (eval): {np.mean([c['label'] == 'True' for c in test_c]):.0%}")
print("facts per claim (eval): median", int(np.median([len(c["docs"]) for c in test_c])), "max", max(len(c["docs"]) for c in test_c))
print("example:", test_c[0]["claim"], "->", test_c[0]["label"], "|", test_c[0]["docs"][:2])

# %% [markdown]
# ## 3. Build the memory store: encode every fact once
# Facts repeat across claims (the same entity appears in many claims), so each unique fact is encoded only once.

# %%
enc_tok = AutoTokenizer.from_pretrained(ENC_NAME, padding_side="left")
enc = AutoModel.from_pretrained(ENC_NAME, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
QUERY_INSTR = "Instruct: Given a claim, retrieve knowledge-graph facts that help verify it\nQuery: "

@torch.no_grad()
def embed(texts, bs=128, max_len=96):
    order = np.argsort([-len(t) for t in texts])
    out = torch.empty(len(texts), enc.config.hidden_size)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        b = enc_tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        out[torch.as_tensor(idx)] = F.normalize(enc(**b).last_hidden_state[:, -1].float(), dim=-1).cpu()   # last-token pooling
    return out

fact_text = sorted({d for c in train_c + test_c for d in c["docs"]})
fact_id = {t: i for i, t in enumerate(fact_text)}
cache_path = os.path.join(CACHE_DIR, f"facts_{len(fact_text)}_{N_TRAIN}_{N_EVAL}.pt")
t0 = time.time()
FACT_EMB = torch.load(cache_path) if os.path.exists(cache_path) else embed(fact_text).half()
torch.save(FACT_EMB, cache_path)
for c in train_c + test_c:
    c["ids"] = torch.tensor([fact_id[d] for d in c["docs"]])
qembs = embed([QUERY_INSTR + c["claim"] for c in train_c + test_c], bs=256)
for c, e in zip(train_c + test_c, qembs):
    c["qemb"] = e
    sims = FACT_EMB[c["ids"]].float() @ e
    c["rank"] = torch.argsort(sims, descending=True)        # facts ordered by similarity, for RAG and the aux target
print(f"unique facts encoded: {len(fact_text):,} in {time.time() - t0:.0f}s")
H_ENC = qembs.shape[1]
del enc
if DEV == "cuda": torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Frozen LLM and option scoring

# %%
tok = AutoTokenizer.from_pretrained(LLM_NAME)
llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype=torch.bfloat16 if BF16 else torch.float32).to(DEV).eval()
for p in llm.parameters():
    p.requires_grad_(False)
EMB = llm.get_input_embeddings()
H_LLM = EMB.weight.shape[1]
EMB_STD = EMB.weight.float().std().item()
HEADER = "Knowledge-graph facts:\n"
ASK = "Is the following claim true or false? Answer with True or False.\nClaim: "

def chat_ids(claim, system=None):
    q_ids = tok(claim, add_special_tokens=False).input_ids[:MAX_Q_TOKENS]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": ASK + tok.decode(q_ids)}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor(tok(s, add_special_tokens=False).input_ids)

def ans_ids(text):
    return torch.tensor(tok(text, add_special_tokens=False).input_ids[:MAX_ANS_TOKENS])

def embed_ids(ids):
    return EMB(ids.to(DEV))

@torch.no_grad()
def score_options(prefix_embeds, options):
    """Mean log-prob of each option after the prefix. The prefix is computed once and reused (KV cache)."""
    cache = None
    for s0 in range(0, prefix_embeds.shape[1], PREFILL_CHUNK):
        with amp():
            out = llm(inputs_embeds=prefix_embeds[:, s0:s0 + PREFILL_CHUNK], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    first = out.logits[:, -1].float()
    scores = []
    for o in options:
        ids = ans_ids(o).to(DEV)[None]
        with amp():
            r = llm(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = torch.cat([first[:, None], r.logits[:, :-1].float()], 1)
        scores.append(logits.log_softmax(-1).gather(-1, ids[..., None]).mean().item())
        cache.crop(-ids.shape[1])   # drop this option's tokens, keep the prefix
    return scores

def facts_text(c, idx):
    return "\n".join(c["docs"][i] for i in sorted(idx))   # keep the original order

def n_tokens(text):
    return len(tok(text, add_special_tokens=False).input_ids)

print("LLM hidden:", H_LLM, "| encoder dim:", H_ENC, "| token-embedding std:", round(EMB_STD, 4))

# %% [markdown]
# ## 5. The memory module
# Same 4 stages as the paper (Sec. 2.2):
# 1. **K learned slots** `Q0`.
# 2. **Query conditioning:** `Qc = LN(Q0 + CrossAttn(Q0, z_claim))`.
# 3. **Read all facts:** `N_BLOCKS` × (cross-attention over every fact embedding + FFN).
# 4. **Project** into the LLM's token-embedding space → K soft tokens, put in front of the prompt.
#
# Only this module trains. The encoder and the LLM stay frozen.

# %%
class ReadBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_q, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, R, Z, pad_mask):
        a, w = self.attn(self.ln_q(R), Z, Z, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=True)
        R = R + a
        return R + self.ffn(self.ln_f(R)), w

class MemoryModule(nn.Module):
    def __init__(self, h_enc, h_llm, k, n_blocks, heads, emb_std):
        super().__init__()
        self.Q0 = nn.Parameter(torch.randn(k, h_enc) * 0.02)
        self.q_attn = nn.MultiheadAttention(h_enc, heads, batch_first=True)
        self.ln_c, self.ln_z, self.ln_out = nn.LayerNorm(h_enc), nn.LayerNorm(h_enc), nn.LayerNorm(h_enc)
        self.blocks = nn.ModuleList(ReadBlock(h_enc, heads) for _ in range(n_blocks))
        self.proj = nn.Linear(h_enc, h_llm, bias=False)
        self.scale = nn.Parameter(torch.tensor(emb_std / 0.577))   # match the LLM's token-embedding scale

    def forward(self, Z, pad_mask, zq):
        B = Z.shape[0]
        Q0 = self.Q0[None].expand(B, -1, -1)
        a, _ = self.q_attn(Q0, zq[:, None], zq[:, None])
        R = self.ln_c(Q0 + a)
        Zn = self.ln_z(Z)
        attns = []
        for blk in self.blocks:
            R, w = blk(R, Zn, pad_mask)
            attns.append(w)                        # [B, K, N]
        return self.proj(self.ln_out(R)) * self.scale, attns

def batch_store(id_lists):
    N = max(len(ids) for ids in id_lists)
    Z = torch.zeros(len(id_lists), N, H_ENC)
    mask = torch.ones(len(id_lists), N, dtype=torch.bool)
    for i, ids in enumerate(id_lists):
        Z[i, :len(ids)], mask[i, :len(ids)] = FACT_EMB[ids].float(), False
    return Z.to(DEV), mask.to(DEV)

def memory_for(cs, id_lists=None):
    Z, mask = batch_store(id_lists or [c["ids"] for c in cs])
    zq = torch.stack([c["qemb"] for c in cs]).to(DEV)
    return mem(Z, mask, zq)

mem = MemoryModule(H_ENC, H_LLM, K_SLOTS, N_BLOCKS, N_HEADS, EMB_STD).to(DEV)
print(f"trainable params: {sum(p.numel() for p in mem.parameters()) / 1e6:.1f}M (LLM frozen: {sum(p.numel() for p in llm.parameters()) / 1e6:.0f}M)")

# %% [markdown]
# ## 6. Train
# Loss = cross-entropy over the `True`/`False` log-likelihoods + `AUX_WEIGHT` × NLL of restating the claim's most similar fact from the memory tokens. The second term is a training-only signal that teaches the slots to carry fact content, in the spirit of the paper's reconstruction pre-training. Gradients flow through the frozen LLM into the soft tokens.

# %%
AUX_LEAD = "Relevant fact: "

def option_logprobs(M, cs, with_aux=False):
    """Differentiable mean log-prob of every option, given soft tokens M [B,K,H]. Returns [B, n_opts (+1 if with_aux)]."""
    dt = M.dtype
    seqs = []
    for i, c in enumerate(cs):
        prefix = torch.cat([M[i], embed_ids(chat_ids(c["claim"])).to(dt)], 0)
        targets = [ans_ids(o) for o in c["options"]]
        if with_aux:
            targets.append(torch.tensor(tok(AUX_LEAD + c["docs"][int(c["rank"][0])], add_special_tokens=False).input_ids[:48]))
        for a in targets:
            a = a.to(DEV)
            seqs.append((torch.cat([prefix, embed_ids(a).to(dt)], 0), prefix.shape[0], a))
    T = max(s[0].shape[0] for s in seqs)
    X = torch.zeros(len(seqs), T, H_LLM, device=DEV, dtype=dt)
    att = torch.zeros(len(seqs), T, dtype=torch.long, device=DEV)
    rows, cols, tgt, seg = [], [], [], []
    for j, (x, P, a) in enumerate(seqs):
        X[j, :len(x)], att[j, :len(x)] = x, 1
        pos = torch.arange(P - 1, P - 1 + len(a), device=DEV)
        rows.append(torch.full_like(pos, j)); cols.append(pos); tgt.append(a); seg.append(torch.full_like(pos, j))
    rows, cols, tgt, seg = map(torch.cat, (rows, cols, tgt, seg))
    with amp():
        h = llm.model(inputs_embeds=X, attention_mask=att).last_hidden_state[rows, cols]
        logits = llm.lm_head(h)
    lp = -F.cross_entropy(logits.float(), tgt, reduction="none")
    per_seq = torch.zeros(len(seqs), device=DEV).index_add(0, seg, lp) / torch.bincount(seg, minlength=len(seqs)).clamp(min=1)
    return per_seq.view(len(cs), -1)

opt = torch.optim.AdamW(mem.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(train_c) / (BATCH * GRAD_ACCUM))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda" and not BF16)
history, step, t0 = [], 0, time.time()
mem.train()
for ep in range(EPOCHS):
    order = list(range(len(train_c))); random.shuffle(order)
    for b in range(0, len(order), BATCH * GRAD_ACCUM):
        group = order[b:b + BATCH * GRAD_ACCUM]
        opt.zero_grad(set_to_none=True)
        stats = []
        for a0 in range(0, len(group), BATCH):
            cs = [train_c[i] for i in group[a0:a0 + BATCH]]
            M, _ = memory_for(cs)
            s = option_logprobs(M, cs, with_aux=AUX_WEIGHT > 0)
            s, s_aux = (s[:, :-1], s[:, -1]) if AUX_WEIGHT > 0 else (s, torch.zeros(len(cs), device=DEV))
            loss = F.cross_entropy(s / TAU, torch.zeros(len(cs), dtype=torch.long, device=DEV)) + AUX_WEIGHT * (-s_aux.mean())
            scaler.scale(loss * len(cs) / len(group)).backward()
            stats.append((loss.item(), (s.argmax(1) == 0).float().mean().item(), -s_aux.mean().item()))
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        l, a, ax = np.mean(stats, 0)
        history.append(dict(step=step, loss=l, acc=a, aux_nll=ax))
        if step % 50 == 0 or step == steps:
            h = pd.DataFrame(history[-50:])
            print(f"ep {ep} step {step}/{steps} loss {h.loss.mean():.3f} train-acc {h.acc.mean():.1%} "
                  f"fact-nll {h.aux_nll.mean():.2f} amp-scale {scaler.get_scale() if scaler.is_enabled() else 1:.0f} {time.time() - t0:.0f}s")
mem.eval()
if DEV == "cuda": torch.cuda.empty_cache()
torch.save(mem.state_dict(), os.path.join(OUT_DIR, "memory_module.pt"))

# %% [markdown]
# ## 7. Evaluate all methods on held-out claims

# %%
def timed(fn):
    if DEV == "cuda": torch.cuda.synchronize()
    t = time.time(); r = fn()
    if DEV == "cuda": torch.cuda.synchronize()
    return r, time.time() - t

METHODS = ([("closed_book", "Closed-book")] + [(f"rag{k}", f"RAG top-{k}") for k in RAG_KS] +
           [("full", "Full context"), ("memory", f"Latent memory (K={K_SLOTS})"), ("memory_wrong", "Latent memory, wrong facts")])
rows = []
for n, c in enumerate(test_c):
    r = dict(i=n, claim=c["claim"], label=c["label"], reasoning=c["reasoning"], n_facts=c["n_facts"], style=c["style"], kb_size=len(c["docs"]))
    def run(name, prefix, mem_tokens):
        sc, sec = timed(lambda: score_options(prefix, c["options"]))
        r[name], r[name + "_pick"], r[name + "_tok"], r[name + "_sec"] = int(np.argmax(sc) == 0), c["options"][int(np.argmax(sc))], mem_tokens, sec
    with torch.no_grad():
        run("closed_book", embed_ids(chat_ids(c["claim"]))[None], 0)
        for k in RAG_KS:
            text = facts_text(c, c["rank"][:k].tolist())
            run(f"rag{k}", embed_ids(chat_ids(c["claim"], HEADER + text))[None], n_tokens(text))
        text = facts_text(c, range(len(c["docs"])))
        run("full", embed_ids(chat_ids(c["claim"], HEADER + text))[None], n_tokens(text))
        with amp():
            M, attns = memory_for([c])
        run("memory", torch.cat([M.to(EMB.weight.dtype), embed_ids(chat_ids(c["claim"]))[None]], 1), K_SLOTS)
        c["attn"] = torch.stack([a[0].float().cpu() for a in attns])     # [blocks, K, N] for the GIF
        other = test_c[(n + 1) % len(test_c)]
        with amp():
            Mw, _ = memory_for([c], id_lists=[other["ids"]])
        run("memory_wrong", torch.cat([Mw.to(EMB.weight.dtype), embed_ids(chat_ids(c["claim"]))[None]], 1), K_SLOTS)
    rows.append(r)
    if n % 100 == 0:
        print(n, {m: round(np.mean([x[m] for x in rows]), 3) for m, _ in METHODS})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "results_per_claim.csv"), index=False)

# %% [markdown]
# ## 8. Results table

# %%
def ci(x):
    p = x.mean(); return 1.96 * math.sqrt(p * (1 - p) / max(len(x), 1))
table = pd.DataFrame([dict(method=lbl, accuracy=res[m].mean(), ci95=ci(res[m]), memory_tokens=res[m + "_tok"].mean(),
                           sec_per_claim=res[m + "_sec"].mean()) for m, lbl in METHODS])
table.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
print(f"eval claims: {len(res)}. Random guess = 50%; always answering the majority label = {max(res.label.eq('True').mean(), res.label.eq('False').mean()):.1%}.")
display(table.style.format({"accuracy": "{:.1%}", "ci95": "±{:.1%}", "memory_tokens": "{:,.0f}", "sec_per_claim": "{:.3f}"}))

# %% [markdown]
# ## 9. Charts for the post

# %%
COLORS = {"closed_book": "#888780", "full": "#D85A30", "memory": "#7F77DD", "memory_wrong": "#B4B2A9",
          **{f"rag{k}": col for k, col in zip(RAG_KS, ["#85B7EB", "#378ADD", "#185FA5"])}}
LBL = dict(METHODS)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# 1) accuracy vs memory tokens
fig, ax = plt.subplots(figsize=(7, 4.2))
for m, lbl in METHODS[:-1]:
    x, y = max(res[m + "_tok"].mean(), 1), res[m].mean()
    ax.scatter(x, y * 100, s=200 if m == "memory" else 90, color=COLORS[m], marker="*" if m == "memory" else "o", zorder=3)
    ax.annotate(f"{lbl}\n{y:.1%} · {res[m + '_tok'].mean():,.0f} tok", (x, y * 100), textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.axhline(50, ls=":", color="#B4B2A9"); ax.text(1.1, 50.6, "random guess", fontsize=8, color="#888780")
ax.set_xscale("log"); ax.set_xlabel("memory tokens given to the LLM (log scale)"); ax.set_ylabel("accuracy (%)")
ax.set_title(f"FactKG claim checking · frozen {LLM_NAME.split('/')[-1]} · {len(res)} held-out claims")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "1_accuracy_vs_tokens.png")); plt.show()

# 2) accuracy vs number of facts the claim combines (paper Fig. 4c)
show = [f"rag{k}" for k in RAG_KS] + ["full", "memory"]
g = res[res.n_facts >= 1].groupby("n_facts")[show].mean() * 100
ax = g.plot(marker="o", figsize=(6.5, 4), color=[COLORS[m] for m in show])
ax.set_xticks(g.index); ax.set_xlabel("facts combined in the claim"); ax.set_ylabel("accuracy (%)")
ax.legend([LBL[m] for m in show], fontsize=8); ax.set_title("Does accuracy hold as claims combine more facts?")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "2_accuracy_vs_facts_combined.png")); plt.show()

# 3) accuracy vs knowledge-base size (facts available for the claim)
res["kb_bin"] = pd.cut(res.kb_size, [0, 50, 150, 1000], labels=["1-50 facts", "51-150", "151-200"])
g = res.groupby("kb_bin", observed=False)[show].mean() * 100
ax = g.plot.bar(figsize=(6.5, 3.8), color=[COLORS[m] for m in show], rot=0, width=0.75)
ax.set_xlabel("size of the knowledge base for the claim"); ax.set_ylabel("accuracy (%)")
ax.legend([LBL[m] for m in show], fontsize=8); ax.set_title("Bigger knowledge base, same 16 tokens")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "3_accuracy_vs_kb_size.png")); plt.show()

# 4) reasoning types
def kind(rt):
    return "negation" if "negation" in rt else "existence" if "existence" in rt else "substitution" if "substitution" in rt else "direct"
res["kind"] = res.reasoning.map(kind)
display((res.groupby("kind")[show + ["closed_book"]].mean() * 100).round(1).join(res.groupby("kind").size().rename("n")))

# %% [markdown]
# ## 10. Attention GIF
# One held-out claim. Each frame is one memory slot. The bars show the facts that slot reads most (of all the claim's facts).

# %%
cand = res[(res.memory == 1) & (res.kb_size >= 20) & (res.n_facts >= 2)]
r0 = f"rag{RAG_KS[0]}"
cand = cand[cand[r0] == 0] if (cand[r0] == 0).any() else cand
qi = int(cand.i.iloc[0]) if len(cand) else 0
c = test_c[qi]
A = c["attn"][-1].numpy()                                   # [K, N] last read block
K, N = A.shape
TOP = min(8, N)
fig, ax = plt.subplots(figsize=(8, 4.2))
def frame(k):
    ax.clear()
    w = A[k] if k < K else A.mean(0)
    top = np.argsort(-w)[:TOP][::-1]
    ax.barh(range(TOP), w[top], color="#7F77DD")
    ax.set_yticks(range(TOP)); ax.set_yticklabels([textwrap.shorten(c["docs"][j], 55) for j in top], fontsize=7)
    ax.set_xlim(0, max(A.max(), 1e-6) * 1.1); ax.set_xlabel(f"attention (out of {N} facts)")
    who = f"slot m{k + 1} of {K}" if k < K else f"all {K} slots (mean)"
    ax.set_title(textwrap.shorten(f'Claim: "{c["claim"]}" ({c["label"]})', 90) + f"\n{who}", fontsize=9)
    fig.tight_layout()
ani = animation.FuncAnimation(fig, frame, frames=K + 1, interval=700)
ani.save(os.path.join(OUT_DIR, "4_memory_attention.gif"), writer=animation.PillowWriter(fps=1.5))
plt.close(fig)
from IPython.display import Image
Image(filename=os.path.join(OUT_DIR, "4_memory_attention.gif"))

# %% [markdown]
# ## 11. One example, side by side

# %%
r = res[res.i == qi].iloc[0]
print("CLAIM:", c["claim"], "| TRUE LABEL:", c["label"], "| facts available:", len(c["docs"]), "| facts combined:", c["n_facts"])
for m, lbl in METHODS:
    print(f"{lbl:30s} {'RIGHT' if r[m] else 'wrong'} -> {r[m + '_pick']}")

# %% [markdown]
# ## 12. Summary for the post

# %%
t = table.set_index("method")
mm, fc = LBL["memory"], LBL["full"]
rag_lines = "\n".join(f"  - {LBL[f'rag{k}']}: {t.loc[LBL[f'rag{k}'], 'accuracy']:.1%} with ~{t.loc[LBL[f'rag{k}'], 'memory_tokens']:,.0f} tokens" for k in RAG_KS)
print(f"""On {len(res)} held-out FactKG claims (True/False, random = 50%), frozen {LLM_NAME}:
  - Closed-book: {t.loc['Closed-book', 'accuracy']:.1%}
{rag_lines}
  - Full context: {t.loc[fc, 'accuracy']:.1%} with ~{t.loc[fc, 'memory_tokens']:,.0f} tokens, {t.loc[fc, 'sec_per_claim']:.3f}s/claim
  - {mm}: {t.loc[mm, 'accuracy']:.1%} with {K_SLOTS} tokens ({t.loc[fc, 'memory_tokens'] / K_SLOTS:,.0f}x fewer than full context), {t.loc[mm, 'sec_per_claim']:.3f}s/claim
  - Sanity check, memory built from the wrong facts: {t.loc[LBL['memory_wrong'], 'accuracy']:.1%}
Trainable params: {sum(p.numel() for p in mem.parameters()) / 1e6:.1f}M. LLM and encoder frozen.""")
