# %% [markdown]
# # latent-user-memory: a user's 32K-token chat history in 16 tokens
#
# A small reproduction of **MEMORILLA: Latent Semantic Memory for LLMs** (Stanford, arXiv:2610.21375) on the **PersonaMem-v2** benchmark.
#
# **The question:** an AI assistant has ~32,000 tokens of past chats with a user. A new request comes in. Can a frozen LLM give the *personalized* answer if it only gets **16 learned memory tokens** instead of the whole history?
#
# **What this notebook does**
# 1. Splits each user's chat history into chunks and encodes every chunk **one time** with a frozen encoder (the memory store).
# 2. Trains a small Perceiver-style **memory module** (Memorilla-mini). For each question it reads *all* chunks and writes **K soft tokens** into the LLM's input embedding space. The LLM stays frozen.
# 3. Compares five setups on held-out users:
#
# | Method | What the LLM gets as memory |
# |---|---|
# | Closed-book | nothing |
# | RAG top-k | k most similar chunks, as text |
# | Full context | the whole history, as text (~30K tokens) |
# | **Memorilla-mini** | **K soft tokens** |
# | Memorilla-mini, wrong user | K soft tokens built from *another* user's history (sanity check) |
#
# 4. Makes charts for a post: accuracy vs tokens, accuracy vs how far back the memory is, preference-update cases, "did the model look at the right chunk?", and an attention GIF.
#
# **Scoring.** PersonaMem-v2 gives 1 correct + 3 incorrect answers per question. The frozen LLM scores each option by average log-likelihood; the highest wins. Random guess = 25%.
#
# **Runtime.** Runs on Kaggle or Colab. T4: about 1.5–2.5 h with the default config (most of it is embedding the histories and the full-context baseline). L4/A100: much faster. Set `QUICK_RUN = True` for a ~15 min end-to-end check.
#
# **Kaggle:** turn on *Internet* and a *GPU T4* accelerator. Outputs land in `/kaggle/working/memorilla_out` (the notebook's Output tab).
#
# **Honest notes.** PersonaMem-v2 is synthetic (LLM-generated personas and chats). This is a scaled-down model pair (0.6B encoder + 0.6B LLM vs 4B + 8B in the paper) and a shorter training curriculum, so absolute numbers will differ from the paper.

# %%
!pip -q install -U "transformers>=4.56" "datasets>=3.0" huggingface_hub matplotlib pillow

# %% [markdown]
# ## 1. Config
# Change only this cell to scale the experiment up or down.

# %%
import os
QUICK_RUN = False      # True: tiny sizes, ~15 min end-to-end check

LLM_NAME = "Qwen/Qwen3-0.6B"              # frozen decoder (try "Qwen/Qwen3-1.7B" on L4/A100)
ENC_NAME = "Qwen/Qwen3-Embedding-0.6B"    # frozen encoder for the memory store and RAG
DATA_REPO = "bowen-upenn/PersonaMem-v2"

N_TRAIN_HIST = 400      # training users (of 800). Each history is ~32K tokens
N_TEST_HIST = 100       # held-out users (of 200) from the benchmark split
N_EVAL = 500            # benchmark questions to evaluate
N_FULL_EVAL = 150       # subset that also runs the expensive full-context baseline

CHUNK_TOKENS = 256      # tokens per memory chunk
K_SLOTS = 16            # memory tokens given to the LLM (paper: 16)
N_BLOCKS = 2            # cross-attention read blocks (paper: 2)
N_HEADS = 8
RAG_K = 3               # chunks for the RAG baseline
FULL_CTX_MAX_TOKENS = 30000

MAX_Q_TOKENS = 128
MAX_ANS_TOKENS = 80
EPOCHS = 2
LR = 2e-4
TAU = 0.1               # temperature of the multiple-choice loss
NLL_WEIGHT = 0.5        # weight of the paper's answer-NLL loss
SEED = 0

if QUICK_RUN:
    N_TRAIN_HIST, N_TEST_HIST, N_EVAL, N_FULL_EVAL, EPOCHS = 40, 10, 60, 15, 1

ON_KAGGLE = os.path.exists("/kaggle/working")
CACHE_DIR = "/tmp/lum_cache" if ON_KAGGLE else "cache"   # on Colab, point this to Google Drive to keep embeddings
OUT_DIR = "/kaggle/working/memorilla_out" if ON_KAGGLE else "memorilla_out"

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
import os, json, ast, math, random, time, re
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from contextlib import nullcontext
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8   # T4 has no real bf16
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
BATCH = 8 if BF16 else 4

def amp():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if DEV == "cuda" else nullcontext()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", "| bf16:", BF16, "| batch:", BATCH)

# %% [markdown]
# ## 2. Load PersonaMem-v2
# Each row is one question for one user. `chat_history_32k_link` points to that user's history file. Many questions share one history, so each history is encoded only once.

# %%
def parse_obj(x):
    if not isinstance(x, str):
        return x
    for f in (json.loads, ast.literal_eval):
        try:
            return f(x)
        except Exception:
            pass
    return x

def to_question(r):
    q = parse_obj(r["user_query"])
    q = q.get("content", "") if isinstance(q, dict) else str(q)
    wrong = parse_obj(r["incorrect_answers"])
    wrong = [str(w) for w in (wrong if isinstance(wrong, list) else [wrong])]
    snip = parse_obj(r["related_conversation_snippet"])
    return dict(link=r["chat_history_32k_link"], persona=int(r["persona_id"]), query=q,
                options=[str(r["correct_answer"])] + wrong, snippet=snip if isinstance(snip, list) else [],
                distance=int(r["distance_from_related_snippet_to_query_32k"]), updated=bool(r["updated"]),
                pref_type=str(r["pref_type"]), preference=str(r["preference"]))

train_df = load_dataset(DATA_REPO, "benchmark", split="train_text").to_pandas()
test_df = load_dataset(DATA_REPO, "benchmark", split="benchmark_text").to_pandas()

rng = random.Random(SEED)
train_links = sorted(train_df.chat_history_32k_link.unique()); rng.shuffle(train_links)
test_links = sorted(test_df.chat_history_32k_link.unique()); rng.shuffle(test_links)
train_links, test_links = train_links[:N_TRAIN_HIST], test_links[:N_TEST_HIST]

train_q = [to_question(r) for _, r in train_df[train_df.chat_history_32k_link.isin(train_links)].iterrows()]
test_q = [to_question(r) for _, r in test_df[test_df.chat_history_32k_link.isin(test_links)].iterrows()]
rng.shuffle(test_q); test_q = test_q[:N_EVAL]

overlap = set(q["persona"] for q in train_q) & set(q["persona"] for q in test_q)
print(f"train: {len(train_q)} questions / {len(train_links)} users | eval: {len(test_q)} questions / {len(set(q['link'] for q in test_q))} users")
print("persona overlap train/eval:", len(overlap), "(0 means eval users are unseen)")
print("example:", test_q[0]["query"], "->", test_q[0]["options"][0][:120])

# %% [markdown]
# ## 3. Build the memory store: chunk once, encode once
# **Important:** each history file starts with a *system* message that contains the full persona. We drop it, otherwise every method could read the answer directly.

# %%
local = snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=sorted(set(train_links + test_links)),
                          local_dir=os.path.join(CACHE_DIR, "personamem"))

def msg_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c)

def load_history(link):
    d = json.load(open(os.path.join(local, link), encoding="utf-8"))
    msgs = d["chat_history"] if isinstance(d, dict) else d
    return [m for m in msgs if m.get("role") != "system"]

def chunk_history(msgs, tok, max_tokens):
    chunks, cur, cur_len = [], [], 0
    for m in msgs:
        line = f"{m.get('role', 'user').capitalize()}: {msg_text(m)}"
        ids = tok(line, add_special_tokens=False).input_ids
        pieces = [line] if len(ids) <= max_tokens else [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]
        for p in pieces:
            n = len(tok(p, add_special_tokens=False).input_ids) if len(pieces) > 1 else len(ids)
            if cur and cur_len + n > max_tokens:
                chunks.append("\n".join(cur)); cur, cur_len = [], 0
            cur.append(p); cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def find_gold_chunk(chunks_norm, snippet):
    # use a distinctive piece of the longest user message in the related snippet
    texts = sorted((msg_text(m) for m in snippet if m.get("role") == "user"), key=len, reverse=True)
    for t in texts[:2]:
        t = norm(t)
        for start in (len(t) // 3, 0, len(t) // 2):
            key = t[start:start + 60]
            if len(key) < 25:
                continue
            hits = [i for i, c in enumerate(chunks_norm) if key in c]
            if hits:
                return hits[0]
    return -1

# %%
enc_tok = AutoTokenizer.from_pretrained(ENC_NAME, padding_side="left")
enc = AutoModel.from_pretrained(ENC_NAME, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
QUERY_INSTR = ("Instruct: Given a user's new request to their AI assistant, retrieve past conversation "
               "snippets that reveal the user's relevant background or preferences\nQuery: ")

@torch.no_grad()
def embed(texts, bs=32, max_len=CHUNK_TOKENS + 64):
    order = np.argsort([-len(t) for t in texts])
    out = [None] * len(texts)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        b = enc_tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        h = F.normalize(enc(**b).last_hidden_state[:, -1].float(), dim=-1).cpu()   # last-token pooling
        for j, v in zip(idx, h):
            out[j] = v
    return torch.stack(out)

store = {}   # link -> dict(chunks, norm, emb)
t0 = time.time()
for n, link in enumerate(sorted(set(train_links + test_links))):
    path = os.path.join(CACHE_DIR, "emb", link.replace("/", "_") + f".{CHUNK_TOKENS}.pt")
    if os.path.exists(path):
        store[link] = torch.load(path)
        continue
    chunks = chunk_history(load_history(link), enc_tok, CHUNK_TOKENS)
    store[link] = dict(chunks=chunks, norm=[norm(c) for c in chunks], emb=embed(chunks).half())
    os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(store[link], path)
    if n % 25 == 0:
        print(f"{n} histories encoded, {time.time() - t0:.0f}s")

for q in train_q + test_q:
    q["qemb"] = None
qembs = embed([QUERY_INSTR + q["query"] for q in train_q + test_q], bs=64)
for q, e in zip(train_q + test_q, qembs):
    q["qemb"] = e
for q in test_q:
    q["gold"] = find_gold_chunk(store[q["link"]]["norm"], q["snippet"])

n_chunks = [len(s["chunks"]) for s in store.values()]
print(f"store: {len(store)} users, {np.mean(n_chunks):.0f} chunks/user avg, gold chunk found for "
      f"{np.mean([q['gold'] >= 0 for q in test_q]):.0%} of eval questions")
H_ENC = qembs.shape[1]
del enc; torch.cuda.empty_cache() if DEV == "cuda" else None

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
HEADER = "Below is your past conversation history with this user. Use it to personalize your answer.\n\n"

def chat_ids(query, system=None):
    q_ids = tok(query, add_special_tokens=False).input_ids[:MAX_Q_TOKENS]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": tok.decode(q_ids)}]
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
    L = prefix_embeds.shape[1]
    with amp():
        out = llm(inputs_embeds=prefix_embeds, use_cache=True, logits_to_keep=1)
    cache, first = out.past_key_values, out.logits[:, -1].float()
    scores = []
    for o in options:
        ids = ans_ids(o).to(DEV)[None]
        with amp():
            r = llm(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = torch.cat([first[:, None], r.logits[:, :-1].float()], 1)
        scores.append(logits.log_softmax(-1).gather(-1, ids[..., None]).mean().item())
        cache.crop(-ids.shape[1])   # drop this option's tokens, keep the prefix
    return scores

def history_text(link, max_tokens):
    ids = tok("\n".join(store[link]["chunks"]), add_special_tokens=False).input_ids
    return tok.decode(ids[-max_tokens:]), min(len(ids), max_tokens)   # keep the most recent tokens

def rag_text(q, k=RAG_K):
    s = store[q["link"]]
    top = torch.topk(s["emb"].float() @ q["qemb"], min(k, len(s["chunks"]))).indices.tolist()
    text = "\n...\n".join(s["chunks"][i] for i in sorted(top))   # keep chronological order
    return text, len(tok(text, add_special_tokens=False).input_ids), top

print("LLM hidden:", H_LLM, "| encoder dim:", H_ENC, "| token-embedding std:", round(EMB_STD, 4))

# %% [markdown]
# ## 5. Memorilla-mini: the memory module
# Same 4 stages as the paper (Sec. 2.2):
# 1. **K learned slots** `Q0`.
# 2. **Query conditioning:** `Qc = LN(Q0 + CrossAttn(Q0, z_q))`.
# 3. **Read all chunks:** `N_BLOCKS` × (cross-attention over every chunk embedding + FFN).
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

def batch_store(links):
    embs = [store[l]["emb"].float() for l in links]
    N = max(e.shape[0] for e in embs)
    Z = torch.zeros(len(embs), N, H_ENC)
    mask = torch.ones(len(embs), N, dtype=torch.bool)
    for i, e in enumerate(embs):
        Z[i, :len(e)], mask[i, :len(e)] = e, False
    return Z.to(DEV), mask.to(DEV)

mem = MemoryModule(H_ENC, H_LLM, K_SLOTS, N_BLOCKS, N_HEADS, EMB_STD).to(DEV)
print(f"trainable params: {sum(p.numel() for p in mem.parameters()) / 1e6:.1f}M (LLM frozen: {sum(p.numel() for p in llm.parameters()) / 1e6:.0f}M)")

# %% [markdown]
# ## 6. Train
# Loss = multiple-choice cross-entropy over the 4 options' log-likelihoods + `NLL_WEIGHT` × the paper's answer NLL. Gradients flow through the frozen LLM into the soft tokens.

# %%
def option_logprobs(M, qs):
    """Differentiable mean log-prob of every option, given soft tokens M [B,K,H]. Returns [B, n_opts]."""
    seqs, targets = [], []
    for i, q in enumerate(qs):
        prefix = torch.cat([M[i], embed_ids(chat_ids(q["query"])).to(M.dtype)], 0)
        for o in q["options"]:
            a = ans_ids(o).to(DEV)
            seqs.append((torch.cat([prefix, embed_ids(a).to(M.dtype)], 0), prefix.shape[0], a))
    T = max(s[0].shape[0] for s in seqs)
    X = torch.zeros(len(seqs), T, H_LLM, device=DEV, dtype=M.dtype)
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
    return per_seq.view(len(qs), -1)

def memory_for(qs, links=None):
    Z, mask = batch_store(links or [q["link"] for q in qs])
    zq = torch.stack([q["qemb"] for q in qs]).to(DEV)
    return mem(Z, mask, zq)

opt = torch.optim.AdamW(mem.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(train_q) / BATCH)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda" and not BF16)
history, step, t0 = [], 0, time.time()
mem.train()
for ep in range(EPOCHS):
    order = list(range(len(train_q))); random.shuffle(order)
    for b in range(0, len(order), BATCH):
        qs = [train_q[i] for i in order[b:b + BATCH]]
        M, _ = memory_for(qs)
        s = option_logprobs(M, qs)
        loss = F.cross_entropy(s / TAU, torch.zeros(len(qs), dtype=torch.long, device=DEV)) + NLL_WEIGHT * (-s[:, 0].mean())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        history.append(dict(step=step, loss=loss.item(), acc=(s.argmax(1) == 0).float().mean().item()))
        if step % 50 == 0 or step == steps:
            h = pd.DataFrame(history[-50:])
            print(f"ep {ep} step {step}/{steps} loss {h.loss.mean():.3f} train-acc {h.acc.mean():.1%} {time.time() - t0:.0f}s")
mem.eval()
torch.save(mem.state_dict(), os.path.join(OUT_DIR, "memory_module.pt"))

# %% [markdown]
# ## 7. Evaluate all methods on held-out users

# %%
def timed(fn):
    if DEV == "cuda": torch.cuda.synchronize()
    t = time.time(); r = fn()
    if DEV == "cuda": torch.cuda.synchronize()
    return r, time.time() - t

rows = []
other = {l: test_links[(i + 1) % len(test_links)] for i, l in enumerate(test_links)}   # a different user
for n, q in enumerate(test_q):
    r = dict(i=n, query=q["query"], distance=q["distance"], updated=q["updated"], pref_type=q["pref_type"], gold=q["gold"],
             n_chunks=len(store[q["link"]]["chunks"]))
    def run(name, prefix, mem_tokens):
        sc, sec = timed(lambda: score_options(prefix, q["options"]))
        r[name], r[name + "_pick"], r[name + "_tok"], r[name + "_sec"] = int(np.argmax(sc) == 0), int(np.argmax(sc)), mem_tokens, sec
    with torch.no_grad():
        run("closed_book", embed_ids(chat_ids(q["query"]))[None], 0)
        text, ntok, top = rag_text(q)
        run("rag", embed_ids(chat_ids(q["query"], HEADER + text))[None], ntok)
        r["rag_hit"] = int(q["gold"] in top) if q["gold"] >= 0 else np.nan
        if n < N_FULL_EVAL:
            text, ntok = history_text(q["link"], FULL_CTX_MAX_TOKENS)
            run("full", embed_ids(chat_ids(q["query"], HEADER + text))[None], ntok)
        with amp():
            M, attns = memory_for([q])
        run("memorilla", torch.cat([M.to(EMB.weight.dtype), embed_ids(chat_ids(q["query"]))[None]], 1), K_SLOTS)
        w = attns[-1][0].float().mean(0)                                 # last read block, averaged over slots
        r["mem_hit"] = int(q["gold"] in torch.topk(w, min(RAG_K, len(w))).indices.tolist()) if q["gold"] >= 0 else np.nan
        q["attn"] = torch.stack([a[0].float().cpu() for a in attns])     # [blocks, K, N] for the GIF
        with amp():
            Mw, _ = memory_for([q], links=[other[q["link"]]])
        run("memorilla_wrong_user", torch.cat([Mw.to(EMB.weight.dtype), embed_ids(chat_ids(q["query"]))[None]], 1), K_SLOTS)
    rows.append(r)
    if n % 50 == 0:
        print(n, {k: round(np.nanmean([x.get(k, np.nan) for x in rows]), 3) for k in ["closed_book", "rag", "full", "memorilla"]})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "results_per_question.csv"), index=False)

# %% [markdown]
# ## 8. Results table

# %%
METHODS = [("closed_book", "Closed-book"), ("rag", f"RAG top-{RAG_K}"), ("full", "Full context"),
           ("memorilla", f"Memorilla-mini (K={K_SLOTS})"), ("memorilla_wrong_user", "Memorilla, wrong user")]
sub = res[res.i < N_FULL_EVAL]
def ci(x):
    x = x.dropna(); p = x.mean(); return 1.96 * math.sqrt(p * (1 - p) / max(len(x), 1))
table = pd.DataFrame([dict(method=lbl,
                           acc_all=res[m].mean() if m in res else np.nan,
                           ci95=ci(res[m]) if m in res else np.nan,
                           acc_full_subset=sub[m].mean() if m in sub else np.nan,
                           memory_tokens=res[m + "_tok"].mean() if m in res else np.nan,
                           sec_per_question=res[m + "_sec"].mean() if m in res else np.nan)
                      for m, lbl in METHODS])
table.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
print(f"eval questions: {len(res)} (full context on first {len(sub)}). Random guess = 25%.")
table.style.format({"acc_all": "{:.1%}", "ci95": "±{:.1%}", "acc_full_subset": "{:.1%}", "memory_tokens": "{:,.0f}", "sec_per_question": "{:.2f}"})

# %% [markdown]
# ## 9. Charts for the post

# %%
COLORS = {"closed_book": "#888780", "rag": "#378ADD", "full": "#D85A30", "memorilla": "#7F77DD", "memorilla_wrong_user": "#B4B2A9"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# 1) accuracy vs memory tokens (on the subset where every method ran)
fig, ax = plt.subplots(figsize=(7, 4.2))
for m, lbl in METHODS[:4]:
    x, y = max(sub[m + "_tok"].mean(), 1), sub[m].mean()
    ax.scatter(x, y * 100, s=180 if m == "memorilla" else 90, color=COLORS[m], marker="*" if m == "memorilla" else "o", zorder=3)
    ax.annotate(f"{lbl}\n{y:.0%} · {sub[m + '_tok'].mean():,.0f} tok", (x, y * 100), textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.axhline(25, ls=":", color="#B4B2A9"); ax.text(1.1, 26, "random guess", fontsize=8, color="#888780")
ax.set_xscale("log"); ax.set_xlabel("memory tokens given to the LLM (log scale)"); ax.set_ylabel("accuracy (%)")
ax.set_title(f"PersonaMem-v2 · frozen {LLM_NAME.split('/')[-1]} · {len(sub)} held-out questions")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "1_accuracy_vs_tokens.png")); plt.show()

# 2) accuracy vs how far back the relevant memory is
res["dist_bin"] = pd.cut(res.distance, [-1, 8000, 16000, 24000, 1e9], labels=["0-8K", "8-16K", "16-24K", "24K+"])
show = ["closed_book", "rag", "memorilla"]
g = res.groupby("dist_bin", observed=False)[show].mean() * 100
ax = g.plot.bar(figsize=(7, 4), color=[COLORS[m] for m in show], rot=0, width=0.75)
ax.set_xlabel("tokens between the relevant past chat and the question"); ax.set_ylabel("accuracy (%)")
ax.legend([dict(METHODS)[m] for m in show], fontsize=8); ax.set_title("Does memory fade with distance?")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "2_accuracy_vs_distance.png")); plt.show()

# 3) preference updates and preference types
g = res.groupby("updated")[show].mean() * 100
ax = g.rename(index={False: "stable preference", True: "preference changed"}).plot.bar(figsize=(6, 3.6), color=[COLORS[m] for m in show], rot=0)
ax.set_ylabel("accuracy (%)"); ax.legend([dict(METHODS)[m] for m in show], fontsize=8); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "3_updates.png")); plt.show()
display(res.groupby("pref_type")[show + ["i"]].agg({**{m: "mean" for m in show}, "i": "count"}).rename(columns={"i": "n"}).round(3))

# 4) did the method look at the right past chat?
hit = res.dropna(subset=["rag_hit"])
rand = (RAG_K / hit.n_chunks).mean()
fig, ax = plt.subplots(figsize=(5.5, 3.4))
vals = [rand * 100, hit.rag_hit.mean() * 100, hit.mem_hit.mean() * 100]
ax.bar(["random", f"RAG top-{RAG_K}", f"Memorilla attention top-{RAG_K}"], vals, color=["#B4B2A9", COLORS["rag"], COLORS["memorilla"]])
for i, v in enumerate(vals): ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_ylabel("relevant chunk found (%)"); ax.set_title("Where did each method look?")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "4_found_relevant_chunk.png")); plt.show()

# %% [markdown]
# ## 10. Attention GIF
# One held-out question. Each frame is one memory slot. The bars show how much that slot reads each chunk of the user's history (left = oldest, right = newest). The amber bar is the chunk that holds the relevant past chat.

# %%
cand = res[(res.memorilla == 1) & (res.gold >= 0)]
cand = cand[cand.rag == 0] if (cand.rag == 0).any() else cand
qi = int(cand.i.iloc[0]) if len(cand) else 0
q = test_q[qi]
A = q["attn"][-1].numpy()                                   # [K, N] last read block
K, N = A.shape
fig, ax = plt.subplots(figsize=(8, 3.6))
def frame(k):
    ax.clear()
    cols = ["#EF9F27" if j == q["gold"] else "#AFA9EC" for j in range(N)]
    ax.bar(range(N), A[k] if k < K else A.mean(0), color=cols, width=1.0)
    ax.set_ylim(0, max(A.max(), 1e-6) * 1.1); ax.set_xlim(-1, N)
    ax.set_xlabel(f"history chunks (oldest -> newest, {N} x {CHUNK_TOKENS} tokens)"); ax.set_ylabel("attention")
    who = f"slot m{k + 1} of {K}" if k < K else f"all {K} slots (mean)"
    ax.set_title(f'"{q["query"][:70]}..."\n{who}  ·  amber = the relevant past chat', fontsize=9)
ani = animation.FuncAnimation(fig, frame, frames=K + 1, interval=700)
ani.save(os.path.join(OUT_DIR, "5_memory_attention.gif"), writer=animation.PillowWriter(fps=1.5))
plt.close(fig)
from IPython.display import Image
Image(filename=os.path.join(OUT_DIR, "5_memory_attention.gif"))

# %% [markdown]
# ## 11. One example, side by side

# %%
r = res[res.i == qi].iloc[0]
print("QUESTION:", q["query"]); print("HIDDEN PREFERENCE:", q["preference"]); print()
for m, lbl in METHODS:
    if m in r and not pd.isna(r[m]):
        print(f"{lbl:28s} {'RIGHT' if r[m] else 'wrong'} -> {q['options'][int(r[m + '_pick'])][:150]}")

# %% [markdown]
# ## 12. Summary for the post

# %%
t = table.set_index("method")
mm, fc, rg = f"Memorilla-mini (K={K_SLOTS})", "Full context", f"RAG top-{RAG_K}"
print(f"""On {len(sub)} held-out PersonaMem-v2 questions (4 options, random = 25%), frozen {LLM_NAME}:
  - Full context: {t.loc[fc, 'acc_full_subset']:.1%} with ~{t.loc[fc, 'memory_tokens']:,.0f} memory tokens, {t.loc[fc, 'sec_per_question']:.2f}s/question
  - {rg}: {t.loc[rg, 'acc_full_subset']:.1%} with ~{t.loc[rg, 'memory_tokens']:,.0f} tokens
  - {mm}: {t.loc[mm, 'acc_full_subset']:.1%} with {K_SLOTS} tokens ({t.loc[fc, 'memory_tokens'] / K_SLOTS:,.0f}x fewer than full context), {t.loc[mm, 'sec_per_question']:.2f}s/question
  - Sanity check, memory from the wrong user: {t.loc['Memorilla, wrong user', 'acc_all']:.1%} (vs {t.loc[mm, 'acc_all']:.1%} with the right user)
Trainable params: {sum(p.numel() for p in mem.parameters()) / 1e6:.1f}M. LLM and encoder frozen.""")
