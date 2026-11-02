"""Build an .ipynb from a notebook source file, then smoke-test every code cell on CPU with tiny stub models + fake data.

Usage: python dev/build_and_smoke_test.py OUT.ipynb [--src dev/factkg_src.py] [--build-only]
"""
import json, os, random, sys, re, types
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else os.path.join(HERE, "notebook_src.py")
OUT = sys.argv[1]
FACTKG = "factkg" in os.path.basename(SRC).lower()


def parse_cells(src):
    cells = []
    for block in re.split(r"^# %%", src, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        if head.strip() == "[markdown]":
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
            cells.append(("md", md))
        else:
            cells.append(("code", body.strip("\n")))
    return cells


cells = parse_cells(open(SRC, encoding="utf-8").read())
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "cells:", len(nb.cells))
if "--build-only" in sys.argv:
    sys.exit(0)

# ---------------- smoke test ----------------
import torch, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

work = os.path.join(os.environ.get("TEMP", "/tmp"), "lum_smoke")
os.makedirs(work, exist_ok=True); os.chdir(work)
rnd = random.Random(1)
WORDS = "yoga budget medication garden coffee email teacher student stress travel music dog cat running vegan fish family work".split()

def sent(n):
    return " ".join(rnd.choice(WORDS) for _ in range(n)).capitalize() + "."

# fake PersonaMem-v2 histories + rows
os.makedirs("pmdata/data/chat_history_32k", exist_ok=True)
def make_split(prefix, users, per_user):
    rows = []
    for u in users:
        link = f"data/chat_history_32k/{prefix}_persona{u}.json"
        msgs = [{"role": "system", "content": "PERSONA SECRET"}]
        for t in range(24):
            msgs.append({"role": "user", "content": sent(rnd.randint(8, 60)) + f" unique marker {u} {t} " + sent(10)})
            msgs.append({"role": "assistant", "content": sent(rnd.randint(8, 40))})
        json.dump({"metadata": {}, "chat_history": msgs}, open(os.path.join("pmdata", link), "w"))
        for j in range(per_user):
            t = rnd.randint(0, 23)
            snip = [m for m in msgs[1 + 2 * t: 3 + 2 * t]]
            rows.append(dict(persona_id=u, chat_history_32k_link=link,
                             user_query=str({"role": "user", "content": sent(8) + "?"}),
                             correct_answer=sent(15), incorrect_answers=json.dumps([sent(15) for _ in range(3)]),
                             related_conversation_snippet=json.dumps(snip), distance_from_related_snippet_to_query_32k=rnd.randint(0, 30000),
                             updated=rnd.random() < 0.3, pref_type=rnd.choice(["a", "b"]), preference=sent(6)))
    return pd.DataFrame(rows)

def make_factkg(n):
    rows = []
    for i in range(n):
        ents = [f"Entity{rnd.randint(0, 30)}" for _ in range(rnd.randint(1, 3))]
        docs = [f"{rnd.choice(ents)} → {rnd.choice(WORDS)} → {rnd.choice(WORDS)} {rnd.randint(1, 99)}" for _ in range(rnd.randint(1, 40))]
        rt = rnd.choice(["num1", "num2|substitution", "num3", "existence", "negation|num2", "num4"])
        rows.append(dict(question=f"Is the following claim true or false? Claim: {ents[0]} {sent(6)}", documents=np.array(docs, dtype=object),
                         correct_answer=rnd.choice(["True", "False"]), reasoning_type=rt, style=rnd.choice(["written", "colloquial"]),
                         entities=np.array(ents, dtype=object)))
    return pd.DataFrame(rows)

if FACTKG:
    dfs = {"train": make_factkg(40), "test": make_factkg(15)}
else:
    dfs = {"train_text": make_split("tr", range(5), 6), "benchmark_text": make_split("te", range(100, 103), 6)}

# tiny byte-level BPE tokenizer with a chat template
corpus = [sent(30) for _ in range(300)] + ["User: Assistant: System: unique marker ? . , 0123456789"]
bt = Tokenizer(models.BPE()); bt.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); bt.decoder = decoders.ByteLevel()
bt.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<pad>", "<eos>", "<|im_start|>", "<|im_end|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
def make_tok(padding_side="right"):
    t = PreTrainedTokenizerFast(tokenizer_object=bt, pad_token="<pad>", eos_token="<eos>", padding_side=padding_side)
    t.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                       "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return t

V = bt.get_vocab_size()
def cfg(h):
    return Qwen3Config(vocab_size=V, hidden_size=h, intermediate_size=2 * h, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=h // 4, max_position_embeddings=8192, tie_word_embeddings=True)

class StubAutoTok:
    @staticmethod
    def from_pretrained(name, padding_side="right", **kw): return make_tok(padding_side)
class StubAutoModel:
    @staticmethod
    def from_pretrained(name, **kw): torch.manual_seed(0); return Qwen3Model(cfg(32))
class StubCausal:
    @staticmethod
    def from_pretrained(name, **kw): torch.manual_seed(0); return Qwen3ForCausalLM(cfg(48))
class DS:
    def __init__(self, df): self.df = df
    def to_pandas(self): return self.df.copy()

def stub_load_dataset(repo, config=None, split=None): return DS(dfs[split])
def stub_snapshot(repo, repo_type, allow_patterns, local_dir):
    assert all(os.path.exists(os.path.join("pmdata", p)) for p in allow_patterns); return os.path.abspath("pmdata")

ns = {"display": lambda *a, **k: print("[display]", type(a[0]).__name__), "Image": lambda filename: print("[gif]", filename)}
code_cells = [c for k, c in cells if k == "code"]
for idx, code in enumerate(code_cells):
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("!", "from IPython")))
    if not code.strip():
        continue
    print(f"--- cell {idx}: {code.strip().splitlines()[0][:70]}")
    exec(compile(code, f"cell{idx}", "exec"), ns)
    if "LLM_NAME =" in code and FACTKG:   # shrink config for CPU
        ns.update(N_TRAIN=24, N_EVAL=10, K_SLOTS=4, N_HEADS=4, RAG_KS=[2, 5], EPOCHS=2, CACHE_DIR="cache_fkg", OUT_DIR="out_fkg")
    elif "LLM_NAME =" in code:
        ns.update(N_TRAIN_HIST=5, N_TEST_HIST=3, N_EVAL=12, N_FULL_EVAL=6, CHUNK_TOKENS=64, K_SLOTS=4, N_HEADS=4,
                  FULL_CTX_MAX_TOKENS=1500, MAX_ANS_TOKENS=24, EPOCHS=2, CACHE_DIR="cache", OUT_DIR="out")
    if "from transformers import" in code:
        ns.update(load_dataset=stub_load_dataset, snapshot_download=stub_snapshot, AutoTokenizer=StubAutoTok,
                  AutoModel=StubAutoModel, AutoModelForCausalLM=StubCausal)

print("\nSMOKE OK. outputs:", sorted(os.listdir(ns["OUT_DIR"])))
print(ns["table"].to_string())
