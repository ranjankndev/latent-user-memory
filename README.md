# latent-user-memory

**Give a frozen LLM a whole memory in 16 tokens.**

A small, runnable reproduction of the idea in
**MEMORILLA: Latent Semantic Memory for LLMs** (Fallahpour, Idehpour, Kothapalli et al., Stanford, [arXiv:2610.21375](https://arxiv.org/abs/2610.21375), [project page](https://memory.stanford.edu/)).
It is an independent, scaled-down implementation, not the authors' code.

## How it works

1. **Store once.** Every document (a fact, a chat chunk) is turned into one vector by a frozen encoder (`Qwen3-Embedding-0.6B`), one time.
2. **Ask.** The new question is encoded into a query vector. It tunes **K = 16 learned slots**.
3. **Read.** Each slot reads *all* stored vectors with cross-attention and keeps what matters for this question.
4. **Answer.** The 16 slots are projected into the LLM's token-embedding space and put in front of the question. The LLM (`Qwen3-0.6B`) stays frozen; only the small memory module (~30M params) trains.

## Experiments

| Notebook | Use case | Dataset | Status |
|---|---|---|---|
| [`latent_kb_memory_factkg.ipynb`](latent_kb_memory_factkg.ipynb) | **Checking claims against a knowledge base** (support, sales, compliance) | [FactKG](https://huggingface.co/datasets/MemoryAsModality/FactKG): 86K train / 9K test claims, each with up to 200 knowledge-graph facts | main experiment, results below |
| [`latent_user_memory.ipynb`](latent_user_memory.ipynb) | Assistant memory of a user's ~32K-token chat history | [PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) (CC-BY-4.0) | negative result at this scale (below) |

### FactKG: knowledge-base claim checking

A claim must be judged **True or False** from the claim's own knowledge-graph facts (for example `Paul Boateng → successor → Dawn Butler`). Many claims combine 2–4 facts. This is where the paper reports its largest gain (84.3% vs 80.4% for full context, with 108x fewer tokens).

| Method | Memory given to the frozen LLM |
|---|---|
| Closed-book | nothing |
| RAG top-5 / top-20 | the most similar facts, as text |
| Full context | all of the claim's facts, as text (~2K tokens) |
| Latent memory | 16 soft tokens |
| Latent memory, wrong facts | 16 soft tokens from another claim's facts (sanity check) |

Outputs (`latent_kb_out/`): `1_accuracy_vs_tokens.png`, `2_accuracy_vs_facts_combined.png` (the paper's Fig. 4c question), `3_accuracy_vs_kb_size.png`, `4_memory_attention.gif` (each slot's most-read facts), `results_table.csv`, `results_per_claim.csv`, `memory_module.pt`.

Training adds a training-only auxiliary loss: the LLM must restate the claim's most similar fact from the memory tokens, so the slots learn to carry fact content (in the spirit of the paper's reconstruction pre-training).

**Results** (full run: 30K training claims, 1,000 held-out test claims, frozen Qwen3-0.6B on a Kaggle T4; files in [`results/factkg/`](results/factkg)):

| Method | Accuracy (±95% CI) | Memory tokens | Seconds per claim |
|---|---|---|---|
| Closed-book | 50.9% ± 3.1 | 0 | 0.17 |
| RAG top-5 | 65.0% ± 3.0 | ~68 | 0.17 |
| RAG top-20 | 67.2% ± 2.9 | ~258 | 0.18 |
| Full context | 64.7% ± 3.0 | ~1,694 | 0.53 |
| **Latent memory** | **65.9% ± 2.9** | **16** | **0.17** |
| Latent memory, wrong facts | 59.7% ± 3.0 | 16 | 0.17 |

- Latent memory matches full context and RAG (paired tests: p = 0.58 vs full context, 0.52 vs RAG top-20) with **106x fewer tokens than full context**.
- It uses the facts: it beats the same module given another claim's facts by 6.2 points (p = 0.0001).
- As claims combine more facts, text methods drop (RAG top-5: 75% at 1 fact → 53% at 3), while latent memory stays at 62–66%. On claims with 151–200 facts it is the best method (64.0% vs 60.6% full context, n = 505).
- The two are complementary: 16.4% of claims are solved only by latent memory and 17.7% only by RAG top-20; either one is right on 83.6%.

Caveats: the memory module is trained on FactKG, while the text baselines are zero-shot; the wrong-facts control (59.7%) shows that about 9 of its 15 points over closed-book come from what the module learned about claims in general, and about 6 from the facts. Per-bucket numbers rest on 115–505 claims each (±4–9 points). All methods lean towards answering False with this small LLM.

### PersonaMem-v2: what happened

Quick run (80 training users, 100 questions from 10 unseen users, frozen Qwen3-0.6B): every method stayed near chance (25%). Closed-book 27%, RAG top-3 24%, full context 27% (15 questions), latent memory 31%, and latent memory built from the *wrong* user also 31%. The module learned on training users (train accuracy 25% → ~40%, preference loss 4.6 → 1.4) but did not carry user-specific information to new users, and the 0.6B LLM could not use the history even as text. The paper reports that all baselines on this benchmark sit near chance even with an 8B model. One finding carried over: full context took **20.1 s per question** vs **0.29 s** with 16 memory tokens on the same T4.

Details of that notebook: the system message of each history file contains the full persona and is removed; options are scored with PMI calibration because they differ only in a small personal detail.

## Run it

**Kaggle (keeps running with your laptop off)**
1. Import a notebook (File → Import Notebook).
2. Settings: **Accelerator: GPU T4**, **Internet: On** (needs a phone-verified account).
3. Optional: set `QUICK_RUN = True` in the config cell for a short end-to-end check first.
4. **Save Version → Save & Run All (Commit)**. Results appear in the Output tab.

**Kaggle from the command line** (Kaggle CLI signed in):

```bash
python dev/kaggle_push.py --user <kaggle-username> --notebook latent_kb_memory_factkg.ipynb --slug latent-kb-memory --quick
```

Drop `--quick` for the full run. `kaggle kernels output <user>/latent-kb-memory -p kaggle_out` downloads the results.

**Colab**: open the notebook, Runtime → GPU, Run all.

No account or token is needed: the Qwen models and the datasets are public. A Hugging Face token (secret named `HF_TOKEN` on Kaggle or Colab) is used automatically if present, only to avoid download rate limits.

## Limits

- Scaled down from the paper: 0.6B encoder + 0.6B LLM (paper: 4B + 8B), one training stage (paper: three). Numbers will differ from the paper.
- PersonaMem-v2 is synthetic (LLM-generated personas and chats).
- Latent memory needs an open-weight LLM (it injects vectors, not text) and a module trained per LLM.
- It tends to lose exact details (names, dates, numbers), where retrieval is stronger; in practice the two work together.

## Development

Each notebook is generated from a source file in `dev/`:

```bash
python dev/build_and_smoke_test.py latent_kb_memory_factkg.ipynb --src dev/factkg_src.py
python dev/build_and_smoke_test.py latent_user_memory.ipynb
```

This rebuilds the notebook and runs every cell on CPU with tiny random Qwen3 models and fake data shaped like the real dataset, to catch errors before a GPU run. Add `--build-only` to skip the test. On Windows set `PYTHONIOENCODING=utf-8` (the FactKG facts contain `→`).
