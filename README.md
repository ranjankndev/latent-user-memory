# latent-user-memory

**Your entire chat history, remembered in 16 tokens.**

An AI assistant has ~32,000 tokens of past chats with a user. A new request comes in. Can a *frozen* LLM still give the personalized answer if it only gets **16 learned memory tokens** instead of the whole history?

This repo is a small, runnable reproduction of the idea in
**MEMORILLA: Latent Semantic Memory for LLMs** (Fallahpour, Idehpour, Kothapalli et al., Stanford, [arXiv:2610.21375](https://arxiv.org/abs/2610.21375), [project page](https://memory.stanford.edu/)),
on the **PersonaMem-v2** benchmark ([dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), CC-BY-4.0).
It is an independent, scaled-down implementation, not the authors' code.

## How it works

1. **Store once.** Each user's chat history is cut into 256-token chunks. A frozen encoder (`Qwen3-Embedding-0.6B`) turns each chunk into one vector, one time.
2. **Ask.** A new request is encoded into a query vector. It tunes **K = 16 learned slots**.
3. **Read.** Each slot reads *all* the user's chunk vectors with cross-attention and keeps what matters for this request.
4. **Answer.** The 16 slots are projected into the LLM's token-embedding space and put in front of the question. The LLM (`Qwen3-0.6B`) stays frozen; only the small memory module (~25M params) trains.

## What it measures (on held-out users)

| Method | Memory given to the frozen LLM |
|---|---|
| Closed-book | nothing |
| RAG top-3 | 3 most similar chunks as text (~800 tokens) |
| Full context | the whole history as text (~30K tokens) |
| Latent memory (Memorilla-mini) | 16 soft tokens |
| Latent memory, wrong user | 16 soft tokens from another user's history (sanity check) |

Each question has 1 correct and 3 incorrect personalized answers that differ only in a small personal detail. Each method scores an option by how much its memory raises the option's mean log-likelihood compared with no memory (PMI calibration); the highest wins (random = 25%). The same calibrated score is used in the training loss.

## Run it

**Kaggle (keeps running with your laptop off)**
1. Import `latent_user_memory.ipynb` (File → Import Notebook).
2. Settings: **Accelerator: GPU T4**, **Internet: On** (needs a phone-verified account).
3. Optional: set `QUICK_RUN = True` in the config cell for a ~30 min check first.
4. **Save Version → Save & Run All (Commit)**. Results appear in the Output tab under `memorilla_out/`.

**Kaggle from the command line**: with the Kaggle CLI signed in, `python dev/kaggle_push.py --user <kaggle-username> --quick` pushes a private T4 job (drop `--quick` for the full run). `kaggle kernels output <user>/latent-user-memory -p kaggle_out` downloads the results.

**Colab**: open the notebook, Runtime → GPU, Run all. Point `CACHE_DIR` at Google Drive to keep embeddings across disconnects.

Default config: 400 training users, 100 held-out users, 500 eval questions (150 with the full-context baseline). About 1.5–2.5 h on a T4.

No account or token is needed: the Qwen models and the dataset are public. A Hugging Face token (secret named `HF_TOKEN` on Kaggle or Colab) is used automatically if present, only to avoid download rate limits.

## Outputs (`memorilla_out/`)

| File | Shows |
|---|---|
| `1_accuracy_vs_tokens.png` | accuracy vs memory tokens per method |
| `2_accuracy_vs_distance.png` | accuracy vs how far back the relevant chat is |
| `3_updates.png` | stable vs changed user preferences |
| `4_found_relevant_chunk.png` | did RAG / the memory slots look at the right past chat? |
| `5_memory_attention.gif` | each memory slot reading the user's history |
| `results_table.csv`, `results_per_question.csv` | raw numbers |
| `memory_module.pt` | trained memory module weights |

## Notes and limits

- Each PersonaMem-v2 history file starts with a system message that contains the full persona. The notebook removes it so no method can read the answer directly.
- Scaled down from the paper: 0.6B encoder + 0.6B LLM (paper: 4B + 8B), one training stage (paper: three), multiple-choice + NLL loss (paper: NLL only). Numbers will differ from the paper.
- PersonaMem-v2 is synthetic (LLM-generated personas and chats).
- Latent memory needs an open-weight LLM (it injects vectors, not text) and a module trained per LLM. It tends to lose exact details (names, dates, numbers), where retrieval is stronger; in practice the two work together.

## Development

The notebook is generated from `dev/notebook_src.py`:

```bash
python dev/build_and_smoke_test.py latent_user_memory.ipynb
```

This rebuilds the notebook and runs every cell on CPU with tiny random Qwen3 models and fake PersonaMem-shaped data, to catch errors before a GPU run. Add `--build-only` to skip the test.
