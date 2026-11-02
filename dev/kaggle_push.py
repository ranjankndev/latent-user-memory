"""Push the notebook to Kaggle as a private GPU job.

Usage:
    python dev/kaggle_push.py --user <kaggle-username> [--notebook latent_kb_memory_factkg.ipynb --slug latent-kb-memory] [--quick]

Needs the Kaggle CLI signed in (`kaggle kernels list --mine` should work).
--quick sets QUICK_RUN = True in the pushed copy; the repo notebook is not changed.
"""
import argparse, json, os, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--notebook", default="latent_user_memory.ipynb")
ap.add_argument("--slug", default="latent-user-memory")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--timeout", type=int, default=6 * 3600, help="max run seconds")
args = ap.parse_args()

NB = os.path.join(ROOT, args.notebook)
nb = json.load(open(NB, encoding="utf-8"))
hits = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "QUICK_RUN = False" in src:
            cell["source"] = src.replace("QUICK_RUN = False", f"QUICK_RUN = {args.quick}")
            hits += 1
assert hits == 1, "QUICK_RUN flag not found exactly once"

out = tempfile.mkdtemp(prefix="kaggle_push_")
json.dump(nb, open(os.path.join(out, args.notebook), "w", encoding="utf-8"), indent=1)
meta = {
    "id": f"{args.user}/{args.slug}",
    "title": args.slug,
    "code_file": args.notebook,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
print(f"pushing {meta['id']} (quick={args.quick}) from {out}")
subprocess.run(["kaggle", "kernels", "push", "-p", out, "--accelerator", "NvidiaTeslaT4", "-t", str(args.timeout)], check=True)
shutil.rmtree(out, ignore_errors=True)
