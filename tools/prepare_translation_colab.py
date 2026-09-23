"""Generate notebooks/22_translate_packets_colab.ipynb.

    .venv/bin/python tools/prepare_translation_colab.py

The notebook translates one batch's review-packet reports on the
participant's Colab GPU with the same open-weight model that extracted the
labels (Qwen2.5-14B-Instruct-AWQ through vLLM). Report text never leaves that
session for a hosted service. It embeds src/labels/translate.py verbatim and
refuses to translate anything until the model passes the synthetic
self-check.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.make_notebook import Notebook

NOTEBOOK = ROOT / 'notebooks/22_translate_packets_colab.ipynb'
MODULE = ROOT / 'src/labels/translate.py'


def make():
    nb = Notebook(gpu=True)
    nb.md("""
# Translate review-packet reports (local model)

Translation aid for the solo report audit (label-quality plan, step 4). It runs an
open-weight model on this Colab GPU: **no report text is sent to a hosted service.**

1. Upload the batch's `*_requests.jsonl` (from `outputs/label_quality/review/translate/`)
   to `MyDrive/knee_review/`.
2. Runtime → Change runtime type → **A100 GPU**. Run every cell top to bottom.
3. The self-check (cell 6) must pass before anything is translated. It uses synthetic
   sentences, not reports.
4. Download `*_translations.jsonl` from the same Drive folder and pass it to
   `build_blinded_packets.py render --translations`.

The translation is an aid only. Every ledger quote comes from the original report.
""")
    nb.code("""
import subprocess, torch
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
except FileNotFoundError:
    print("nvidia-smi not found")
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime -> Change runtime type -> A100 GPU.")
""")
    nb.code("""
!pip install -q vllm
!pip uninstall -y -q torchaudio
import importlib.metadata
print("vllm", importlib.metadata.version("vllm"), "installed. Restart the runtime once if vllm fails to import below.")
""")
    source = MODULE.read_text(encoding='utf-8')
    nb.code(f"""
# Generated from src/labels/translate.py -- edit the module and regenerate, never here.
import importlib, pathlib, sys
TRANSLATE_SOURCE = {source!r}
pathlib.Path("translate.py").write_text(TRANSLATE_SOURCE, encoding="utf-8")
sys.path.insert(0, ".")
importlib.invalidate_caches()
import translate
""")
    nb.code("""
import hashlib, json, pathlib
from google.colab import drive
drive.mount("/content/drive")
FOLDER = pathlib.Path("/content/drive/MyDrive/knee_review")
REQUESTS = sorted(FOLDER.glob("*_requests.jsonl"))
if len(REQUESTS) != 1:
    raise SystemExit(f"expected exactly one *_requests.jsonl in {FOLDER}, found {[p.name for p in REQUESTS]}")
REQUESTS = REQUESTS[0]
OUTPUT = FOLDER / REQUESTS.name.replace("_requests.jsonl", "_translations.jsonl")
requests = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line]
print(f"{len(requests)} reports to translate from {REQUESTS.name}")
""")
    nb.code("""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
MAX_TOKENS = 2048
tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="auto", gpu_memory_utilization=0.90, max_model_len=8192,
          trust_remote_code=True)
SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)


def generate(items):
    prompts = [tokenizer.apply_chat_template(translate.messages(text, language), tokenize=False,
                                             add_generation_prompt=True) for text, language in items]
    return [out.outputs[0] for out in llm.generate(prompts, SAMPLING)]


def translate_batch(items):
    return [o.text for o in generate(items)]
""")
    nb.code("""
# Self-check on synthetic sentences: every negation, hedge, side and grade must survive.
failures = translate.self_check(translate_batch)
for f in failures:
    print(f"FAIL {f['language']}: {f['source']!r} -> {f['translation']!r}; missing {f['missing']}")
if failures:
    raise SystemExit("The model lost required terms on synthetic checks; do not use its translations.")
print(f"self-check passed on {len(translate.SELF_CHECKS)} synthetic sentences")
""")
    nb.code("""
outputs = generate([(r["report"], r["language"]) for r in requests])
results = [{"case_id": r["case_id"], "translation": translate.clean(o.text), "finish_reason": o.finish_reason,
            "model": MODEL} for r, o in zip(requests, outputs)]
problems = translate.check_results(requests, results)
if problems:
    print("\\n".join(problems))
    raise SystemExit("translation output failed its checks; nothing written")
with OUTPUT.open("w", encoding="utf-8") as handle:
    for row in results:
        handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
print(f"wrote {len(results)} translations to {OUTPUT.name}; input sha256 "
      f"{hashlib.sha256(REQUESTS.read_bytes()).hexdigest()[:16]}")
""")
    return nb


def build(path=NOTEBOOK):
    make().write(path)
    return path


if __name__ == '__main__':
    build()
