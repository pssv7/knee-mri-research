"""Execute the generated notebook's cells locally, stubbing only the GPU parts.

Static parsing missed a real bug once already -- every cell had collapsed to a
single line and still parsed, because the check joined the source list the same
wrong way the generator wrote it. So this runs the cells instead.

    python3 tools/check_notebook.py

Cells 2, 10 run against stubbed torch/vllm/transformers; cells 6, 8, 12, 16 run
for real against data/meta/train.csv. Cell 4 (pip install) and the real model
load cannot be verified without a GPU and are reported as such.
"""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NOTEBOOK = os.path.join(ROOT, "notebooks", "03_extract_labels.ipynb")

GPU_TIERS = [("A100-80GB", 80), ("A100-40GB", 42), ("L4", 24), ("T4", 15)]

# The notebook extracts probabilities now, so the stub answers in that shape.
# MCL is present but low and Fracture is omitted, so the assertions can tell
# "parsed a low value" apart from "failed to parse".
STUB_RESPONSE = '{"ACL": 92, "MCL": 8}'


def load_code_cells():
    with open(NOTEBOOK) as handle:
        notebook = json.load(handle)
    return {
        index: "".join(cell["source"])
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    }


def stub_torch(total_gb, available=True):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_properties=lambda index: types.SimpleNamespace(
            total_memory=total_gb * 1e9
        ),
    )
    sys.modules["torch"] = module


def stub_vllm():
    vllm = types.ModuleType("vllm")

    class StubLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling):
            sample = type("Sample", (), {"text": STUB_RESPONSE})
            return [
                type("Result", (), {"outputs": [sample() for _ in range(3)]})()
                for _ in prompts
            ]

    vllm.LLM = StubLLM
    vllm.SamplingParams = lambda **kwargs: kwargs
    sys.modules["vllm"] = vllm

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name: types.SimpleNamespace(
            apply_chat_template=lambda messages, tokenize, add_generation_prompt:
                messages[1]["content"]
        )
    )
    sys.modules["transformers"] = transformers


def stub_colab():
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(
        upload=lambda: {},
        download=lambda path: print(f"      [stub] download {path}"),
    )
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab


def check_gpu_cell(cells):
    print("cell 2 -- GPU detection and model selection")
    for name, memory in GPU_TIERS:
        stub_torch(memory)
        namespace = {}
        exec(cells[2], namespace)  # noqa: S102
        print(f"      {name:10s} -> {namespace['MODEL']}")

    stub_torch(0, available=False)
    try:
        exec(cells[2], {})  # noqa: S102
    except SystemExit as exc:
        print(f"      {'no GPU':10s} -> aborts: {exc}")
        return
    raise AssertionError("no-GPU path did not abort")


def check_logic_cells(cells):
    from src.labels import lexicon

    stub_colab()
    namespace = {"__name__": "__main__"}

    print("cell 6 -- data loading (real train.csv)")
    source = cells[6].replace(
        'for candidate in ["/content/drive/MyDrive/knee/train.csv", "/content/train.csv"]:',
        'for candidate in ["data/meta/train.csv"]:',
    )
    exec(source, namespace)  # noqa: S102

    print("cell 8 -- extractor module")
    exec(cells[8], namespace)  # noqa: S102

    # Stand in for the model by driving the notebook's own aggregation path
    # with lexicon output mapped to probabilities, so cells 12 and 16 handle
    # objects of the real shape.
    def fake_run(reports, desc="", variant="baseline", n_samples=3):
        results = []
        for report in reports:
            extracted = lexicon.extract_report(str(report))
            sample = {}
            for name, value in extracted.items():
                if value["present"] == 1:
                    sample[name] = 0.9
                elif value["present"] == 0:
                    sample[name] = 0.1
                else:
                    sample[name] = None
            results.append(namespace["aggregate_probabilities"]([sample] * 3))
        return results

    namespace["run_extraction"] = fake_run
    import numpy
    namespace["np"] = numpy

    print("cell 12 -- gold checkpoint and AUC table")
    exec(cells[12], namespace)  # noqa: S102

    print("cell 16 -- frame build and save")
    namespace["mined"] = fake_run(namespace["unlabelled"]["Report"].tolist()[:50])
    namespace["unlabelled"] = namespace["unlabelled"].head(50)
    exec(cells[16], namespace)  # noqa: S102


def check_model_cell(cells):
    print("cell 10 -- model load and extraction wiring (stubbed vllm)")
    stub_vllm()
    namespace = {"MODEL": "stub"}
    exec(cells[8], namespace)  # noqa: S102
    exec(cells[10], namespace)  # noqa: S102
    results = namespace["run_extraction"](["Rotura del menisco interno."], "(stub)")
    assert results[0]["ACL"]["probability"] > 0.5
    assert results[0]["MCL"]["probability"] < 0.5
    # An omitted finding must stay None, not default to anything.
    assert results[0]["Fracture"]["probability"] is None
    print(f"      n_samples={namespace['N_SAMPLES']} "
          f"max_model_len={namespace['llm'].kwargs.get('max_model_len')}")


def main():
    os.chdir(ROOT)
    cells = load_code_cells()

    check_gpu_cell(cells)
    check_model_cell(cells)
    check_logic_cells(cells)

    print("\nnot verifiable without a GPU:")
    print("      cell 4  -- pip install vllm")
    print("      cell 10 -- real model weights load")
    print("      cell 14 -- full corpus generation")
    print("\nall locally verifiable cells OK")


if __name__ == "__main__":
    main()
