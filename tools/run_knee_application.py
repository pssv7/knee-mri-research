"""Run configured local inference, preserving a receipt and per-study results."""
import argparse
import json

from src.application.runtime import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--studies", required=True, help="JSON array of study IDs")
    parser.add_argument("--output", required=True, help="New run directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with open(args.studies) as handle:
        studies = json.load(handle)
    if not isinstance(studies, list):
        parser.error("--studies must contain a JSON array")
    receipt = run_batch(args.config, studies, args.output, args.device)
    print(json.dumps({key: receipt[key] for key in ("status", "completed", "failed")}))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
