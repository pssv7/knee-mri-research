"""Exercise application plumbing with synthetic data, not medical predictions.

Run: python -m tools.smoke_knee_application --output /tmp/knee-application-smoke
"""
import argparse
import json

from src.application.pipeline import LocalResultStore, Pipeline
from src.config import ID_COL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pipeline = Pipeline(
        model_version="synthetic-smoke-v1", preprocessing_id="synthetic-v1",
        prepare=lambda request: {ID_COL: request.study_id, "volumes": [0]},
        predict=lambda prepared: [0.5] * 12, store=LocalResultStore(args.output))
    responses = pipeline.execute([{
        "schema_version": 1, "request_id": "synthetic-smoke-v1",
        ID_COL: "synthetic-study", "model_version": "synthetic-smoke-v1",
        "preprocessing_id": "synthetic-v1"}])
    print(json.dumps({"synthetic_only": True, "responses": responses}, indent=2))
    return 0 if responses[0]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
