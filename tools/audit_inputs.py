"""Measure source metadata and the protocol coverage of each input selector."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.model.dataset import select_series, series_metadata


def audit(series, maximum=5):
    if series.SeriesInstanceUID.duplicated().any():
        raise ValueError("duplicate series UIDs")
    columns = ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]
    for _, row in series.iterrows():
        series_metadata(row)
    result = {"studies": int(series[config.ID_COL].nunique()), "series": len(series),
              "missing_flags": series[columns].isna().sum().to_dict(), "policies": {}}
    for policy in ("uniform", "protocol"):
        missing_planes, missing_protocols, count = 0, 0, 0
        for _, rows in series.sort_values("SeriesInstanceUID").groupby(config.ID_COL):
            picked = rows.iloc[select_series(rows, maximum, policy)]
            count += len(picked)
            missing_planes += bool(set(rows.Anatomical_Plane) - set(picked.Anatomical_Plane))
            missing_protocols += len(rows[columns].drop_duplicates()) - len(picked[columns].drop_duplicates())
        result["policies"][policy] = {"retained_series": count,
                                      "studies_losing_a_plane": missing_planes,
                                      "omitted_protocol_combinations": missing_protocols}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="data/meta/train_series.csv")
    parser.add_argument("--out", default="data/meta/input_audit.json")
    args = parser.parse_args()
    result = audit(pd.read_csv(args.series))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
