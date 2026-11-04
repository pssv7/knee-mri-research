"""Freeze and build an isolated training-label repair preview; never launch Kaggle."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.labels.extraction_repair import repair
from src.labels.report_candidate import request_body
from src.labels.openai_extract import request_hash
from src.validation.proxy import ID, LABELS, sha, write_json_new
from tools import fresh_report_extraction as fresh

POLICY = "configs/extraction_repair_policy.json"
BASE = "outputs/distinct_sampling_pair_v1"
SOURCES = ["src/labels/extraction_repair.py", "tools/label_repair_experiment.py", POLICY,
           "src/labels/fuse.py", "src/labels/mode_fusion.py", "src/labels/llm_extract.py",
           "src/labels/lexicon.py", "src/validation/proxy.py"]


def freeze(out, extraction):
    out, extraction = Path(out), Path(extraction)
    if out.exists():
        raise FileExistsError("Experiment plan is frozen; use a new version")
    base = ROOT / BASE
    spec = json.loads((base/"control/spec.json").read_text())
    for name, expected in spec["runtime_code_sha256"].items():
        if sha(ROOT/name) != expected: raise ValueError("Baseline runtime changed: "+name)
    for filename, key in [("roles.csv","roles_sha256"),("proxy_labels.parquet","proxy_labels_sha256")]:
        if sha(base/"control"/filename) != spec[key]: raise ValueError("Baseline input changed")
    roles = pd.read_csv(base/"control/roles.csv")
    reports = pd.read_csv(ROOT/"data/meta/train.csv").set_index(ID)
    packet = json.loads((ROOT/"outputs/report_label_evaluation_v1/reviewer/fresh/packet.json").read_text())
    checks = (ROOT/fresh.CLARIFICATIONS).read_text()
    extraction_manifest = json.loads((extraction/"manifest.json").read_text())
    frozen_fresh = {r["request_sha256"] for r in extraction_manifest["cases"]}
    rows = []
    for uid in sorted(roles.loc[roles.role=="train",ID]):
        body = request_body(reports.loc[uid,"Report"],packet["rubric"],checks)
        rows.append({ID:uid,"request_sha256":request_hash(body),"already_in_fresh_plan":request_hash(body) in frozen_fresh})
    inputs = ["data/meta/train.csv","data/meta/llm_labels.parquet","data/meta/llm_probs.parquet",
              BASE+"/control/spec.json",BASE+"/control/roles.csv",BASE+"/control/proxy_labels.parquet",
              BASE+"/exclusions.csv",BASE+"/raw_counts.parquet","outputs/report_label_evaluation_v1/custodian/fresh_key.jsonl",
              str((extraction/"manifest.json").relative_to(ROOT))]
    plan = {
        "status":"frozen_before_fresh_results_unsealed",
        "source_sha256":{p:sha(ROOT/p) for p in SOURCES},
        "inputs_sha256":{p:sha(ROOT/p) for p in inputs},
        "extraction":str(extraction.relative_to(ROOT)),
        "baseline":BASE,
        "training_studies":len(rows),"fresh_planned_training_studies":sum(r["already_in_fresh_plan"] for r in rows),
        "remaining_training_studies":sum(not r["already_in_fresh_plan"] for r in rows),
        "unique_training_requests":len({r["request_sha256"] for r in rows}),
        "policy":json.loads((ROOT/POLICY).read_text()),
        "paired_recipe":{key:spec[key] for key in ["backbone","cache_size","slices","max_series","series_selection","epochs","seed","effective_batch","learning_rate","schedule","warmup_epochs","initialization","sampling_policy"]},
        "adoption_gate":{"macro_auc_delta_at_least":.005,"per_finding_point_delta_at_least":-.02,"paired_component_bootstrap_lower_bound_above":0,"confirmation_required":True},
        "scope":"A fresh-sample preview is not a completed full-training relabel. Full-cohort inference and a matched GPU comparison remain necessary.",
        "training_launched":False,"image_accuracy_gain_established":False,
        "kaggle_owner":"user",
        "cost_estimate":{"basis":"Historical Sonnet pilot mean, excludes revised-prompt overhead and review calls",
                         "new_sonnet_cost_all_training_usd":len({r["request_sha256"] for r in rows})*1.091174/40,
                         "existing_shared_cost_before_fresh_usd":4.110458,
                         "current_cap_usd":10,"full_cohort_fits_current_cap":False}
    }
    out.mkdir(parents=True)
    (out/"training_request_inventory.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    plan["training_request_inventory_sha256"]=sha(out/"training_request_inventory.jsonl")
    write_json_new(out/"plan.json",plan)
    return {k:plan[k] for k in ["training_studies","fresh_planned_training_studies","remaining_training_studies","unique_training_requests","cost_estimate","training_launched"]}


def verify_plan(out):
    out=Path(out)
    plan=json.loads((out/"plan.json").read_text())
    for group in ["source_sha256","inputs_sha256"]:
        for path,expected in plan[group].items():
            if sha(ROOT/path)!=expected:raise ValueError("Frozen experiment input/source changed: "+path)
    if sha(out/"training_request_inventory.jsonl")!=plan["training_request_inventory_sha256"]:
        raise ValueError("Training request inventory changed")
    return plan


def build_preview(out):
    out=Path(out);plan=verify_plan(out)
    extraction=ROOT/plan["extraction"]
    manifest,packet,records=fresh.load(extraction)
    if len(records)!=len(manifest["cases"]):
        raise ValueError("Fresh inference is incomplete; do not unseal a partial cohort")
    mapping=pd.read_json(ROOT/"outputs/report_label_evaluation_v1/custodian/fresh_key.jsonl",lines=True).set_index("case_id")
    assertions={mapping.loc[r["case_id"],ID]:r["assertions"] for r in records.values() if r["status"]=="accepted"}
    base=ROOT/plan["baseline"]
    control=pd.read_parquet(base/"control/proxy_labels.parquet")
    roles=pd.read_csv(base/"control/roles.csv")
    reports=pd.read_csv(ROOT/"data/meta/train.csv").set_index(ID)
    cat=pd.read_parquet(ROOT/"data/meta/llm_labels.parquet");prob=pd.read_parquet(ROOT/"data/meta/llm_probs.parquet")
    candidate,changes,keeps=repair(control,roles,cat,prob,reports,assertions)
    target=out/"fresh_sample_preview";target.mkdir(exist_ok=False)
    candidate.to_parquet(target/"candidate_proxy_labels.parquet",index=False)
    (target/"changes.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in changes))
    (target/"kept.jsonl").write_text("".join(json.dumps(r)+"\n" for r in keeps))
    receipt={"status":"limited_fresh_sample_preview_not_full_training_relabel",
             "reports_inferred":len(records),"reports_accepted":len(assertions),
             "training_studies_total":plan["training_studies"],
             "changed_finding_slots":len(changes),"changed_studies":len({r[ID] for r in changes}),
             "candidate_sha256":sha(target/"candidate_proxy_labels.parquet"),
             "plan_sha256":sha(out/"plan.json"),
             "response_record_sha256":{p.name:sha(p) for p in sorted((extraction/"attempts").glob("*.json"))},
             "selection_outer_unchanged":True,"sample_weights_unchanged":True,
             "production_labels_changed":False,"training_launched":False}
    write_json_new(target/"receipt.json",receipt)
    return {k:v for k,v in receipt.items() if k!="response_record_sha256"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    freeze_parser=commands.add_parser("freeze")
    freeze_parser.add_argument("--out",type=Path,default=ROOT/"outputs/extraction_repair_experiment_v1")
    freeze_parser.add_argument("--extraction",type=Path,default=ROOT/"outputs/fresh_report_extraction_v1")
    build=commands.add_parser("build-preview")
    build.add_argument("--out",type=Path,default=ROOT/"outputs/extraction_repair_experiment_v1")
    args=vars(parser.parse_args());action=args.pop("action")
    print(json.dumps(freeze(**args) if action=="freeze" else build_preview(**args),indent=2))

