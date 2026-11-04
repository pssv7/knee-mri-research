"""Compare the completed fresh extraction offline; no independent accuracy claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.validation.proxy import ID,LABELS,sha,write_json_new
from tools import fresh_report_extraction as fresh


def summarize(out, result_dir):
    out,result_dir=Path(out),Path(result_dir)
    if result_dir.exists():raise FileExistsError("Preserve the previous readout")
    manifest,packet,records=fresh.load(out)
    if len(records)!=len(manifest["cases"]):raise ValueError("Incomplete cohort remains sealed")
    prior=fresh.base.carried_cost(manifest)
    key=pd.read_json(ROOT/"outputs/report_label_evaluation_v1/custodian/fresh_key.jsonl",lines=True).set_index("case_id")
    old=pd.read_json(ROOT/"outputs/report_label_evaluation_v1/custodian/fresh_old_extractor.jsonl",lines=True).set_index(["case_id","finding"])
    rows=[];rejections=[];usage=Counter();response_ids=set()
    for record in records.values():
        case=record["case_id"];mapping=key.loc[case]
        response_id=record["response"]["id"]
        if response_id in response_ids:raise ValueError("Repeated response ID")
        response_ids.add(response_id)
        for field in ["input_tokens","output_tokens","cache_creation_input_tokens","cache_read_input_tokens"]:
            usage[field]+=record["response"]["usage"].get(field,0)
        if record["status"]=="rejected":
            rejections.append({"case_id":case,ID:mapping[ID],"validation_error":record["validation_error"]})
        for finding in LABELS:
            a=record.get("assertions",{}).get(finding)
            call=old.loc[(case,finding),"old_categorical_call"]
            old_state="silent" if pd.isna(call) else "positive" if call==1 else "negative"
            new_state=a["report_states"] if a is not None else "unavailable"
            rows.append({"case_id":case,ID:mapping[ID],"finding":finding,"cohort":mapping["cohort"],
                         "language_heuristic":mapping["language_heuristic"],"old_call":old_state,"new_state":new_state,
                         "new_assertion":a,"original_target":float(old.loc[(case,finding),"old_fused_target"])})
    def describe(subset):
        available=[r for r in subset if r["new_assertion"] is not None]
        definite=[r for r in available if r["new_state"] in {"present","negated"} and r["old_call"] in {"positive","negative"}]
        agrees=sum((r["new_state"]=="present")==(r["old_call"]=="positive") for r in definite)
        return {"slots":len(subset),"accepted_slots":len(available),"unavailable_slots":len(subset)-len(available),
                "new_state_counts":dict(Counter(r["new_state"] for r in subset)),
                "new_present_old_silent":sum(r["new_state"]=="present" and r["old_call"]=="silent" for r in available),
                "new_present_old_negative":sum(r["new_state"]=="present" and r["old_call"]=="negative" for r in available),
                "new_negated_old_positive":sum(r["new_state"]=="negated" and r["old_call"]=="positive" for r in available),
                "definite_call_comparable_slots":len(definite),"definite_call_agreements":agrees,
                "definite_call_agreement_fraction":agrees/len(definite) if definite else None}
    spent=sum(r["charged_usd_upper"] for r in records.values())
    summary={"status":"complete_unsealed_offline","model":manifest["config"]["model"],"reports":len(records),
             "accepted_reports":sum(r["status"]=="accepted" for r in records.values()),"rejected_reports":len(rejections),
             "distinct_api_responses":len(response_ids),"usage":dict(usage),
             "fresh_token_cost_usd":round(spent,8),"shared_accounted_usd_upper":round(prior+spent,8),
             "validation_error_counts":dict(Counter(r["validation_error"] for r in rejections)),
             "all_cohorts_descriptive_only":describe(rows),
             "cohorts":{name:describe([r for r in rows if r["cohort"]==name]) for name in sorted(key.cohort.unique())},
             "per_finding":{f:describe([r for r in rows if r["finding"]==f]) for f in LABELS},
             "candidate_status":"AI-generated experimental extraction candidate; no independently measured accuracy",
             "opus_review_of_this_fresh_cohort":False,"independent_accuracy_established":False,
             "production_labels_changed":False,
             "denominator_note":"Report all 2400 planned slots; invalid reports contribute 12 unavailable slots each. Conditional definite-call agreement excludes silence and uncertainty and is not overall accuracy.",
             "provenance":{"extraction_manifest_sha256":sha(out/"manifest.json"),"source_sha256":sha(Path(__file__)),
                           "response_record_sha256":{p.name:sha(p) for p in sorted((out/"attempts").glob("*.json"))}}}
    result_dir.mkdir(parents=True)
    for name,values in [("comparisons.jsonl",rows),("rejected_reports.jsonl",rejections)]:
        (result_dir/name).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in values))
    write_json_new(result_dir/"summary.json",summary)
    return {k:v for k,v in summary.items() if k not in {"per_finding","provenance"}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"outputs/fresh_report_extraction_v1")
    parser.add_argument("--result-dir",type=Path,default=ROOT/"outputs/fresh_report_extraction_v1/comparison")
    print(json.dumps(summarize(**vars(parser.parse_args())),indent=2))

