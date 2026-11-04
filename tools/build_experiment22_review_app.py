#!/usr/bin/env python3
"""Safely unpack Experiment 22 and build a local blinded review interface."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/pssv/Downloads/experiment22_blinded_review_packet.zip")
OUT = ROOT / "outputs/experiment22_image_hard_cases_v1/review_workspace"


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(SOURCE) as archive:
        names = archive.namelist()
        allowed = {"expert_label_template.csv", "REVIEW_INSTRUCTIONS.txt"}
        allowed |= {f"images/E22-{index:03d}_{'pfoa' if index <= 40 else 'synovitis'}.png"
                    for index in range(1, 81)}
        if set(names) != allowed:
            raise RuntimeError("Review ZIP inventory changed")
        for name in names:
            destination = (OUT / name).resolve()
            if OUT.resolve() not in destination.parents:
                raise RuntimeError("Unsafe ZIP path: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    with (OUT / "expert_label_template.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 80:
        raise RuntimeError("Expected 80 review rows")
    compact = [{"case_id": row["case_id"], "arm": row["arm"],
                "montage_file": "images/" + row["montage_file"]} for row in rows]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment 22 blinded review</title>
<style>
body{{font:16px system-ui;background:#111;color:#eee;margin:0}} header,main{{max-width:1500px;margin:auto;padding:12px 20px}}
header{{display:flex;gap:18px;align-items:center;position:sticky;top:0;background:#111;z-index:2;border-bottom:1px solid #444}}
button,select,textarea{{font:inherit}} button{{padding:9px 16px;cursor:pointer}} button.active{{outline:3px solid #65aaff}}
#image{{width:100%;max-height:70vh;object-fit:contain;background:#000}} .row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}}
textarea{{width:100%;min-height:70px}} .label button{{font-size:20px;min-width:145px}} #status{{margin-left:auto}}
</style></head><body><header><strong id="case"></strong><span id="target"></span><span id="status"></span>
<button onclick="move(-1)">← Previous</button><button onclick="move(1)">Next →</button></header><main>
<img id="image" alt="MRI montage"><div class="row label"><strong>Target:</strong>
<button id="absent" onclick="setLabel(0)">0 — Absent</button><button id="present" onclick="setLabel(1)">1 — Present</button></div>
<div class="row"><label>Image quality <select id="quality" onchange="save()"><option value=""></option><option>diagnostic</option><option>limited</option><option>unreadable</option></select></label>
<label>Confidence <select id="confidence" onchange="save()"><option value=""></option><option>low</option><option>medium</option><option>high</option></select></label></div>
<textarea id="notes" placeholder="Optional notes" oninput="save()"></textarea><div class="row">
<button onclick="downloadCsv()">Download completed CSV</button><button onclick="clearAll()">Clear saved review</button>
<span>Keyboard: 0 absent, 1 present, ←/→ navigate.</span></div></main>
<script>
const cases={json.dumps(compact)}; let index=0; const key='experiment22-review-v1'; let answers=JSON.parse(localStorage.getItem(key)||'{{}}');
function current(){{return cases[index]}} function answer(){{return answers[current().case_id]||{{}}}}
function render(){{const c=current(),a=answer();document.getElementById('case').textContent=c.case_id;document.getElementById('target').textContent=c.arm==='pfoa'?'PF OA':'Synovitis';
document.getElementById('image').src=c.montage_file;document.getElementById('quality').value=a.image_quality||'';document.getElementById('confidence').value=a.reviewer_confidence||'';document.getElementById('notes').value=a.notes||'';
document.getElementById('absent').classList.toggle('active',a.expert_label===0);document.getElementById('present').classList.toggle('active',a.expert_label===1);
const done=cases.filter(x=>answers[x.case_id]&&[0,1].includes(answers[x.case_id].expert_label)).length;document.getElementById('status').textContent=`${{done}} / ${{cases.length}} labelled`;}}
function save(){{const c=current();answers[c.case_id]={{...(answers[c.case_id]||{{}}),image_quality:quality.value,reviewer_confidence:confidence.value,notes:notes.value}};localStorage.setItem(key,JSON.stringify(answers));render()}}
function setLabel(v){{answers[current().case_id]={{...answer(),expert_label:v}};localStorage.setItem(key,JSON.stringify(answers));render();if(index<cases.length-1)setTimeout(()=>move(1),180)}}
function move(d){{index=Math.max(0,Math.min(cases.length-1,index+d));render()}}
function quote(v){{v=String(v??'');return /[\",\n]/.test(v)?'\"'+v.replaceAll('\"','\"\"')+'\"':v}}
function downloadCsv(){{const columns=['case_id','arm','montage_file','expert_label','image_quality','reviewer_confidence','notes'];let lines=[columns.join(',')];for(const c of cases){{const a=answers[c.case_id]||{{}};lines.push(columns.map(k=>quote(k in a?a[k]:c[k]??'')).join(','))}}const b=new Blob([lines.join('\n')+'\n'],{{type:'text/csv'}});const u=URL.createObjectURL(b);const x=document.createElement('a');x.href=u;x.download='experiment22_expert_labels_completed.csv';x.click();URL.revokeObjectURL(u)}}
function clearAll(){{if(confirm('Clear all saved labels?')){{answers={{}};localStorage.removeItem(key);render()}}}}
addEventListener('keydown',e=>{{if(e.target.tagName==='TEXTAREA')return;if(e.key==='0')setLabel(0);if(e.key==='1')setLabel(1);if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)}});render();
</script></body></html>"""
    (OUT / "review.html").write_text(page)
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
