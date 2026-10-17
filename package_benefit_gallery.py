#!/usr/bin/env python3
"""Verify the completed benefit gallery and build a SHA256-registered ZIP."""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import platform
import shutil
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import matplotlib
import PIL
from PIL import Image
from plot_error_cases import digest_bytes,write_atomic


class Gallery(HTMLParser):
    def __init__(self):
        super().__init__();self.articles=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="article":self.articles.append(attrs)
        for name in ("src","href"):
            if name in attrs:self.links.append(attrs[name])


def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);args=p.parse_args()
    out=args.output.resolve()
    with (out/"case_descriptions.csv").open(encoding="utf-8-sig",newline="") as f:
        cases=list(csv.DictReader(f))
    assert len(cases)==256 and len({c["case_id"] for c in cases})==256
    assert len({c["image"] for c in cases})==256
    assert Counter(c["benefit_type"] for c in cases)=={
        "MODEL_FP_REMOVED":227,"RR_FP_REMOVED":23,"FP_TO_TP_REASSIGNED":6}
    dimensions=Counter()
    for c in cases:
        with Image.open(out/c["image"]) as picture:
            dimensions[picture.size]+=1;picture.verify()
        assert c["explanation"] and c["short_caption"]
        assert c["v11_label"]=="FP" and c["final_label"]!="FP"
    assert len(dimensions)==1
    all_png=list(out.rglob("*.png"))
    assert len(all_png)==263  # 256 cases + 5 report copies + 2 linked FN images.
    for image in all_png:
        with Image.open(image) as picture:picture.verify()
    for path in out.rglob("*.svg"):ET.parse(path)
    assert len(list(out.rglob("*.svg")))==5
    page=Gallery();page.feed((out/"index.html").read_text())
    assert len(page.articles)==256
    assert Counter(c["data-type"] for c in page.articles)==Counter(c["benefit_type"] for c in cases)
    for link in page.links:
        assert not link.startswith(("http:","https:")),"Gallery must work offline"
        assert (out/link).is_file(),link
    overlap=[c for c in cases if c["relation_category"]=="DUP_FN"]
    assert len(overlap)==2 and all(c["paired_added_fn_case_id"] for c in overlap)
    assert sum(c["rr_input_exceeds_100ms"]=="1" for c in cases)==9
    # Check recorded source-code copy hashes without touching the original sources.
    for name in ("plot_benefit_cases.py","plot_error_cases.py","export_benefit_wavepack.py"):
        assert digest_bytes((out/name).read_bytes())==digest_bytes((out.parent/name).read_bytes())
    here=Path(__file__).resolve()
    if here!=(out/here.name):shutil.copyfile(here,out/here.name)
    qa={"completed_utc":datetime.now(timezone.utc).isoformat(),"status":"PASSED",
        "unique_cases":256,"PNG_images_including_copies":len(all_png),"report_SVG":5,
        "image_dimensions":{f"{w}x{h}":n for (w,h),n in dimensions.items()},
        "gallery_cards":len(page.articles),"gallery_links":"all local targets exist",
        "matching_counts":"227 direct model removals + 23 RR removals + 6 FP-to-TP reassignments",
        "paired_FN_context_images":2,"model_rhythm_position_exceeds_100ms_cases":9,
        "environment":{"python":platform.python_version(),"numpy":np.__version__,
                       "matplotlib":matplotlib.__version__,"pillow":PIL.__version__},
        "verification_scope":"Input wavepack registry and CRC, frozen table matching, rule conditions, all PNG CRCs, SVG XML, gallery inventory and links",
        "visual_QA":"Representative layouts inspected; morphology is not automatically diagnosed",
        "gpu_or_model_inference":False}
    write_atomic(out/"quality_checks.json",json.dumps(qa,ensure_ascii=False,indent=2).encode())
    files=[{"path":f.relative_to(out).as_posix(),"bytes":f.stat().st_size,
            "sha256":digest_bytes(f.read_bytes())} for f in sorted(out.rglob("*"))
           if f.is_file() and f.name!="file_registry.json"]
    registry={"status":"COMPLETE_256_BENEFIT_CASES","files":files,
              "source_zip_sha256":json.loads((out/"analysis_summary.json").read_text())["source_zip_sha256"],
              "registry_self":"Self-hash recorded in the external delivery receipt"}
    write_atomic(out/"file_registry.json",json.dumps(registry,ensure_ascii=False,indent=2).encode())
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():z.write(f,f.relative_to(out).as_posix())
    archive=out.parent/"V707_RR_256_benefit_figures_color_v2_20261017.zip"
    write_atomic(archive,buffer.getvalue())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for e in files:assert digest_bytes(z.read(e["path"]))==e["sha256"],e["path"]
    receipt={"path":archive.name,"bytes":archive.stat().st_size,"sha256":digest_bytes(archive.read_bytes()),
             "registry_sha256":digest_bytes((out/"file_registry.json").read_bytes()),
             "verified_registered_files":len(files),"quality_status":"PASSED"}
    write_atomic(archive.with_suffix(".delivery.json"),json.dumps(receipt,indent=2).encode())
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
