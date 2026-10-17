#!/usr/bin/env python3
"""Export the FP benefit cases of the confirmed frozen TEST27 run, CPU only.

Read cached candidate/event tables and original signal files. No torch import,
model loading, inference, tuning, or modification of any original experiment.
Distinguish direct FP removal from FP-to-TP reassignment under greedy matching.
All outputs, logs, manifests, and the ZIP stay in the independent analysis root.
"""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import zipfile
import numpy as np

NAME = "V7_0_7_RR_error_analysis_20261015"
RUN_NAME = "test27_current_20261015T070332_413151Z"
SUMMARY_SHA256 = "1df9a037f52314bec91f132e6ed9f9a2b1490d7ad89535d49ae9ec975c5c6a51"
FS = 2000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    require(bool(rows), "Empty CSV: " + str(path))
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def log(output, message):
    print(message, flush=True)
    with (output / "export.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def registry(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    write_json(folder / "file_registry.json", {
        "files": [{"path": p.relative_to(folder).as_posix(),
                   "bytes": p.stat().st_size, "sha256": digest(p)} for p in files],
        "registry_self": "Self-hash is in delivery_registry.json; omitted here to avoid recursion."})


def transition_counts(candidates, before, after):
    """Exact identity: delta_FP = removed_FP + FP_to_TP - TP_to_FP."""
    removed = sum(r[before + "_label"] == "FP" and
                  r[after + "_label"] == "NOT_KEPT" for r in candidates)
    reassigned = sum(r[before + "_label"] == "FP" and
                     r[after + "_label"] == "TP" for r in candidates)
    new_fp = sum(r[before + "_label"] == "TP" and
                 r[after + "_label"] == "FP" for r in candidates)
    before_fp = sum(r[before + "_label"] == "FP" for r in candidates)
    after_fp = sum(r[after + "_label"] == "FP" for r in candidates)
    require(before_fp - after_fp == removed + reassigned - new_fp,
            "FP transition reconciliation failed: " + before + " -> " + after)
    return {"direct_fp_removed": removed, "fp_to_tp_reassigned": reassigned,
            "tp_to_fp_reassigned": new_fp, "net_fp_reduction": before_fp - after_fp}


def audit_record(row, candidates, events):
    require(len(candidates) == int(row["n_ap"]), "Candidate count mismatch")
    require(len(events) == int(row["n_expert"]), "Expert count mismatch")
    require([int(r["ap_index"]) for r in candidates] == list(range(len(candidates))),
            "AP index ordering mismatch")
    require([int(e["expert_event_id"]) for e in events] == list(range(len(events))),
            "Expert event ordering mismatch")
    require(all(int(candidates[i]["ap_position"]) < int(candidates[i+1]["ap_position"])
                for i in range(len(candidates)-1)), "AP positions are not increasing")
    for stage in ("v11", "v707", "final"):
        counts = Counter(r[stage + "_label"] for r in candidates)
        event_counts = Counter(e[stage + "_event_label"] for e in events)
        require(set(counts) <= {"TP", "FP", "NOT_KEPT"}, "Unknown candidate label")
        for label in ("tp", "fp"):
            require(counts[label.upper()] == int(row[stage + "_" + label]),
                    "Stage candidate counts differ from frozen results")
        require(event_counts["FN"] == int(row[stage + "_fn"]), "Stage FN mismatch")
        require(event_counts["TP"] == counts["TP"], "Candidate/event TP mismatch")
        for r in candidates:
            if r[stage + "_label"] == "TP":
                event_id = int(r[stage + "_expert_event_id"])
                require(events[event_id][stage + "_matched_ap_index"] == r["ap_index"],
                        "Candidate/event matching identity mismatch")
    for r in candidates:
        require(r["v11_label"] in {"TP", "FP"}, "Unexpected baseline candidate removal")
        require((r["v707_label"] != "NOT_KEPT") == (r["v707_keep"] == "1"), "Model keep mismatch")
        require((r["final_label"] != "NOT_KEPT") == (r["final_keep"] == "1"), "Final keep mismatch")
        require(not (r["final_keep"] == "1" and r["v707_keep"] == "0"), "RR added a candidate")
    audit = {"db_name": row["db_name"], "record": row["record"]}
    for key, before, after in (("total", "v11", "final"),
                               ("model", "v11", "v707"), ("rr", "v707", "final")):
        audit.update({key + "_" + k: v for k, v in transition_counts(candidates, before, after).items()})
    return audit


def make_cases(db, record, candidates, events):
    ap = np.asarray([int(r["ap_position"]) for r in candidates], dtype=np.int64)
    gt = np.asarray([int(e["expert_position"]) for e in events], dtype=np.int64)
    require(len(gt) > 0 and np.all(np.diff(gt) >= 0), "Invalid expert sequence")
    selected, previous = [], -1
    for i, r in enumerate(candidates):
        if r["v11_label"] == "FP" and r["final_label"] != "FP":
            reassigned = r["final_label"] == "TP"
            if reassigned:
                kind = "FP_TO_TP_REASSIGNED"
                route = "MODEL_MATCH_REASSIGNMENT" if r["v707_label"] == "TP" else "RR_MATCH_REASSIGNMENT"
                expert_id = int(r["final_expert_event_id"])
                expert_role = "final_matched_true_event"
            else:
                kind = "RR_FP_REMOVED" if r["v707_keep"] == "1" else "MODEL_FP_REMOVED"
                route = kind
                expert_id = int(np.argmin(np.abs(gt - ap[i])))
                expert_role = "nearest_reference_only"
            event = events[expert_id]
            offset = (int(gt[expert_id]) - int(ap[i])) * 1000 / FS
            nearest_id = int(np.argmin(np.abs(gt - ap[i])))
            partner = int(event["v11_matched_ap_index"]) if reassigned and event["v11_matched_ap_index"] else -1
            title = {"MODEL_FP_REMOVED": "模型消除底层FP", "RR_FP_REMOVED": "RR规则消除底层FP",
                     "FP_TO_TP_REASSIGNED": "候选保留，FP重新匹配为TP"}[kind]
            text = title + "。"
            if reassigned:
                text += f"该AP未删除，最终匹配专家事件{expert_id}（{event['beat_symbol']}），偏移{offset:+.1f}ms；"
                text += f"该专家事件在V11_B阶段匹配AP{partner}。这是评价匹配变化收益。"
            else:
                text += f"该AP在V11_B为FP，最终删除；p_real={float(r['p_real']):.6f}。"
                if kind == "RR_FP_REMOVED":
                    text += f"模型保留后触发{r['delete_rule']}。"
                text += f"最近专家参考标注偏移{offset:+.1f}ms（{event['beat_symbol']}）。"
            if r["model_evaluated"] == "0":
                text += "概率为前30个AP直接放行值，非模型输出。"
            if r["rr_input_exceeds_100ms"] == "1":
                text += "该样本模型节律输入引用了AP后超过100ms的位置。"
            text += "形态原因待波形复核。"
            case_id = f"{db.upper()}_{record}_BENEFIT_ap{i:05d}"
            selected.append({"case_id": case_id, "case_type": "FP_BENEFIT",
                "benefit_type": kind, "benefit_title": title, "stage_route": route,
                **r, "expert_event_id": expert_id, "expert_position": int(gt[expert_id]),
                "expert_symbol": event["beat_symbol"], "expert_offset_ms": offset,
                "expert_role": expert_role, "nearest_expert_event_id": nearest_id,
                "nearest_expert_offset_ms": (int(gt[nearest_id])-int(ap[i]))*1000/FS,
                "baseline_partner_ap_index": partner if partner >= 0 else "",
                "baseline_partner_ap_position": int(ap[partner]) if partner >= 0 else "",
                "baseline_partner_v707_label": candidates[partner]["v707_label"] if partner >= 0 else "",
                "baseline_partner_final_label": candidates[partner]["final_label"] if partner >= 0 else "",
                "previous_final_keep_ap_index": previous if previous >= 0 else "",
                "previous_final_keep_ap_position": int(ap[previous]) if previous >= 0 else "",
                "waveform_error_type": "待人工复核", "brief_text": text,
                "waveform_file": f"waveforms/{case_id}.npz"})
        if r["final_keep"] == "1":
            previous = i
    return selected


def execute(root, analysis, output, script):
    run = analysis / RUN_NAME
    source_entries = json.loads((run / "file_registry.json").read_text())["files"]
    entries = {e["path"]: e for e in source_entries}
    checked = {}

    def verified(path):
        key = path.relative_to(analysis).as_posix()
        require(key in entries, "Unregistered source artifact: " + key)
        actual = digest(path)
        require(actual == entries[key]["sha256"], "Source artifact changed: " + key)
        checked[key] = actual
        return path

    summary_path = verified(run / "summary.json")
    require(digest(summary_path) == SUMMARY_SHA256, "Wrong frozen run summary")
    summary = json.loads(summary_path.read_text())
    require(summary["status"] == "CURRENT_FROZEN_TEST27_COMPLETE", "Incomplete TEST27 run")
    require(summary["original_files_unchanged"] is True, "Original input verification missing")
    require([summary["totals"][s]["fp"] for s in ("v11", "v707", "final")] == [353,120,97],
            "Unexpected frozen FP counts")
    require([summary["totals"][s]["fn"] for s in ("v11", "v707", "final")] == [348,426,428],
            "Unexpected frozen FN counts")
    record_path = verified(run / "all_records_results.csv")
    records = read_csv(record_path)
    require(len(records) == 27 and len({(r["db_name"],r["record"]) for r in records}) == 27,
            "Expected 27 unique records")
    (output / "tables").mkdir()
    (output / "waveforms").mkdir()
    shutil.copyfile(script, output / script.name)
    shutil.copyfile(summary_path, output / "source_summary.json")
    shutil.copyfile(record_path, output / "all_records_results.csv")
    audits, groups = [], []
    for row in records:
        db, record = row["db_name"], row["record"]
        log(output, "Audit cached tables: " + db + "/" + record)
        cp = verified(run / db / record / "all_ap_details.csv")
        ep = verified(run / db / record / "expert_events.csv")
        candidates, events = read_csv(cp), read_csv(ep)
        audit = audit_record(row, candidates, events)
        cases = make_cases(db, record, candidates, events)
        require(len(cases) == audit["total_direct_fp_removed"] + audit["total_fp_to_tp_reassigned"],
                "Benefit case reconciliation failed")
        audits.append(audit)
        if cases:
            dest = output / "tables" / db / record
            dest.mkdir(parents=True)
            for p in (cp,ep):
                shutil.copyfile(p, dest / p.name)
                require(digest(dest / p.name) == digest(p), "Table copy mismatch")
            groups.append((db, record, cases))
    counts = {k: sum(r[k] for r in audits) for k in audits[0] if k not in ("db_name","record")}
    require(counts["total_net_fp_reduction"] == 256, "Expected net FP reduction 256")
    require(counts["model_net_fp_reduction"] == 233 and counts["rr_net_fp_reduction"] == 23,
            "Stage FP benefit reconciliation failed")
    require(counts["total_tp_to_fp_reassigned"] == 0, "Unexpected adverse reassignment; inspect audit first")
    counts["benefit_cases"] = sum(len(cases) for _,_,cases in groups)
    require(counts["benefit_cases"] == 256, "Expected 256 benefit candidates")
    log(output, "Candidate audit: " + json.dumps(counts, ensure_ascii=False))
    write_csv(output / "benefit_counts_by_record.csv", audits)
    input_signals = {}
    all_cases = []
    for db,record,cases in groups:
        log(output, "Export waveforms: " + db + "/" + record + " (" + str(len(cases)) + " cases)")
        arrays = {c["case_id"]: {} for c in cases}
        for filename,key,left,right in (("ECG_FILT.txt","ecg",20*FS,3*FS),
                                         ("MEAN_TEST.txt","mean",FS//2,FS//2)):
            source = root / "data/parsed_features_11_B" / db / record / filename
            rel = source.relative_to(root).as_posix()
            expected = summary["input_sha256_before"].get(rel)
            require(expected is not None, "Missing frozen signal fingerprint: " + rel)
            before = source.stat()
            require(digest(source) == expected, "Signal changed since frozen prediction: " + rel)
            signal = np.atleast_1d(np.loadtxt(source, dtype=np.float32))
            require(signal.ndim == 1, "Expected a single-channel source signal")
            require(digest(source) == expected, "Signal changed during export: " + rel)
            after = source.stat()
            require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns),
                    "Signal changed during export: " + rel)
            input_signals[rel] = expected
            for c in cases:
                center = int(c["ap_position"])
                start, end = max(0,center-left), min(len(signal),center+right+1)
                require(0 <= start <= center < end <= len(signal), "Invalid waveform bounds")
                piece = signal[start:end].copy()
                require(np.isfinite(piece).all(), "Non-finite waveform; needs separate audit")
                arrays[c["case_id"]][key] = piece
                arrays[c["case_id"]][key+"_start"] = np.asarray(start,dtype=np.int64)
                c[key+"_start"],c[key+"_length"] = start,len(piece)
            del signal
        for c in cases:
            np.savez_compressed(output / c["waveform_file"], **arrays[c["case_id"]],
                                fs=np.asarray(FS,dtype=np.int64))
        all_cases.extend(cases)
        del arrays
    require(len({c["case_id"] for c in all_cases}) == 256, "Duplicate or missing case IDs")
    # Confirm cached prediction tables are unchanged after signal export.
    for rel,expected in checked.items():
        require(digest(analysis / rel) == expected, "Cached artifact changed during export: " + rel)
    write_csv(output / "benefit_cases.csv", all_cases)
    write_json(output / "source_artifacts_checked.json", {"artifacts": checked, "signals": input_signals})
    write_json(output / "plot_standard.json", {
        "current_review_candidate": "red five-point star; baseline FP, may be deleted or reassigned TP",
        "all_original_action_points": "green downward triangles and green dotted vertical lines",
        "expert_annotations": "blue upward triangles and blue dashed vertical lines",
        "panels": ["ECG_FILT -0.5..+0.5 s", "MEAN_TEST -0.5..+0.5 s",
                   "ECG_FILT -3..+3 s", "ECG_FILT -20..0 s"],
        "waveform_input_window_seconds": [-0.15,0.10], "input_last_sample_ms": 99.5,
        "review_context": "Offline human review only; actual model history is previous 30 original AP windows",
        "classification": "Removal stage and matching identity are verified; morphology requires waveform review"})
    write_json(output / "pack_summary.json", {
        "status": "COMPLETE", "source_run": str(run), "source_summary_sha256": SUMMARY_SHA256,
        "counts": counts, "benefit_types": dict(Counter(c["benefit_type"] for c in all_cases)),
        "stage_routes": dict(Counter(c["stage_route"] for c in all_cases)),
        "scope": "All baseline FP candidates no longer final FP, distinguishing removal and FP-to-TP reassignment",
        "real_time_status": "NOT_PASSED; frozen original offline analysis",
        "gpu_inference_performed": False, "script_sha256": digest(script)})
    log(output, "COMPLETE: " + str(len(all_cases)) + " benefit cases; CPU file export only")
    return counts


def main():
    script = Path(__file__).resolve()
    analysis = script.parent
    require(analysis.name == NAME and analysis.parent.name == "runs",
            "Place this script in ECG/runs/" + NAME)
    root = analysis.parent.parent
    output = analysis / ("benefit_wavepack_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    log(output, "Output: " + str(output))
    try:
        execute(root, analysis, output, script)
    except Exception as exc:
        with (output / "export.log").open("a",encoding="utf-8") as f:
            traceback.print_exc(file=f)
        write_json(output / "FAILED.json", {"status": "FAILED_INCOMPLETE", "error": str(exc)})
        registry(output)
        traceback.print_exc()
        return 1
    registry(output)
    archive = analysis / (output.name + ".zip")
    try:
        with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    z.write(path,path.relative_to(output).as_posix())
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, "ZIP integrity check failed")
        write_json(output / "delivery_registry.json", {
            "archive": {"path": "../" + archive.name, "bytes": archive.stat().st_size,
                        "sha256": digest(archive)},
            "internal_registry_sha256": digest(output / "file_registry.json"),
            "delivery_registry_self": "Self-hash omitted to avoid recursion"})
    except Exception:
        traceback.print_exc()
        return 1
    print("UPLOAD_ZIP:",archive,flush=True)
    print("ZIP_MiB:",round(archive.stat().st_size / 1024**2,2),flush=True)
    print("ZIP_SHA256:",digest(archive),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
