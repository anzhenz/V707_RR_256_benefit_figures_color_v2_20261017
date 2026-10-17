#!/usr/bin/env python3
"""Generate all 256 frozen FP-benefit figures and the offline review gallery.

CPU plotting only. No inference, threshold search, or original source edits.
Imports the unchanged plotting helpers and exporter bundled alongside this file.
"""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import textwrap
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image
import plot_error_cases as base
import export_benefit_wavepack as audit
from plot_error_cases import WAVE,AP_COLOR,BLUE,RED,SHADE,minmax_decimate,write_atomic,setup_font

CATEGORY_NAMES={
 "MODEL_FP_REMOVED":"优势｜模型删除底层FP",
 "RR_FP_REMOVED":"优势｜RR规则删除底层FP",
 "FP_TO_TP_REASSIGNED":"优势｜FP重新匹配为TP（候选保留）"}
RELATION_NAMES={"EARLY":"最近专家标注前超过150ms", "LATE":"最近专家标注后超过150ms",
                "DUP_TP":"容差内重复候选，专家事件仍检出",
                "DUP_FN":"容差内重复候选，专家事件最终漏检",
                "REASSIGNED":"FP重新匹配为TP"}


def load_pack(path):
    z=zipfile.ZipFile(path)
    if z.testzip() is not None:raise ValueError("ZIP CRC check failed")
    entries=json.loads(z.read("file_registry.json"))["files"]
    for e in entries:
        if base.digest_bytes(z.read(e["path"]))!=e["sha256"]:
            raise ValueError("Input hash mismatch: "+e["path"])
    pack=json.loads(z.read("pack_summary.json"))
    if pack["status"]!="COMPLETE" or pack["counts"]["total_net_fp_reduction"]!=256:
        raise ValueError("Wrong benefit pack")
    if base.digest_bytes(z.read("source_summary.json"))!=audit.SUMMARY_SHA256:
        raise ValueError("Wrong frozen source summary")
    cases=base.rows(z.read("benefit_cases.csv"))
    if len(cases)!=256 or len({c["case_id"] for c in cases})!=256:
        raise ValueError("Incomplete or duplicate case inventory")
    record_rows={(r["db_name"],r["record"]):r for r in base.rows(z.read("all_records_results.csv"))}
    tables={}
    for c in cases:
        key=c["db_name"],c["record"]
        if key not in tables:
            prefix=f"tables/{key[0]}/{key[1]}/"
            ca,ev=base.rows(z.read(prefix+"all_ap_details.csv")),base.rows(z.read(prefix+"expert_events.csv"))
            audit.audit_record(record_rows[key],ca,ev)
            tables[key]=ca,ev
        ca,ev=tables[key]
        original=ca[int(c["ap_index"])]
        if any(c[k]!=v for k,v in original.items()):raise ValueError("Case/table discrepancy")
        if c["v11_label"]!="FP" or c["final_label"]=="FP":raise ValueError("Invalid benefit case")
        with np.load(io.BytesIO(z.read(c["waveform_file"])),allow_pickle=False) as w:
            if int(w["fs"])!=2000:raise ValueError("Wrong sample rate")
            for k in ("ecg","mean"):
                if len(w[k])!=int(c[k+"_length"]) or int(w[k+"_start"])!=int(c[k+"_start"]):
                    raise ValueError("Waveform index discrepancy")
                if not np.isfinite(w[k]).all():raise ValueError("Non-finite waveform")
    if Counter(c["benefit_type"] for c in cases)!=Counter(pack["benefit_types"]):
        raise ValueError("Benefit category discrepancy")
    return z,cases,tables,pack


def classify(c,tables):
    ca,ev=tables[c["db_name"],c["record"]]
    kind=c["benefit_type"]; d=float(c["nearest_expert_offset_ms"])
    e=ev[int(c["nearest_expert_event_id"])]
    nearest_final_fn=e["final_event_label"]=="FN"
    c["nearest_event_final_label"]=e["final_event_label"]
    c["nearest_event_added_fn"]=e["added_fn_vs_v11"]
    c["paired_added_fn_case_id"]=""
    if kind=="FP_TO_TP_REASSIGNED":
        relation="REASSIGNED"
        partner=c["baseline_partner_ap_index"]
        if e["final_matched_ap_index"]!=c["ap_index"] or c["baseline_partner_final_label"]!="NOT_KEPT":
            raise ValueError("Unexpected reassignment chain")
        explanation=(f"当前AP在底层被评价为FP，但未被模型或RR删除；原匹配AP#{partner}被模型删除后，"
                     f"当前AP以p_real={float(c['p_real']):.4f}重新匹配专家事件{c['expert_event_id']}"
                     f"（{c['expert_symbol']}，相对AP{float(c['expert_offset_ms']):+.1f}ms）。"
                     "该专家事件在各阶段均检出，这是候选匹配变化，不是新增检出一个心搏。")
        short=f"当前AP保留为TP；原匹配AP#{partner}被删除。专家事件一直检出，收益来自一对一匹配关系变化。"
    else:
        if abs(d)<=150:
            relation="DUP_FN" if nearest_final_fn else "DUP_TP"
            if not e["v11_matched_ap_index"]:raise ValueError("Tolerance duplicate lacks baseline matching AP")
            explanation=(f"底层FP位于最近专家标注{d:+.1f}ms，在±150ms匹配容差内；"
                         f"该专家事件在底层已由AP#{e['v11_matched_ap_index']}匹配。")
            if nearest_final_fn:
                if e["added_fn_vs_v11"]!="1":raise ValueError("Unexpected duplicate event status")
                fnid=f"{c['db_name'].upper()}_{c['record']}_FN_event{int(e['expert_event_id']):05d}_ap{int(e['v11_matched_ap_index']):05d}"
                c["paired_added_fn_case_id"]=fnid
                explanation+=("当前FP候选及原匹配TP候选均被删除，该专家事件最终漏检。"
                              "同一片段同时计入候选级FP减少和事件级新增FN，不能据FP标签把该波形判为噪声。")
                short="当前底层FP被删除，但原匹配TP也被删除，专家事件最终FN。该片段同时存在FP下降和新增漏检。"
            else:
                explanation+=f"删除当前额外候选后，该专家事件仍由AP#{e['final_matched_ap_index']}检出。"
                short=f"删除同一心搏附近的额外候选；专家事件仍由AP#{e['final_matched_ap_index']}匹配，当前偏移{d:+.1f}ms。"
        else:
            relation="EARLY" if d>150 else "LATE"
            side="前" if d>0 else "后"
            explanation=(f"当前AP在底层为FP，位于最近专家参考标注{side}{abs(d):.1f}ms，超出±150ms评价容差。")
            short=f"删除底层额外候选：位于最近专家标注{side}{abs(d):.1f}ms，超出±150ms评价容差。"
        if kind=="MODEL_FP_REMOVED":
            explanation+=f"模型p_real={float(c['p_real']):.4f}、p_noise={float(c['p_noise']):.4f}，未通过原0.70噪声阈值；RR未评估此候选。"
        elif kind=="RR_FP_REMOVED":
            rule="Rule 1" if c["rule1_delete"]=="1" else "Rule 2"
            explanation+=(f"模型以p_real={float(c['p_real']):.4f}保留，随后{rule}删除；"
                          f"RR={float(c['rr_ms']):.1f}ms，参考RR={float(c['rr_ref_ms']):.1f}ms，比值={float(c['rr_ratio']):.3f}，"
                          f"历史稳定={c['history_stable']}、cooldown={c['cooldown_before']}。")
            if rule=="Rule 2":
                explanation+=f"此前3秒低置信AP数={c['recent_low_conf_count']}，前一保留概率减当前概率={float(c['delta_p_prev_keep']):.3f}。"
            short+=f"{rule}：RR/参考={float(c['rr_ratio']):.3f}。"
    if c["rr_input_exceeds_100ms"]=="1":
        explanation+="该候选的模型节律输入引用了AP后超过100ms的位置，属于冻结模型离线画像。"
    c["relation_category"]=relation
    c["relation_category_zh"]=RELATION_NAMES[relation]
    c["short_caption"]=short
    return kind,explanation


def make_figure(c,z,tables,out,svg=False):
    ca, ev = tables[(c["db_name"],c["record"])]
    ap0=int(c["ap_position"])
    aps=np.asarray([int(r["ap_position"]) for r in ca])
    gt=np.asarray([int(e["expert_position"]) for e in ev])
    with np.load(io.BytesIO(z.read(c["waveform_file"])),allow_pickle=False) as w:
        ecg=w["ecg"].copy(); es=int(w["ecg_start"])
        mean=w["mean"].copy(); ms=int(w["mean_start"])
    category, explanation=classify(c,tables)
    fig=plt.figure(figsize=(11.2,9.1))
    grid=fig.add_gridspec(4,1,left=.071,right=.98,top=.795,bottom=.183,
                        hspace=.63,height_ratios=[1.12,1.0,1.03,.92])
    axes=[fig.add_subplot(grid[i]) for i in range(4)]
    specs=[(ecg,es,-.5,.5,"a  ECG_FILT：局部波形 ±0.5 s"),
           (mean,ms,-.5,.5,"b  MEAN_TEST：局部特征 ±0.5 s"),
           (ecg,es,-3,3,"c  ECG_FILT：节律上下文 ±3 s（离线复核）"),
           (ecg,es,-20,0,"d  ECG_FILT：前20秒历史背景（离线复核）")]
    for panel,(ax,(signal,start,left,right,title)) in enumerate(zip(axes,specs)):
        sample_left=max(start,int(round(ap0+left*2000)))
        sample_right=min(start+len(signal)-1,int(round(ap0+right*2000)))
        if sample_right<sample_left:
            raise ValueError("Missing panel data: "+c["case_id"])
        pos=np.arange(sample_left,sample_right+1,dtype=np.int64)
        xx=(pos-ap0)/2000
        yy=signal[pos-start]
        xp,yp=minmax_decimate(xx,yy,3600 if panel>=2 else 8000)
        ax.plot(xp,yp,color=WAVE,lw=.77 if panel<3 else .58,zorder=2)
        lo,hi=float(yy.min()),float(yy.max())
        span=max(hi-lo,1.0)
        ax.set_ylim(lo-.14*span,hi+.20*span)
        ax.set_xlim(left,right)
        if panel<3:
            ax.axvspan(-.15,.1,color=SHADE,alpha=.82,zorder=0)
            ax.axvline(.1,color="#7f91a1",lw=.8,ls="-.",zorder=1)
        # Original APs are shown regardless of the model/RR keep decision.
        aa=aps[(aps>=sample_left)&(aps<=sample_right)]
        gg=gt[(gt>=sample_left)&(gt<=sample_right)]
        for p in aa:
            ax.axvline((p-ap0)/2000,color=AP_COLOR,ls=":",lw=.90,alpha=.80 if panel>=2 else .96,zorder=1)
        for p in gg:
            ax.axvline((p-ap0)/2000,color=BLUE,ls="--",lw=.95,alpha=.80 if panel>=2 else .96,zorder=1)
        marker_size=3.0 if panel>=2 else 5.0
        ax.plot((aa-ap0)/2000,signal[aa-start],ls="none",marker="v",ms=marker_size,
                color=AP_COLOR,mec="white",mew=.22,zorder=4,clip_on=True)
        ax.plot((gg-ap0)/2000,signal[gg-start],ls="none",marker="^",ms=marker_size+.6,
                color=BLUE,mec="white",mew=.22,zorder=5,clip_on=True)
        if start<=ap0<start+len(signal):
            ax.plot(0,float(signal[ap0-start]),marker="*",ms=12 if panel<2 else 10,
                    mfc=RED,mec="white",mew=.5,ls="none",zorder=8,clip_on=False)
        ax.set_title(title,loc="left",fontsize=9.4,pad=4)
        ax.set_ylabel("幅值 / AD")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="y",color="#e4e8eb",lw=.5,zorder=0)
        ax.tick_params(length=3,pad=2)
        for edge in ("top","right"):
            ax.spines[edge].set_visible(False)
        for edge in ("left","bottom"):
            ax.spines[edge].set_color("#a7b0b7")
        ax.set_xlabel("相对当前AP的时间 / s",labelpad=2)
        if panel<2:
            ax.set_xticks([-.5,-.25,0,.1,.25,.5])
        if panel==0:
            ax.text(.1,.98,"+100ms截止",transform=ax.get_xaxis_transform(),fontsize=7.8,
                    color="#61778a",ha="left",va="top")
            target=int(c["expert_position"])
            if sample_left<=target<=sample_right:
                ax.annotate(c["expert_symbol"],((target-ap0)/2000,float(signal[target-start])),
                            xytext=(3,8),textcoords="offset points",fontsize=8,color=BLUE,
                            bbox={"facecolor":"white","alpha":.78,"edgecolor":"none","pad":.5})
    if c["benefit_type"]=="FP_TO_TP_REASSIGNED":
        partner=int(c["baseline_partner_ap_index"])
        pp=int(ca[partner]["ap_position"])
        if es<=pp<es+len(ecg) and abs(pp-ap0)<=1000:
            axes[0].annotate(f"原匹配AP#{partner}（已删）",((pp-ap0)/2000,float(ecg[pp-es])),
                             xytext=(5,22),textcoords="offset points",fontsize=7.7,color="#586b7a",
                             arrowprops={"arrowstyle":"-","lw":.6,"color":"#788997"},
                             bbox={"facecolor":"white","alpha":.9,"edgecolor":"none","pad":1})
    label="FP减少｜模型删除候选，关联事件FN" if c["relation_category"]=="DUP_FN" else CATEGORY_NAMES[category]
    fig.text(.071,.976,f"{c['db_name'].upper()} / {c['record']}   ·   AP #{int(c['ap_index']):05d}   ·   {label}",
             fontsize=12.0,fontweight="bold",ha="left",va="top",color="#152b3d")
    probability=(f"p_real={float(c['p_real']):.4f}  |  p_noise={float(c['p_noise']):.4f}"
                 if c["model_evaluated"]=="1" else "前30个AP直接放行：p_real=1（非模型输出）")
    expert_role="最终匹配专家" if c["benefit_type"]=="FP_TO_TP_REASSIGNED" else "最近专家参考"
    fig.text(.071,.943,f"{probability}   |   {expert_role} {c['expert_symbol']}：{float(c['expert_offset_ms']):+.1f} ms   |   AP={ap0}",
             fontsize=9.1,ha="left",va="top",color="#31485a")
    states={"FP":"FP", "TP":"TP", "NOT_KEPT":"删除"}
    stage_line=(f"V11_B：{states[c['v11_label']]}  →  V7.0.7：{states[c['v707_label']]}  →  最终：{states[c['final_label']]}"
                f"   |   模型节律输入位置超过+100ms：{'是' if c['rr_input_exceeds_100ms']=='1' else '否'}")
    fig.text(.071,.916,stage_line,fontsize=9.0,ha="left",va="top",color="#31485a")
    def number(k,places=1):
        value=float(c[k]);return f"{value:.{places}f}" if np.isfinite(value) else "未建立"
    if c["v707_keep"]=="0":
        rrline="模型阶段已删除；RR后处理未评估此候选，空RR字段不表示节律不稳定"
    else:
        rule="Rule 1删除" if c["rule1_delete"]=="1" else "Rule 2删除" if c["rule2_delete"]=="1" else "最终保留"
        rrline=(f"RR={number('rr_ms')}ms / 参考={number('rr_ref_ms')}ms / 比值={number('rr_ratio',3)}"
                f"   |   稳定={c['history_stable']}，保护计数={c['cooldown_before']}   |   {rule}")
    fig.text(.071,.889,rrline,fontsize=8.6,ha="left",va="top",color="#526677")
    handles=[Line2D([],[],color=RED,marker="*",ms=10,ls="none",label="当前复核AP（原FP）"),
             Line2D([],[],color=AP_COLOR,marker="v",ms=5,ls=":",lw=.8,label="动态阈值候选AP"),
             Line2D([],[],color=BLUE,marker="^",ms=5,ls="--",lw=.8,label="专家标注"),
             Patch(facecolor=SHADE,label="波形输入窗 −150～+100ms")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.063,.860),ncol=4,
               frameon=False,fontsize=8.9,columnspacing=2.0,handlelength=2.0)
    # Caption remains factual; the full matching and rule chain is in the CSV.
    short="\n".join(textwrap.wrap("案例说明："+c["short_caption"],width=67,
                                  break_long_words=True,break_on_hyphens=False))
    fig.text(.071,.102,short,fontsize=9.5,ha="left",va="top",color="#243c4e",linespacing=1.25)
    fig.text(.071,.063,"说明：窗口外及历史波形仅供人工复核；模型历史为前30个原始AP，20秒视图仅作背景。",
             fontsize=8.1,ha="left",va="top",color="#657685")
    fig.text(.071,.034,c["case_id"]+"   |   当前冻结权重：离线优势分析；完整流程实时性审计尚未通过",
             fontsize=7.8,ha="left",va="top",color="#7a8792")
    out.parent.mkdir(parents=True,exist_ok=True)
    png_buffer=io.BytesIO()
    fig.savefig(png_buffer,dpi=180,format="png")
    payload=png_buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as picture:
        picture.verify()
    write_atomic(out,payload)
    if svg:
        svg_buffer=io.BytesIO()
        fig.savefig(svg_buffer,format="svg")
        write_atomic(out.with_suffix(".svg"),svg_buffer.getvalue())
    plt.close(fig)
    return category, explanation


EXAMPLES=[
 ("01_model_extra_candidate","MITDB_117_BENEFIT_ap00606","模型删除评价容差外的额外候选"),
 ("02_RR_rule1","MITDB_105_BENEFIT_ap00873","Rule 1删除稳定节律中的额外候选"),
 ("03_matching_reassignment","MITDB_105_BENEFIT_ap01704","候选保留，FP重新匹配为TP"),
 ("04_RR_rule2_duplicate","MITDB_203_BENEFIT_ap01045","Rule 2删除同一心搏附近的额外候选"),
 ("05_FP_reduction_and_FN","MITDB_105_BENEFIT_ap01906","同一片段兼有FP减少与新增FN"),
]


def save_csv(path,rows):
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    write_atomic(path,buf.getvalue().encode("utf-8-sig"))


def finish(input_path,output,z,results,tables,pack):
    types=Counter(c["benefit_type"] for c in results)
    relations=Counter(c["relation_category"] for c in results)
    by_type=[{"类别":CATEGORY_NAMES[k],"数量":n,"占256例比例":f"{n/256:.2%}"} for k,n in types.items()]
    by_relation=[{"类别":RELATION_NAMES[k],"数量":n,"占256例比例":f"{n/256:.2%}"} for k,n in relations.items()]
    crossed=Counter((c["benefit_type"],c["relation_category"]) for c in results)
    save_csv(output/"benefit_type_counts.csv",by_type)
    save_csv(output/"benefit_relation_counts.csv",by_relation)
    save_csv(output/"benefit_stage_relation_counts.csv",[
        {"判决阶段":CATEGORY_NAMES[k],"匹配位置关系":RELATION_NAMES[r],"数量":n,
         "占256例比例":f"{n/256:.2%}"} for (k,r),n in crossed.items()])
    save_csv(output/"case_descriptions.csv",results)
    audit_dir=output/"audit";audit_dir.mkdir(exist_ok=True)
    for name in ("benefit_counts_by_record.csv","all_records_results.csv","pack_summary.json","source_summary.json"):
        write_atomic(audit_dir/name,z.read(name))
    source_summary=json.loads(z.read("source_summary.json"))
    refs={c["case_id"]:c for c in results}
    example_dir=output/"report_examples";example_dir.mkdir(exist_ok=True)
    report=["# 优势部分：报告代表案例\n",
            "当前冻结模型的FP净减少为256：直接删除250例，重新匹配为TP 6例。前三例分别展示模型删除、RR删除、匹配变化。\n"]
    for filename,case_id,title in EXAMPLES:
        c=refs[case_id]
        make_figure(c,z,tables,example_dir/(filename+".png"),svg=True)
        report.append(f"## {title}\n\n![{title}](report_examples/{filename}.png)\n\n{c['explanation']}\n\n案例：`{case_id}`。\n")
    paired_dir=output/"paired_FN_context"
    previous=output.parent/"error_figures_20261015_v1"
    paired=[]
    for c in results:
        if not c["paired_added_fn_case_id"]:continue
        fnid=c["paired_added_fn_case_id"]
        image=previous/"images/FN"/c["record"]/(fnid+".png")
        if image.exists():
            paired_dir.mkdir(exist_ok=True)
            shutil.copyfile(image,paired_dir/image.name)
            paired.append({"benefit_case_id":c["case_id"],"FN_case_id":fnid,
                           "FN_image":(paired_dir/image.name).relative_to(output).as_posix()})
    if paired:
        save_csv(output/"paired_FP_reduction_FN.csv",paired)
        report.append("## FP下降与FN增加的关联复核\n\n"
                      "两例容差内候选删除同时伴随对应专家事件新增FN。原177例图集的关联FN图已放入 `paired_FN_context/`。\n\n")
        for item in paired:
            report.append(f"- `{item['benefit_case_id']}` 对应 `{item['FN_case_id']}`。\n")
    write_atomic(output/"report_captions.md","\n".join(report).encode())
    weight=source_summary["input_sha256_before"]["runs/V7.0.7/checkpoints/best_model_v7_0_7_best.pth"]
    stats={"status":"COMPLETE_256_BENEFIT_CASES","source_zip_sha256":base.digest_bytes(input_path.read_bytes()),
           "counts":pack["counts"],"benefit_types":dict(types),"relation_categories":dict(relations),
           "RR_rules":dict(Counter(c["delete_rule"] for c in results if c["benefit_type"]=="RR_FP_REMOVED")),
           "rr_input_exceeds_100ms":sum(c["rr_input_exceeds_100ms"]=="1" for c in results),
           "duplicate_removed_with_added_FN":[c["case_id"] for c in results if c["relation_category"]=="DUP_FN"],
           "paired_FN_images":len(paired),"weight_sha256":weight,"new_inference_performed":False}
    write_atomic(output/"analysis_summary.json",json.dumps(stats,ensure_ascii=False,indent=2).encode())
    distribution="\n".join(f"| {r['类别']} | {r['数量']} | {r['占256例比例']} |" for r in by_type)
    geometry="\n".join(f"| {r['类别']} | {r['数量']} | {r['占256例比例']} |" for r in by_relation)
    readme=f'''# ECG：全部256例FP减少案例

双击 `index.html` 筛选浏览全部256例，或直接查看 `images/`。每例沿用177例错误图集的四视图、颜色和尺寸，图内附简要说明；完整字段与核实说明在 `case_descriptions.csv`。

## 当前口径

| 阶段 | FP | FN | FP净减少 | FN净增加 |
|---|---:|---:|---:|---:|
| V11_B | 353 | 348 | — | — |
| V7.0.7 | 120 | 426 | 233 | 78 |
| V7.0.7 + RR | 97 | 428 | 再减少23 | 再增加2 |

最终相对V11_B：FP净减少256，专家事件新增FN 80、恢复0。旧报告的75与243属于历史结果；本图集采用当前冻结权重，未尝试重现已覆盖的历史权重。

## 优势的组成

| 类别 | 数量 | 占256例比例 |
|---|---:|---:|
{distribution}

FP净减少 = 250个底层FP候选直接删除 + 6个FP重新匹配为TP − 0个TP重新匹配为FP = 256。
模型直接删除227个FP，同时产生6个匹配重分配收益，因此模型阶段FP净减少233；RR再直接删除23个FP。
RR这23例中，Rule 1为19例，Rule 2为4例。原RR总删除25例还包括2个TP转为FN，已收入原177例错误图集。
6个重分配案例的专家事件各阶段均被检出，不能写成“额外检出了6个真实心搏”；当前AP均保留，原匹配AP均在模型阶段删除。

## 匹配与位置分类

| 类别 | 数量 | 占256例比例 |
|---|---:|---:|
{geometry}

这些分类由冻结的标签、时序和匹配关系核实，不直接确定伪迹、T波或异常心搏等形态/生理来源。±150ms是评价匹配容差，不能当作模型的+100ms可用观察上限。

**两例需要同时看劣势：**105/AP1906、203/AP891虽然在底层被标为FP，但均处于真实心搏匹配容差内；它们和原匹配TP候选一起被模型删除，对应专家事件也最终漏检。
它们可计入候选级FP下降，却不能当作单纯的“正确去噪”证据。这两例与原80个新增FN事件有交集，关联编号在 `case_descriptions.csv`，关联FN图片在 `paired_FN_context/`（复用原图，未改动）。

## 图例

- 红色五角星：当前复核AP，即底层阶段的FP。该点最终可被删除，也可保留为TP。
- 橙色倒三角及点线：全部原始动态阈值AP，包括后来被删除的候选。
- 蓝色上三角及虚线：专家真实心搏参考位置，不要求恰在正向峰顶。
- 浅色区域：波形输入−150～+100ms，实际末采样点+99.5ms。
- 四视图：ECG局部±0.5s、MEAN_TEST局部±0.5s、ECG节律±3s、ECG历史−20～0s。

长视图使用分箱最小/最大值保留极值，局部视图保留全部采样点。20秒背景及窗口外波形仅供离线人工复核，模型实际历史是前30个原始AP窗口。
模型删除后的候选未进入RR评估，其空RR字段和history_stable=0不表示实际节律不稳定。
FP案例的专家符号属于“最近专家参考”，不能据此把该FP直接命名为N类或V类心搏。

## 实时性说明

这里的实时性审计未通过指**输入信息可用时间**，与GPU速度、图片渲染速度无关。
全TEST27的81个模型样本曾在节律输入中引用AP后超过100ms的MEAN_R_POINT位置；本256例中有9例带此标记，其中8个模型删除、1个RR删除。
这使当前结果只能作为原冻结流程的离线画像；上游滤波与特征生成延时也未完整核实。画图展示的±3秒及20秒背景不会反过来成为模型输入。
本轮没有重新推理、训练、改变阈值或用TEST27调参。实际噪声阈值保持原代码0.70。

## 文件与复用

- `images/`：全部256张PNG，按三类处理结果和记录分目录。
- `case_descriptions.csv`：逐例处理结果、概率、专家与匹配关系、RR状态、规则、说明和图片路径。
- `benefit_*_counts.csv`：数量与比例；`audit/`保存原始汇总以便追溯。
- `report_examples/`：5例PNG与SVG；`report_captions.md`给出对应说明。
- `index.html`：无需联网的筛选图集，可按阶段、记录和匹配关系筛选。
- `plot_benefit_cases.py`、`plot_error_cases.py`、`export_benefit_wavepack.py`：本轮代码及原有只读绘图依赖。
- `assets/plot_font.ttf`：统一中文字体；`file_registry.json`登记全部交付文件SHA256。

使用numpy、matplotlib、pillow运行 `python -B plot_benefit_cases.py 原始优势波形包.zip 新输出目录`。首次缺字体时需pymupdf；原177例图集如位于输出目录的同级指定目录，会附带复用两张FN图。

当前权重SHA256：`{weight}`。
输入优势波形包SHA256：`{stats['source_zip_sha256']}`。
'''
    write_atomic(output/"README.md",readme.encode())
    cards=[]
    for c in results:
        caption=html.escape(c["explanation"])
        cards.append(f'''<article data-type="{c['benefit_type']}" data-record="{c['record']}" data-relation="{c['relation_category']}">
<h3>{html.escape(c['case_id'])}</h3><div class="tag">{html.escape(CATEGORY_NAMES[c['benefit_type']])} · {html.escape(c['relation_category_zh'])}</div>
<a href="{c['image']}" target="_blank"><img loading="lazy" src="{c['image']}" alt="{c['case_id']}"></a><p>{caption}</p></article>''')
    def options(values):return ''.join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in values)
    type_options=options(CATEGORY_NAMES.items())
    relation_options=options(RELATION_NAMES.items())
    record_options=options((r,r) for r in sorted({c["record"] for c in results}))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ECG 全部256例优势波形图</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#183244;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header,main{max-width:1500px;margin:auto;padding:24px}header{background:white;border-bottom:1px solid #d9e1e7}h1{margin:0;font-size:28px}
.meta{color:#627787}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}input,select{padding:9px;border:1px solid #c4d0da;border-radius:5px;background:white}
input{min-width:260px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}article{background:white;border:1px solid #dae3e9;border-radius:8px;padding:16px}
h3{font-size:16px;margin:0 0 5px}.tag{font-size:13px;color:#526a7d}img{width:100%;height:auto;display:block}p{margin:10px 0 0;font-size:14px}
details{background:white;padding:12px 16px;margin-bottom:18px}table{border-collapse:collapse}td,th{padding:6px 15px;border-bottom:1px solid #e0e7ec;text-align:left}
a{color:#1e62a1}article[hidden]{display:none}@media(max-width:500px){.grid{grid-template-columns:1fr}header,main{padding:15px}}
</style><header><h1>ECG 全部256例FP减少案例</h1><div class="meta">模型直接删除227 · RR直接删除23 · FP重新匹配为TP 6</div>
<p>红色五角星：当前复核AP（底层FP）；橙色倒三角与点线：全部原始AP；蓝色上三角与虚线：专家参考标注。点击图片查看原图。</p>
<p><a href="README.md">统计与图例说明</a> · <a href="case_descriptions.csv">全部逐例说明CSV</a> · <a href="report_captions.md">5个报告代表案例</a></p>
<p>FP净减少256，同时新增FN 80。6例匹配变化没有删除当前AP；2例容差内候选删除伴随对应专家事件新增FN，已单独标记。</p></header><main>
<div class="controls"><select id="type"><option value="">全部处理阶段</option>TYPE_OPTIONS</select><select id="record"><option value="">全部记录</option>RECORD_OPTIONS</select>
<select id="relation"><option value="">全部匹配位置关系</option>RELATION_OPTIONS</select><input id="query" placeholder="搜索AP编号、规则或说明"><span id="count"></span></div>
<section class="grid">CARDS</section></main><script>
const articles=[...document.querySelectorAll('article')];function filter(){const t=document.getElementById('type').value,r=document.getElementById('record').value,c=document.getElementById('relation').value,q=document.getElementById('query').value.toLowerCase();let n=0;
for(const a of articles){const show=(!t||a.dataset.type===t)&&(!r||a.dataset.record===r)&&(!c||a.dataset.relation===c)&&(!q||a.textContent.toLowerCase().includes(q));a.hidden=!show;if(show)n++;}document.getElementById('count').textContent=`显示 ${n} / 256 例`;}
for(const id of ['type','record','relation','query'])document.getElementById(id).addEventListener('input',filter);filter();</script></html>'''
    for key,value in (("TYPE_OPTIONS",type_options),("RECORD_OPTIONS",record_options),
                       ("RELATION_OPTIONS",relation_options),("CARDS","\n".join(cards))):
        page=page.replace(key,value)
    write_atomic(output/"index.html",page.encode())
    here=Path(__file__).resolve().parent
    for name in ("plot_benefit_cases.py","plot_error_cases.py","export_benefit_wavepack.py"):
        if (here/name).resolve()!=(output/name).resolve():shutil.copyfile(here/name,output/name)
    dimensions=Counter()
    for path in output.rglob("*.png"):
        with Image.open(path) as picture:
            dimensions[picture.size]+=1;picture.verify()
    if len(dimensions)!=1:raise ValueError("Inconsistent figure size")
    if len(list((output/"images").rglob("*.png")))!=256:raise ValueError("Missing case PNGs")
    if len(results)!=256 or len({r["image"] for r in results})!=256:raise ValueError("Incomplete gallery inventory")
    files=[{"path":p.relative_to(output).as_posix(),"bytes":p.stat().st_size,
            "sha256":base.digest_bytes(p.read_bytes())} for p in sorted(output.rglob("*"))
           if p.is_file() and p.name!="file_registry.json"]
    write_atomic(output/"file_registry.json",json.dumps({"status":"COMPLETE_256_BENEFIT_CASES",
        "source_zip_sha256":stats["source_zip_sha256"],"files":files,
        "registry_self":"Self-hash omitted to avoid recursion"},ensure_ascii=False,indent=2).encode())
    print("COMPLETE",json.dumps(stats,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("input",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--sample",action="store_true");parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);setup_font(args.output)
    z,cases,tables,pack=load_pack(args.input)
    selected=cases if not args.sample else [c for c in cases if c["case_id"] in {x[1] for x in EXAMPLES}]
    results=[]
    for i,c in enumerate(selected,1):
        path=args.output/"images"/c["benefit_type"]/c["record"]/(c["case_id"]+".png")
        category,explanation=classify(c,tables)
        ready=False
        if args.resume and path.exists():
            try:
                with Image.open(path) as picture:picture.verify()
                ready=True
            except Exception:pass
        if not ready:make_figure(c,z,tables,path)
        results.append({**c,"benefit_category_zh":CATEGORY_NAMES[category],"explanation":explanation,
                        "image":path.relative_to(args.output).as_posix()})
        if i%16==0 or i==len(selected):print(f"Rendered {i}/{len(selected)}",flush=True)
    if args.sample:save_csv(args.output/"sample_descriptions.csv",results)
    else:finish(args.input,args.output,z,results,tables,pack)
    z.close()
    print("DONE",args.output,flush=True)


if __name__=="__main__":main()


