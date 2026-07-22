"""Aggregate all Phase 1 outputs into the predefined decision matrix/report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.phase1.common import ensure_dir, read_json, runtime_metadata, write_csv, write_json


DATASETS = ("NUAA-SIRST", "IRSTD-1K", "NUDT-SIRST")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_summary(root, task, dataset, split="test"):
    mapping = {
        "P1": root / "P1_gaussian_geometry" / dataset / split / "summary.json",
        "P2": root / "P2_wavelet_consistency" / dataset / split / "summary.json",
        "P3": root / "P3_sampling_geometry" / dataset / split / "summary.json",
        "H": root / "H_cross_analysis" / dataset / "summary.json",
    }
    return read_json(mapping[task]), mapping[task]


def cross_dataset_decision(decisions):
    go_count = sum(value == "Go" for value in decisions.values())
    partial_count = sum(value == "Partial Go" for value in decisions.values())
    if go_count >= 2:
        return "Go"
    if go_count or partial_count:
        return "Partial Go"
    if all(value == "No-Go" for value in decisions.values() if value):
        return "No-Go"
    return "Pending"


def phase2_actions(p1, p2, p3):
    if "Pending" in (p1, p2, p3):
        return {
            "case": "Pending", "allowed": [],
            "cancelled": ["在 Phase 1 完成前不启动 I1/I2/I3/I4/I5/GCSWR"],
        }
    if p1 == "Go" and p2 == "Go" and p3 == "Go":
        return {"case": "A", "allowed": ["I1-G vs I1-S", "I2自适应椭圆螺旋", "独立I3 Gaussian频率一致性"], "cancelled": []}
    if p1 == "Go" and p2 == "Go" and p3 == "No-Go":
        return {"case": "B", "allowed": ["I3", "局部成对/环形算子"], "cancelled": ["螺旋分支", "I1-S", "I2"]}
    if p1 == "Go" and p2 == "No-Go" and p3 == "Go":
        return {"case": "C", "allowed": ["I1/I2几何采样"], "cancelled": ["H/V/D相位一致性"]}
    if p1 == "No-Go" and p2 == "Go" and p3 in ("Go", "Partial Go"):
        return {"case": "D", "allowed": ["方向一致性", "stage/数据驱动结构化采样"], "cancelled": ["Gaussian sigma/ellipse预测"]}
    if p1 == p2 == p3 == "No-Go":
        return {"case": "E", "allowed": ["回到E1/H已验证结构"], "cancelled": ["任务先验路线", "GCSWR"]}
    return {
        "case": "Mixed/Partial", "allowed": ["仅允许各任务被数据支持的子部分，并加入先验置信度"],
        "cancelled": ["未获得跨数据集支持的显式先验模块"],
    }


def main():
    args = parse_args()
    root = Path(args.analysis_root)
    output_dir = ensure_dir(args.output_dir or root / "final")
    dataset_results, missing = {}, []
    for dataset in DATASETS:
        dataset_results[dataset] = {}
        for task in ("P1", "P2", "P3", "H"):
            summary, path = load_summary(root, task, dataset)
            dataset_results[dataset][task] = summary
            if summary is None:
                missing.append(str(path))

    p1_dataset = {
        dataset: (dataset_results[dataset]["P1"] or {}).get("decision", "Pending")
        for dataset in DATASETS
    }
    p2_directional_dataset = {
        dataset: (dataset_results[dataset]["P2"] or {}).get("directional_decision", "Pending")
        for dataset in DATASETS
    }
    p2_low_dataset = {
        dataset: (dataset_results[dataset]["P2"] or {}).get("low_frequency_decision", "Pending")
        for dataset in DATASETS
    }
    p2_dataset = {}
    for dataset in DATASETS:
        values = (p2_directional_dataset[dataset], p2_low_dataset[dataset])
        p2_dataset[dataset] = "Pending" if any(
            value in ("Pending", "Descriptive only") for value in values
        ) else "Go" if values == ("Go", "Go") else (
            "No-Go" if values == ("No-Go", "No-Go") else "Partial Go"
        )
    p3_dataset = {
        dataset: (dataset_results[dataset]["P3"] or {}).get("decision", "Pending")
        for dataset in DATASETS
    }
    decisions = {
        "P1": cross_dataset_decision(p1_dataset),
        "P2": cross_dataset_decision(p2_dataset),
        "P2_directional": cross_dataset_decision(p2_directional_dataset),
        "P2_low_frequency": cross_dataset_decision(p2_low_dataset),
        "P3": cross_dataset_decision(p3_dataset),
    }
    actions = phase2_actions(decisions["P1"], decisions["P2"], decisions["P3"])
    summary = {
        "phase": "Phase 1 IRST task-prior validation",
        "complete": not missing,
        "missing_outputs": missing,
        "dataset_decisions": {
            "P1": p1_dataset, "P2": p2_dataset,
            "P2_directional": p2_directional_dataset,
            "P2_low_frequency": p2_low_dataset, "P3": p3_dataset,
        },
        "cross_dataset_decisions": decisions,
        "decision_case": actions["case"],
        "phase2_allowed": actions["allowed"],
        "experiments_cancelled": actions["cancelled"],
        "dataset_results": dataset_results,
        "runtime": runtime_metadata(command=" ".join(sys.argv)),
    }
    write_json(output_dir / "PHASE1_SUMMARY.json", summary)
    table_rows = []
    for dataset in DATASETS:
        table_rows.append({
            "dataset": dataset,
            "P1": p1_dataset[dataset],
            "P2": p2_dataset[dataset],
            "P2_directional": p2_directional_dataset[dataset],
            "P2_low_frequency": p2_low_dataset[dataset],
            "P3": p3_dataset[dataset],
        })
    write_csv(output_dir / "PHASE1_SUMMARY.csv", table_rows)

    matrix = [
        "# Phase 1 联合决策矩阵", "",
        "| 数据集 | P1 Gaussian/椭圆 | P2 H/V/D与低频 | P3采样几何 |",
        "|---|---|---|---|",
    ]
    for row in table_rows:
        matrix.append(f"| {row['dataset']} | {row['P1']} | {row['P2']} | {row['P3']} |")
    matrix += [
        "", f"跨数据集结论：P1 **{decisions['P1']}**，P2 **{decisions['P2']}**，P3 **{decisions['P3']}**。",
        "", f"决策 Case：**{actions['case']}**", "",
        "## Phase 2 允许启动", "",
    ]
    matrix += [f"- {item}" for item in actions["allowed"]] or ["- 无"]
    matrix += ["", "## 明确取消/暂停", ""]
    matrix += [f"- {item}" for item in actions["cancelled"]] or ["- 无"]
    matrix += ["", "## 缺失输出", ""]
    matrix += [f"- `{path}`" for path in missing] or ["- 无"]
    (output_dir / "PHASE1_DECISION_MATRIX.md").write_text("\n".join(matrix) + "\n", encoding="utf-8")

    report = [
        "# Phase 1 红外小目标任务先验验证报告", "",
        "## 1. 数据与实例统计", "",
    ]
    for dataset in DATASETS:
        p1 = dataset_results[dataset]["P1"] or {}
        report.append(
            f"- {dataset}：图像 {p1.get('images', '待完成')}；样本统计 `{p1.get('class_counts', {})}`。"
        )
    report += ["", "## 2. H/V/D方向校准", ""]
    calibrations = [
        (dataset_results[dataset]["P2"] or {}).get("calibration")
        for dataset in DATASETS
    ]
    calibration = next((item for item in calibrations if item), None)
    if calibration:
        report += [
            f"- H：{calibration['band_response_orientation']['H']}响应；规则 `{calibration['pair_rules']['H']}`。",
            f"- V：{calibration['band_response_orientation']['V']}响应；规则 `{calibration['pair_rules']['V']}`。",
            f"- D：象限模板 `{calibration['d_quadrant_template']}`。",
        ]
    else:
        report.append("- 待P2校准输出。")
    report += ["", "## 3. P1完整结果", ""]
    report += [f"- {dataset}：{p1_dataset[dataset]}" for dataset in DATASETS]
    report += ["", "## 4. P2完整结果", ""]
    report += [
        f"- {dataset}：方向 {p2_directional_dataset[dataset]}；低频 {p2_low_dataset[dataset]}；综合 {p2_dataset[dataset]}。"
        for dataset in DATASETS
    ]
    report += ["", "## 5. P3完整结果", ""]
    report += [f"- {dataset}：{p3_dataset[dataset]}" for dataset in DATASETS]
    report += ["", "## 6. Experiment H交叉分析", ""]
    for dataset in DATASETS:
        h = dataset_results[dataset]["H"]
        report.append(f"- {dataset}：{'已完成' if h else '待完成'}。")
    report += [
        "", "## 7. 每个数据集单独结论", "",
        *[f"- {row['dataset']}：P1={row['P1']}，P2={row['P2']}，P3={row['P3']}。" for row in table_rows],
        "", "## 8. 跨数据集结论", "",
        f"- P1：{decisions['P1']}", f"- P2：{decisions['P2']}", f"- P3：{decisions['P3']}",
        "", "## 9. 反例与失败样本", "",
        "- 由各任务 `visuals/selected_examples.json`、采样失败图及完整非显著统计表保留；不删除失败结果。",
        "", "## 10. Go/Partial-Go/No-Go判定", "",
        f"- 联合决策：Case {actions['case']}。",
        "", "## 11. Phase 2允许启动的实验", "",
        *([f"- {item}" for item in actions['allowed']] or ["- 无"]),
        "", "## 12. 明确取消的实验", "",
        *([f"- {item}" for item in actions['cancelled']] or ["- 无"]),
        "", "## 13. 所有输出文件索引", "",
        "- `PHASE1_SUMMARY.json`", "- `PHASE1_SUMMARY.csv`",
        "- `PHASE1_DECISION_MATRIX.md`", "- 各P1/P2/P3/H目录的CSV、JSON与visuals。",
    ]
    if missing:
        report += ["", "> 当前为阶段性报告；缺失项完成后重新运行聚合脚本。"]
    (output_dir / "PHASE1_TASK_PRIOR_VALIDATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "complete": summary["complete"], "decisions": decisions,
        "case": actions["case"], "missing": missing,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
