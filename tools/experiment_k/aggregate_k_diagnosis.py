"""Lock cross-dataset K-A discovery/confirmation into K_A_DECISION.json."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REQUIRED_GROUPS = ("fidelity", "prior", "counterfactual", "treatment", "mad")
SOURCE_VARIANTS = {
    "raw_ll": "k3_gr_raw_all",
    "lfss_ll": "k4_gr_lfss_s123",
    "guided_ll": "k5_gr_guided_s123",
}


def read_jsons(paths):
    payloads = []
    for path in paths or []:
        source = Path(path)
        if source.is_file():
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload.setdefault("complete", (source.parent / "DIAGNOSIS_COMPLETE").is_file())
            payload["_summary_path"] = str(source.resolve())
            payloads.append(payload)
    return payloads


def read_rows(paths):
    rows = []
    for path in paths or []:
        source = Path(path)
        if not source.is_file():
            continue
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def group_complete(groups):
    return all(
        len(groups[name]) >= 2 and all(item.get("complete", False) for item in groups[name])
        for name in REQUIRED_GROUPS
    )


def numeric(row, key):
    value = row.get(key)
    return None if value in (None, "", "None", "nan") else float(value)


def source_support(correlation_paths):
    """Return source support using only per-dataset treatment-effect rows."""
    rows = read_rows(correlation_paths)
    positive = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("operator") != "C_GR":
            continue
        source = row.get("source")
        if source not in SOURCE_VARIANTS:
            continue
        stage = int(row["stage"])
        if source in ("lfss_ll", "guided_ll") and stage == 4:
            continue
        correlation = numeric(row, "spearman_compactness_delta_loss")
        gap = numeric(row, "treatment_group_gap")
        if correlation is not None and correlation >= 0.10 and gap is not None:
            positive[source][row.get("dataset", "unknown")].append((stage, correlation))

    supported_sources = []
    scores = {}
    stage_support = defaultdict(set)
    for source, datasets in positive.items():
        supported_datasets = {
            dataset: values for dataset, values in datasets.items() if len(values) >= 2
        }
        if len(supported_datasets) < 2:
            continue
        supported_sources.append(source)
        values = [correlation for items in supported_datasets.values() for _, correlation in items]
        scores[source] = sum(values) / len(values)
        for stage in range(1, 5):
            count = sum(any(item_stage == stage for item_stage, _ in items)
                        for items in supported_datasets.values())
            if count >= 2:
                stage_support[source].add(stage)
    return set(supported_sources), scores, stage_support


def summaries_for_record(groups):
    return {
        name: [
            {key: value for key, value in item.items() if key != "_summary_path"}
            for item in payloads
        ]
        for name, payloads in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in ("discovery", "confirmation"):
        for name in REQUIRED_GROUPS:
            parser.add_argument(f"--{name}-{phase}", nargs="*", default=[])
        parser.add_argument(f"--treatment-{phase}-correlations", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    discovery = {
        name: read_jsons(getattr(args, f"{name}_discovery")) for name in REQUIRED_GROUPS
    }
    confirmation = {
        name: read_jsons(getattr(args, f"{name}_confirmation")) for name in REQUIRED_GROUPS
    }
    discovery_complete = group_complete(discovery)
    confirmation_supplied = any(confirmation[name] for name in REQUIRED_GROUPS)
    confirmation_complete = confirmation_supplied and group_complete(confirmation)

    discovery_sources, discovery_scores, discovery_stages = source_support(
        args.treatment_discovery_correlations
    )
    confirmation_sources, _, confirmation_stages = source_support(
        args.treatment_confirmation_correlations
    )
    selected_sources = sorted(
        discovery_sources,
        key=lambda source: (-discovery_scores.get(source, float("-inf")), source),
    )
    preferred_source = selected_sources[0] if selected_sources else None
    active_stages = sorted(discovery_stages.get(preferred_source, set())) if preferred_source else []

    compactness_go = bool(
        discovery_complete
        and confirmation_complete
        and discovery_sources
        and discovery_sources <= confirmation_sources
        and all(discovery_stages[source] <= confirmation_stages[source]
                for source in discovery_sources)
    )
    variants = ["k2_dose_calibrated"]
    cancelled = {}
    reasons = ["K2不依赖Compactness结论，固定执行。"]
    for source, variant in SOURCE_VARIANTS.items():
        if compactness_go and source in discovery_sources:
            variants.append(variant)
        else:
            cancelled[variant] = "该source未在至少两个数据集的train discovery与test confirmation同时满足K-A5条件"

    # K6 is deliberately not inferred from formal test metrics.  A later
    # pre-registration may enable it only when discovery selects different
    # sources by stage and confirmation preserves that exact direction.
    cancelled["k6_gr_selected_hybrid"] = (
        "当前诊断未预注册并确认逐stage异构source；禁止事后依据正式test结果设计K6"
    )
    if compactness_go:
        reasons.append("至少两个数据集的train/test均支持Compactness预测Gaussian伤害。")
    elif discovery_complete and not confirmation_complete:
        reasons.append("train discovery已锁定；等待test confirmation，不反向修改选择。")
    else:
        reasons.append("K-A尚未完整或跨数据集处理效应证据不足，Compactness变体保持禁用。")

    decision = {
        "discovery_complete": discovery_complete,
        "confirmation_complete": confirmation_complete,
        "compactness_treatment_go": compactness_go,
        "preferred_operator": "gaussian_radial" if discovery_sources else None,
        "preferred_source": preferred_source,
        "selected_sources": selected_sources,
        "active_stages": active_stages,
        "stage4_source": preferred_source if 4 in active_stages else None,
        "rho_init": 0.05,
        "rho_max": 0.5,
        "operator_fidelity": summaries_for_record(discovery)["fidelity"],
        "treatment_effect": {
            "discovery": summaries_for_record(discovery)["treatment"],
            "confirmation": summaries_for_record(confirmation)["treatment"],
            "discovery_supported_sources": sorted(discovery_sources),
            "confirmation_supported_sources": sorted(confirmation_sources),
        },
        "prior_drift": summaries_for_record(discovery)["prior"],
        "counterfactual": summaries_for_record(discovery)["counterfactual"],
        "mad_audit": summaries_for_record(discovery)["mad"],
        "variants_to_train": variants,
        "global_candidates_for_nudt": [],
        "cancelled_variants": cancelled,
        "decision_reason": reasons,
        "selection_scope": "train discovery only; test is direction confirmation only",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(decision, ensure_ascii=False, indent=2)
    path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
