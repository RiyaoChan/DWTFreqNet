"""Trainer for Experiment D decoder-HFE relation ablations D2, D3 and D4."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TestSetLoader, TrainSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
    DWTFreqNet_SingleDecoder_HFE_Ablation,
    HFE_ABLATION_VARIANTS,
)
from train_one import (
    append_jsonl,
    checkpoint_state_dict,
    deep_supervision_loss,
    evaluate as evaluate_base,
    init_weights,
    load_scheduler_state_dict,
    save_checkpoint,
    set_seed,
)
from utils import get_optimizer


VARIANT_IDENTITY = {
    "d2_softcos_all": {
        "ablation_id": "D2",
        "model_variant": "dwtfreqnet_single_decoder_hfe_softcos",
        "sd_variant": "sd_awgm_hfe_softcos",
    },
    "d3_scaleaware": {
        "ablation_id": "D3",
        "model_variant": "dwtfreqnet_single_decoder_hfe_scaleaware",
        "sd_variant": "sd_awgm_hfe_scaleaware",
    },
    "d4_no_matching": {
        "ablation_id": "D4",
        "model_variant": "dwtfreqnet_single_decoder_hfe_nomatch",
        "sd_variant": "sd_awgm_hfe_nomatch",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Experiment D HFE relation ablation D2, D3 or D4"
    )
    parser.add_argument(
        "--hfe-ablation", required=True, choices=HFE_ABLATION_VARIANTS
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-start", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def build_model(mode, hfe_ablation):
    return DWTFreqNet_SingleDecoder_HFE_Ablation(
        get_DWTFreqNet_config(),
        hfe_ablation=hfe_ablation,
        mode=mode,
        deepsuper=True,
    )


def validate_checkpoint(checkpoint, path, hfe_ablation):
    identity = VARIANT_IDENTITY[hfe_ablation]
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    observed = {
        "model_variant": checkpoint_args.get("model_variant"),
        "sd_variant": checkpoint_args.get("sd_variant"),
        "hfe_ablation": checkpoint_args.get("hfe_ablation"),
    }
    expected = {
        "model_variant": identity["model_variant"],
        "sd_variant": identity["sd_variant"],
        "hfe_ablation": hfe_ablation,
    }
    if observed != expected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {path}: "
            f"observed={observed}, expected={expected}."
        )


def evaluate(model, loader, device, threshold):
    result = evaluate_base(model, loader, device, threshold)
    statistics = getattr(model, "last_sd_statistics", None)
    if statistics:
        result.update(statistics)
    result.update(
        {
            "dwt_calls": model.last_transform_counts["dwt"],
            "idwt_calls": model.last_transform_counts["idwt"],
        }
    )
    return result


def parameter_count(model, prefixes):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def main():
    args = parse_args()
    identity = VARIANT_IDENTITY[args.hfe_ablation]
    args.ablation_id = identity["ablation_id"]
    args.model_variant = identity["model_variant"]
    args.sd_variant = identity["sd_variant"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    test_set = TestSetLoader(
        args.dataset_dir, args.dataset_name, args.dataset_name, img_norm_cfg=None
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --eval-only")
        model = build_model("test", args.hfe_ablation).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        validate_checkpoint(checkpoint, args.checkpoint, args.hfe_ablation)
        model.load_state_dict(checkpoint_state_dict(checkpoint, model))
        print(json.dumps(evaluate(model, test_loader, device, args.threshold), indent=2))
        return

    train_set = TrainSetLoader(
        args.dataset_dir,
        args.dataset_name,
        patch_size=args.patch_size,
        img_norm_cfg=None,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_model("train", args.hfe_ablation).to(device)
    model.apply(init_weights)
    for key, value in model.experiment_metadata.items():
        setattr(args, key, value)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    direction_parameters = parameter_count(model, ("dir_encoder",))
    awgm_parameters = parameter_count(model, ("stage_awgm",))
    hfe_parameters = parameter_count(model, ("decoder_hfe",))
    relation_parameters = parameter_count(
        model,
        tuple(
            f"decoder_hfe{stage}.hfe.{branch}.relation"
            for stage in range(1, 5)
            for branch in ("attn", "ffn")
        ),
    )
    run_config = {
        "dataset": args.dataset_name,
        **model.experiment_metadata,
        "seed": args.seed,
        "epochs": args.epochs,
        "stop_after_epoch": args.stop_after_epoch,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "eval_start": args.eval_start,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "threshold": args.threshold,
        "parameters": total_parameters,
        "direction_parameters": direction_parameters,
        "awgm_parameters": awgm_parameters,
        "hfe_parameters": hfe_parameters,
        "relation_parameters": relation_parameters,
        "dwt_calls": 4,
        "idwt_calls": 4,
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pth.tar"),
        "latest_checkpoint": str(output_dir / "latest.pth.tar"),
        "metrics_log": str(metrics_path),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    criterion = nn.BCELoss()
    optimizer, scheduler = get_optimizer(
        model,
        "Adam",
        "CosineAnnealingLR",
        {"lr": args.lr},
        {"epochs": args.epochs, "eta_min": 1e-5, "last_epoch": -1},
    )
    start_epoch = 1
    best_miou = -1.0
    resume_path = (
        output_dir / "latest.pth.tar" if args.resume == "auto" else Path(args.resume)
    )
    if args.resume and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        validate_checkpoint(checkpoint, resume_path, args.hfe_ablation)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        load_scheduler_state_dict(scheduler, checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_mIoU", -1.0))
        print(f"Resumed from {resume_path} at epoch {start_epoch}", flush=True)

    writer = SummaryWriter(str(output_dir / "tensorboard"))
    print(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "ablation_id": identity["ablation_id"],
                "hfe_ablation": args.hfe_ablation,
                "model_variant": identity["model_variant"],
                "sd_variant": identity["sd_variant"],
                "train_images": len(train_set),
                "test_images": len(test_set),
                "device": torch.cuda.get_device_name(0),
                "epochs": args.epochs,
                "parameters": total_parameters,
                "hfe_parameters": hfe_parameters,
                "relation_parameters": relation_parameters,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    end_epoch = (
        min(args.epochs, args.stop_after_epoch)
        if args.stop_after_epoch > 0
        else args.epochs
    )
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        for batch_index, (image, target) in enumerate(train_loader, start=1):
            if image.shape[0] == 1:
                continue
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = deep_supervision_loss(model(image), target, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}: {loss}"
                )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        writer.add_scalar("train/loss", record["train_loss"], epoch)
        writer.add_scalar("train/lr", record["lr"], epoch)

        should_evaluate = not args.skip_evaluation and (
            (epoch >= args.eval_start and epoch % args.eval_every == 0)
            or epoch == end_epoch
        )
        if should_evaluate:
            result = evaluate(model, test_loader, device, args.threshold)
            record.update(result)
            for name, value in result.items():
                writer.add_scalar(f"test/{name}", value, epoch)
            if result["mIoU"] > best_miou:
                best_miou = result["mIoU"]
                save_checkpoint(
                    output_dir / "best.pth.tar",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_miou,
                    args,
                )
                best_record = dict(record)
                best_record["checkpoint"] = str(output_dir / "best.pth.tar")
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(best_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        append_jsonl(metrics_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if epoch % args.save_every == 0 or epoch == end_epoch:
            save_checkpoint(
                output_dir / "latest.pth.tar",
                epoch,
                model,
                optimizer,
                scheduler,
                best_miou,
                args,
            )
        writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
