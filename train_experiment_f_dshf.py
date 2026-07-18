"""Independent training entry for Experiment F DSHF variants."""

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
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    lfss_initialization_max_difference,
    snapshot_lfss_special_parameters,
)
from model.DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM,
    EXPERIMENT_F_VARIANTS,
    initialize_experiment_f_model,
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


VARIANT_METADATA = {
    "f1_multiscale": {
        "ablation_id": "F1",
        "model_variant": "dwtfreqnet_e1_dshf_multiscale",
        "sd_variant": "e1_dshf_multiscale",
        "output_name": "F1_multiscale",
    },
    "f2_sparse": {
        "ablation_id": "F2",
        "model_variant": "dwtfreqnet_e1_dshf_sparse",
        "sd_variant": "e1_dshf_sparse",
        "output_name": "F2_sparse",
    },
    "f3_cross_direction": {
        "ablation_id": "F3",
        "model_variant": "dwtfreqnet_e1_dshf_cross_direction",
        "sd_variant": "e1_dshf_cross_direction",
        "output_name": "F3_cross_direction",
    },
    "f4_low_guided_full": {
        "ablation_id": "F4",
        "model_variant": "dwtfreqnet_e1_dshf_low_guided_full",
        "sd_variant": "e1_dshf_low_guided_full",
        "output_name": "F4_low_guided_full",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Experiment F DSHF high-frequency ablations"
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--hf-variant", required=True, choices=EXPERIMENT_F_VARIANTS
    )
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
    parser.add_argument("--resume", default="", help="Checkpoint path, or 'auto'")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def apply_variant_metadata(args):
    metadata = VARIANT_METADATA[args.hf_variant]
    args.ablation_id = metadata["ablation_id"]
    args.model_variant = metadata["model_variant"]
    args.sd_variant = metadata["sd_variant"]


def build_model(args, mode):
    return DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM(
        get_DWTFreqNet_config(),
        hf_variant=args.hf_variant,
        mode=mode,
        deepsuper=True,
    )


def validate_checkpoint(checkpoint, args, path):
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    expected = {
        "model_variant": args.model_variant,
        "hf_variant": args.hf_variant,
        "sd_variant": args.sd_variant,
    }
    actual = {name: checkpoint_args.get(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {path}: "
            f"checkpoint={actual}, requested={expected}"
        )


def evaluate(model, loader, device, threshold):
    result = evaluate_base(model, loader, device, threshold)
    statistics = getattr(model, "last_sd_statistics", None)
    if statistics:
        result.update(statistics)
    result.update(model.lfss_scale_statistics())
    result.update({
        "dwt_calls": model.last_transform_counts["dwt"],
        "idwt_calls": model.last_transform_counts["idwt"],
    })
    return result


def parameter_count(model, prefixes):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def dshf_control_max_abs(model):
    values = []
    for stage in range(1, 5):
        block = getattr(model, f"dir_encoder{stage}")
        if block.use_sparse_gate:
            for gate in (block.sparse_h, block.sparse_v, block.sparse_d):
                last = gate.threshold_predictor[-1]
                values.extend((last.weight.detach().abs().max(), last.bias.detach().abs().max()))
        if block.use_cross_direction:
            last = block.cross_direction.gate[-1]
            values.extend((last.weight.detach().abs().max(), last.bias.detach().abs().max()))
    return max((float(value.cpu()) for value in values), default=0.0)


def main():
    args = parse_args()
    apply_variant_metadata(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Experiment F training")

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
        model = build_model(args, mode="test").to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        validate_checkpoint(checkpoint, args, args.checkpoint)
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

    model = build_model(args, mode="train").to(device)
    lfss_before = snapshot_lfss_special_parameters(model)
    initialize_experiment_f_model(model, init_weights)
    lfss_after = snapshot_lfss_special_parameters(model)
    initialization_max_difference = lfss_initialization_max_difference(
        lfss_before, lfss_after
    )
    if initialization_max_difference != 0.0:
        raise RuntimeError(
            "Baseline initialization modified Wave-Mamba LFSS special parameters: "
            f"max_abs_difference={initialization_max_difference}"
        )
    control_max_abs = dshf_control_max_abs(model)
    if control_max_abs != 0.0:
        raise RuntimeError(
            "DSHF control layers were not restored to zero initialization: "
            f"max_abs_value={control_max_abs}"
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    lfss_parameters = parameter_count(model, ("lfss_blocks.",))
    dshf_parameters = parameter_count(model, ("dir_encoder",))
    awgm_parameters = parameter_count(model, ("stage_awgm",))
    post_encoder_parameters = parameter_count(model, ("local_encoder",))
    encoder_parameters = parameter_count(
        model,
        ("stem", "local_encoder", "dir_encoder", "stage_awgm", "lfss_blocks."),
    )
    run_config = {
        "dataset": args.dataset_name,
        **model.experiment_metadata,
        "seed": args.seed,
        "epochs": args.epochs,
        "stop_after_epoch": args.stop_after_epoch,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "optimizer": "Adam",
        "initial_lr": args.lr,
        "scheduler": "10_epoch_warmup_plus_CosineAnnealingLR",
        "eta_min": 1e-5,
        "eval_start": args.eval_start,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "threshold": args.threshold,
        "random_initialization": True,
        "pretrained_checkpoint": None,
        "initialization_protection_max_abs_difference": initialization_max_difference,
        "dshf_control_initialization_max_abs_value": control_max_abs,
        "parameters": total_parameters,
        "lfss_parameters": lfss_parameters,
        "encoder_parameters": encoder_parameters,
        "dshf_parameters": dshf_parameters,
        "awgm_parameters": awgm_parameters,
        "post_encoder_parameters": post_encoder_parameters,
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
        validate_checkpoint(checkpoint, args, resume_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        load_scheduler_state_dict(scheduler, checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_mIoU", -1.0))
        print(f"Resumed from {resume_path} at epoch {start_epoch}", flush=True)

    writer = SummaryWriter(str(output_dir / "tensorboard"))
    print(json.dumps({
        "dataset": args.dataset_name,
        "hf_variant": args.hf_variant,
        "model_variant": args.model_variant,
        "sd_variant": args.sd_variant,
        "train_images": len(train_set),
        "test_images": len(test_set),
        "device": torch.cuda.get_device_name(0),
        "epochs": args.epochs,
        "parameters": total_parameters,
        "lfss_parameters": lfss_parameters,
        "dshf_parameters": dshf_parameters,
        "output_dir": str(output_dir),
    }, ensure_ascii=False), flush=True)

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
