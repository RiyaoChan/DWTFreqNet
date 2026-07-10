import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage import measure
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TestSetLoader, TrainSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import AWGM_VARIANTS, DWTFreqNet
from utils import get_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DWTFreqNet on one SIRST dataset")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="./runs/default")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=0,
        help=(
            "Stop this invocation after the given epoch while retaining the "
            "scheduler configured for --epochs; 0 runs through --epochs"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-start", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="", help="Checkpoint path, or 'auto'")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Limit batches for a smoke test; 0 means no limit")
    parser.add_argument(
        "--awgm-variant",
        default="awgm_original",
        choices=AWGM_VARIANTS,
    )
    parser.add_argument(
        "--awgm-allow-fallback",
        action="store_true",
        help="Allow smoke-test-only MLP/conv backends when Mamba or DCN is missing",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip validation; intended only for short training smoke tests",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(module):
    if getattr(module, "_skip_external_init", False):
        return
    classname = module.__class__.__name__
    if "Conv" in classname and getattr(module, "weight", None) is not None:
        nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
    elif "Linear" in classname and getattr(module, "weight", None) is not None:
        nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
    elif "BatchNorm" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def final_prediction(output):
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def deep_supervision_loss(outputs, target, criterion):
    if isinstance(outputs, (tuple, list)):
        return sum(criterion(output, target) for output in outputs)
    return criterion(outputs, target)


def collect_awgm_statistics(model):
    samples = []
    for module in model.modules():
        weights = getattr(module, "last_direction_weights", None)
        attention = getattr(module, "last_attention_map", None)
        branch_norms = getattr(module, "last_branch_norms", None)
        if weights is None or attention is None:
            continue
        direction_means = weights.detach().float().mean(dim=(0, 2, 3)).cpu()
        if direction_means.numel() != 3:
            continue
        sample = {
            "mean_G_H": float(direction_means[0]),
            "mean_G_V": float(direction_means[1]),
            "mean_G_D": float(direction_means[2]),
            "attention_mean": float(attention.detach().float().mean().cpu()),
            "attention_std": float(attention.detach().float().std().cpu()),
        }
        if branch_norms:
            sample["axial_feature_norm"] = float(branch_norms["axial"])
            sample["diagonal_feature_norm"] = float(branch_norms["diagonal"])
        samples.append(sample)
    if not samples:
        return {}
    return {
        key: float(np.mean([sample[key] for sample in samples if key in sample]))
        for key in samples[0]
    }


class Metrics:
    """Metrics at one fixed threshold, matching the repository's definitions."""

    def __init__(self):
        self.intersection = 0
        self.union = 0
        self.sample_ious = []
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.matched_targets = 0
        self.targets = 0
        self.false_alarm_pixels = 0
        self.total_pixels = 0

    def update(self, probability, target, height, width, threshold):
        pred = (probability > threshold).to(torch.bool).cpu().numpy()
        gt = (target > 0.5).to(torch.bool).cpu().numpy()

        intersection = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        self.intersection += intersection
        self.union += union
        self.sample_ious.append(intersection / union if union else 1.0)

        self.tp += intersection
        self.fp += int(np.logical_and(pred, np.logical_not(gt)).sum())
        self.fn += int(np.logical_and(np.logical_not(pred), gt).sum())
        self.total_pixels += int(height * width)

        pred_regions = list(measure.regionprops(measure.label(pred.astype(np.uint8), connectivity=2)))
        gt_regions = list(measure.regionprops(measure.label(gt.astype(np.uint8), connectivity=2)))
        self.targets += len(gt_regions)
        unmatched_pred = set(range(len(pred_regions)))

        for gt_region in gt_regions:
            gt_centroid = np.asarray(gt_region.centroid)
            best_index = None
            best_distance = float("inf")
            for pred_index in unmatched_pred:
                distance = np.linalg.norm(
                    np.asarray(pred_regions[pred_index].centroid) - gt_centroid
                )
                if distance < 3 and distance < best_distance:
                    best_index = pred_index
                    best_distance = distance
            if best_index is not None:
                self.matched_targets += 1
                unmatched_pred.remove(best_index)

        self.false_alarm_pixels += sum(pred_regions[index].area for index in unmatched_pred)

    def get(self):
        precision_denominator = self.tp + self.fp
        recall_denominator = self.tp + self.fn
        precision = self.tp / precision_denominator if precision_denominator else 0.0
        recall = self.tp / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "mIoU": self.intersection / self.union if self.union else 1.0,
            "nIoU": float(np.mean(self.sample_ious)) if self.sample_ious else 0.0,
            "F1": f1,
            "Pd": self.matched_targets / self.targets if self.targets else 0.0,
            "Fa": self.false_alarm_pixels / self.total_pixels if self.total_pixels else 0.0,
        }


def evaluate(model, loader, device, threshold):
    model.eval()
    metrics = Metrics()
    criterion = nn.BCELoss()
    losses = []
    awgm_statistics = []
    with torch.no_grad():
        for image, target, size, _ in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = final_prediction(model(image))
            current_statistics = collect_awgm_statistics(model)
            if current_statistics:
                awgm_statistics.append(current_statistics)
            height, width = int(size[0]), int(size[1])
            output = output[:, :, :height, :width]
            target = target[:, :, :height, :width]
            losses.append(float(criterion(output, target).item()))
            metrics.update(output[0, 0], target[0, 0], height, width, threshold)
    result = metrics.get()
    result["loss"] = float(np.mean(losses)) if losses else 0.0
    if awgm_statistics:
        for key in awgm_statistics[0]:
            result[key] = float(np.mean([
                statistics[key] for statistics in awgm_statistics
            ]))
    model.train()
    return result


def scheduler_state_dict(scheduler):
    state = dict(scheduler.state_dict())
    after_scheduler = getattr(scheduler, "after_scheduler", None)
    if after_scheduler is not None and "after_scheduler" in state:
        state["after_scheduler"] = after_scheduler.state_dict()
        state["after_scheduler_class"] = after_scheduler.__class__.__name__
    return state


def load_scheduler_state_dict(scheduler, state):
    state = dict(state)
    after_state = state.pop("after_scheduler", None)
    state.pop("after_scheduler_class", None)

    # GradualWarmupScheduler inherits the default PyTorch state_dict behavior,
    # which serializes custom attributes.  In older checkpoints this included
    # the nested CosineAnnealingLR object itself; loading that object verbatim
    # makes it keep pointing at the optimizer instance from the old process.
    # Keep the freshly constructed nested scheduler and only load its scalar
    # state so resume updates the current optimizer.
    scheduler.load_state_dict(state)
    after_scheduler = getattr(scheduler, "after_scheduler", None)
    if after_scheduler is None or after_state is None:
        return
    if hasattr(after_state, "state_dict"):
        after_state = after_state.state_dict()
    if isinstance(after_state, dict):
        after_scheduler.load_state_dict(after_state)
        last_lrs = after_state.get("_last_lr")
    else:
        last_lrs = None
    if not last_lrs:
        last_lrs = state.get("_last_lr")
    if last_lrs:
        for param_group, lr in zip(scheduler.optimizer.param_groups, last_lrs):
            param_group["lr"] = lr
        scheduler._last_lr = list(last_lrs)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_miou, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state_dict(scheduler),
            "best_mIoU": best_miou,
            "args": vars(args),
        },
        path,
    )


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint_state_dict(checkpoint, model):
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_keys = set(model.state_dict())
    if set(state_dict) == model_keys:
        return state_dict
    for prefix in ("model.", "module.", "module.model."):
        normalized = {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }
        if set(normalized) == model_keys:
            return normalized
    missing = sorted(model_keys - set(state_dict))[:5]
    unexpected = sorted(set(state_dict) - model_keys)[:5]
    raise RuntimeError(
        "Checkpoint keys do not match the model. "
        f"Missing examples: {missing}; unexpected examples: {unexpected}"
    )


def main():
    args = parse_args()
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
        model = DWTFreqNet(
            get_DWTFreqNet_config(),
            mode="test",
            deepsuper=True,
            awgm_variant=args.awgm_variant,
            awgm_allow_fallback=args.awgm_allow_fallback,
        ).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_state_dict(checkpoint, model))
        result = evaluate(model, test_loader, device, args.threshold)
        print(json.dumps(result, indent=2), flush=True)
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

    model = DWTFreqNet(
        get_DWTFreqNet_config(),
        mode="train",
        deepsuper=True,
        awgm_variant=args.awgm_variant,
        awgm_allow_fallback=args.awgm_allow_fallback,
    ).to(device)
    model.apply(init_weights)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    awgm_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("wave_att_")
    )
    run_config = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "stop_after_epoch": args.stop_after_epoch,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "eval_start": args.eval_start,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "threshold": args.threshold,
        "awgm_variant": args.awgm_variant,
        "awgm_backends": model.awgm_backends,
        "parameters": total_parameters,
        "awgm_parameters": awgm_parameters,
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pth.tar"),
        "latest_checkpoint": str(output_dir / "latest.pth.tar"),
        "metrics_log": str(metrics_path),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    criterion = nn.BCELoss()
    optimizer_settings = {"lr": args.lr}
    scheduler_settings = {"epochs": args.epochs, "eta_min": 1e-5, "last_epoch": -1}
    optimizer, scheduler = get_optimizer(
        model, "Adam", "CosineAnnealingLR", optimizer_settings, scheduler_settings
    )

    start_epoch = 1
    best_miou = -1.0
    resume_path = output_dir / "latest.pth.tar" if args.resume == "auto" else Path(args.resume)
    if args.resume and resume_path.exists():
        # Resume files are generated locally by save_checkpoint above and include
        # scheduler objects that are not accepted by PyTorch's weights-only loader.
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
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
                "train_images": len(train_set),
                "test_images": len(test_set),
                "device": torch.cuda.get_device_name(0),
                "epochs": args.epochs,
                "output_dir": str(output_dir),
                "awgm_variant": args.awgm_variant,
                "awgm_backends": model.awgm_backends,
                "parameters": total_parameters,
                "awgm_parameters": awgm_parameters,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    end_epoch = (
        min(args.epochs, args.stop_after_epoch)
        if args.stop_after_epoch > 0 else args.epochs
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
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", record["lr"], epoch)

        should_evaluate = (
            not args.skip_evaluation
            and (
                (epoch >= args.eval_start and epoch % args.eval_every == 0)
                or epoch == end_epoch
            )
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
