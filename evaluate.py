"""
evaluate.py — Post-training evaluation script.

Loads a trained checkpoint, runs autoregressive inference on test samples,
produces visual comparisons (predicted vs ground truth fire arrival maps),
plots training curves from TensorBoard logs, and reports quantitative metrics
as formatted tables.

Usage:
    python evaluate.py --checkpoint ./checkpoints/best-model-99-0.01.pt
    python evaluate.py --checkpoint ./checkpoints/best-model-99-0.01.pt --num_samples 20
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step
from wildfire_simulator.dataloader import TrialCollection, TrialFileLoader, WildfireDataLoader
from wildfire_simulator.datasets import WildfireDataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.config import load_config


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    model = MK_UNet_Regression(
        in_channels=14,
        out_channels=2,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    epoch = checkpoint.get('epoch', -1)
    return model, epoch


def get_test_dataset():
    """Load dataset and return the test split (same split as training)."""
    dataloader = WildfireDataLoader(TrialCollection(TrialFileLoader()))
    dataset = WildfireDataset(dataloader)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    return test_set, transform, dataset


def run_inference(model, sample, transform, device, dt, max_t):
    """
    Run autoregressive rollout on a single sample.
    Returns predicted history and ground truth history at each time step.
    """
    burner = ForwardBurnProcess()

    # Transform the sample to normalized space (where t ∈ [0,1] is meaningful)
    sample_transformed = transform(sample)

    # Ground truth snapshots at each time step (burn in normalized space,
    # then inverse-transform to raw space to match pred_history)
    times = np.arange(dt, max_t, dt)
    gt_history = []
    for t in times:
        burned = burner(sample_transformed, t)
        gt_history.append(transform.inverse(burned))
    gt_history.append(sample)  # final state (raw)

    # Move sample to device and add batch dimension for inference
    # (simulator applies transform internally, expects raw data)
    sample_batched = sample.unsqueeze(0).to(device)

    # Predicted rollout using the simulator
    simulator = ForwardBurnSimulator(
        data=sample_batched,
        model=model,
        step=fire_burn_step,
        transform=transform,
        dt=dt * max_t,
        max_t=max_t,
        t0=dt * max_t
    )
    pred_history = simulator.run_to(max_t, return_history=True)

    # Remove batch dimension and move to CPU for metrics/visualization
    pred_history = [p.squeeze(0).cpu() for p in pred_history]

    return pred_history, gt_history, times


def compute_metrics(pred_frames, gt_frames):
    """
    Compute metrics between predicted and ground truth frame sequences.
    Compares channel 0 (fire mask) and channel 1 (arrival time).
    """
    metrics = {
        'mse_arrival': [],
        'mae_arrival': [],
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
    }

    for pred, gt in zip(pred_frames, gt_frames):
        pred_mask = pred[0].numpy() if isinstance(pred, torch.Tensor) else pred[0]
        gt_mask = gt[0].numpy() if isinstance(gt, torch.Tensor) else gt[0]
        pred_arr = pred[1].numpy() if isinstance(pred, torch.Tensor) else pred[1]
        gt_arr = gt[1].numpy() if isinstance(gt, torch.Tensor) else gt[1]

        # MSE / MAE on continuous values
        metrics['mse_arrival'].append(float(np.mean((pred_arr - gt_arr) ** 2)))
        metrics['mae_arrival'].append(float(np.mean(np.abs(pred_arr - gt_arr))))

        # Binarize masks (threshold at 0.5) for classification metrics
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)

        intersection = float((pred_binary * gt_binary).sum())
        union = float(((pred_binary + gt_binary) > 0).sum())
        pred_sum = float(pred_binary.sum())
        gt_sum = float(gt_binary.sum())

        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (pred_sum + gt_sum + 1e-8)
        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)

        metrics['iou'].append(iou)
        metrics['dice'].append(dice)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

    # Average over time steps
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def save_arrival_comparison(pred_history, gt_history, sample_idx, output_dir):
    """Save side-by-side arrival time comparison at multiple time steps."""
    n_steps = len(pred_history)
    indices = np.linspace(0, n_steps - 1, min(8, n_steps), dtype=int)

    fig, axes = plt.subplots(3, len(indices), figsize=(len(indices) * 3, 9), squeeze=False)

    for col, idx in enumerate(indices):
        pred = pred_history[idx]
        gt = gt_history[idx] if idx < len(gt_history) else gt_history[-1]

        pred_mask = pred[0].numpy() if isinstance(pred, torch.Tensor) else pred[0]
        gt_mask = gt[0].numpy() if isinstance(gt, torch.Tensor) else gt[0]

        axes[0, col].imshow(np.where(pred_mask == 0, np.nan, pred_mask), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        axes[0, col].set_title(f"t={idx}", fontsize=8)
        axes[0, col].axis('off')

        axes[1, col].imshow(np.where(gt_mask == 0, np.nan, gt_mask), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        axes[1, col].axis('off')

        error = np.abs(pred_mask - gt_mask)
        axes[2, col].imshow(error, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("Predicted", fontsize=10)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=10)
    axes[2, 0].set_ylabel("Error", fontsize=10)

    fig.suptitle(f"Sample {sample_idx} — Fire Mask Rollout", fontsize=12)
    fig.tight_layout()

    filepath = os.path.join(output_dir, f"comparison_sample_{sample_idx:03d}.png")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


def save_final_arrival_map(pred_history, gt_history, sample_idx, output_dir):
    """Save final predicted vs ground truth arrival time map."""
    pred_final = pred_history[-1]
    gt_final = gt_history[-1]

    pred_arr = pred_final[1].numpy() if isinstance(pred_final, torch.Tensor) else pred_final[1]
    gt_arr = gt_final[1].numpy() if isinstance(gt_final, torch.Tensor) else gt_final[1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(np.where(pred_arr == 0, np.nan, pred_arr), cmap='YlOrRd', aspect='auto')
    axes[0].set_title("Predicted Arrival Time")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.where(gt_arr == 0, np.nan, gt_arr), cmap='YlOrRd', aspect='auto')
    axes[1].set_title("Ground Truth Arrival Time")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    error = np.abs(pred_arr - gt_arr)
    im2 = axes[2].imshow(error, cmap='Reds', aspect='auto')
    axes[2].set_title(f"Absolute Error (MAE={error.mean():.4f})")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Sample {sample_idx} — Final Arrival Time Map", fontsize=12)
    fig.tight_layout()

    filepath = os.path.join(output_dir, f"arrival_map_sample_{sample_idx:03d}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


def plot_training_curves(logdir, output_dir):
    """Load TensorBoard logs and plot training/validation loss curves.
    Automatically detects and plots per-channel losses if available."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [!] tensorboard not installed, skipping training curve plot")
        return None, None, None

    train_dir = os.path.join(logdir, "train")
    val_dir = os.path.join(logdir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"  [!] Training logs not found at {logdir}")
        return None, None, None

    train_acc = EventAccumulator(train_dir)
    train_acc.Reload()
    val_acc = EventAccumulator(val_dir)
    val_acc.Reload()

    train_scalars = train_acc.Scalars("Loss")
    val_scalars = val_acc.Scalars("Loss")

    train_steps = [s.step for s in train_scalars]
    train_values = [s.value for s in train_scalars]
    val_steps = [s.step for s in val_scalars]
    val_values = [s.value for s in val_scalars]

    # Detect per-channel losses
    available_tags = train_acc.Tags().get('scalars', [])
    has_component_losses = 'Loss/mask_bce' in available_tags or 'Loss/arrival_mse' in available_tags

    if has_component_losses:
        # Plot combined + per-channel on separate subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Combined loss
        axes[0].plot(train_steps, train_values, label='Train', linewidth=1.5)
        axes[0].plot(val_steps, val_values, label='Val', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mask BCE
        if 'Loss/mask_bce' in available_tags:
            train_mask = train_acc.Scalars("Loss/mask_bce")
            val_mask = val_acc.Scalars("Loss/mask_bce") if 'Loss/mask_bce' in val_acc.Tags().get('scalars', []) else []
            axes[1].plot([s.step for s in train_mask], [s.value for s in train_mask], label='Train', linewidth=1.5)
            if val_mask:
                axes[1].plot([s.step for s in val_mask], [s.value for s in val_mask], label='Val', linewidth=1.5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('BCE')
            axes[1].set_title('Mask Loss (BCE)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Arrival MSE
        if 'Loss/arrival_mse' in available_tags:
            train_arr = train_acc.Scalars("Loss/arrival_mse")
            val_arr = val_acc.Scalars("Loss/arrival_mse") if 'Loss/arrival_mse' in val_acc.Tags().get('scalars', []) else []
            axes[2].plot([s.step for s in train_arr], [s.value for s in train_arr], label='Train', linewidth=1.5)
            if val_arr:
                axes[2].plot([s.step for s in val_arr], [s.value for s in val_arr], label='Val', linewidth=1.5)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('MSE')
            axes[2].set_title('Arrival Time Loss (MSE)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        fig.suptitle('Training & Validation Loss', fontsize=13)
        fig.tight_layout()
    else:
        # Single combined plot (legacy / BCELoss only)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_steps, train_values, label='Train Loss', linewidth=1.5)
        ax.plot(val_steps, val_values, label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if len(train_values) > 0 and max(train_values) / (min(train_values) + 1e-10) > 100:
            ax.set_yscale('log')

    filepath = os.path.join(output_dir, "training_curves.png")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return filepath, (train_steps, train_values), (val_steps, val_values)


def format_table(headers, rows, col_width=14):
    """Format a list of rows into an aligned text table."""
    sep = "+" + "+".join(["-" * (col_width + 2)] * len(headers)) + "+"
    header_row = "|" + "|".join(f" {h:^{col_width}} " for h in headers) + "|"
    lines = [sep, header_row, sep]
    for row in rows:
        formatted = []
        for val in row:
            if isinstance(val, float):
                formatted.append(f" {val:>{col_width}.6f} ")
            else:
                formatted.append(f" {str(val):>{col_width}} ")
        lines.append("|" + "|".join(formatted) + "|")
    lines.append(sep)
    return "\n".join(lines)


def build_per_sample_table(all_metrics):
    """Build a table with one row per sample."""
    headers = ["Sample", "IoU", "Dice", "Precision", "Recall", "MAE Arrival"]
    rows = []
    for i, m in enumerate(all_metrics):
        rows.append([
            i, m['iou'], m['dice'], m['precision'], m['recall'],
            m['mae_arrival']
        ])
    return format_table(headers, rows)


def build_summary_table(all_metrics):
    """Build a summary table with mean and std across all samples."""
    headers = ["Metric", "Mean", "Std", "Min", "Max"]
    keys = ['mse_arrival', 'mae_arrival',
            'iou', 'dice', 'precision', 'recall']
    rows = []
    for key in keys:
        values = [m[key] for m in all_metrics]
        rows.append([key, np.mean(values), np.std(values), np.min(values), np.max(values)])
    return format_table(headers, rows)


def build_training_table(train_data, val_data):
    """Build a table summarizing training curve stats."""
    train_steps, train_values = train_data
    val_steps, val_values = val_data

    headers = ["Statistic", "Train Loss", "Val Loss"]
    rows = [
        ["Total Epochs", len(train_steps), len(val_steps)],
        ["Final", train_values[-1], val_values[-1]],
        ["Best", min(train_values), min(val_values)],
        ["Best Epoch", train_steps[int(np.argmin(train_values))],
                       val_steps[int(np.argmin(val_values))]],
        ["Δ (final)", train_values[-1] - val_values[-1], "—"],
    ]
    return format_table(headers, rows)


def save_metrics_csv(all_metrics, filepath):
    """Save per-sample metrics as CSV."""
    keys = ['mse_arrival', 'mae_arrival',
            'iou', 'dice', 'precision', 'recall']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample'] + keys)
        for i, m in enumerate(all_metrics):
            writer.writerow([i] + [m[k] for k in keys])


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained wildfire model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation outputs')
    parser.add_argument('--logdir', type=str, default='./training',
                        help='TensorBoard log directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to evaluate')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Time step for rollout (default: from config)')
    parser.add_argument('--max_t', type=float, default=None,
                        help='Maximum time for rollout (default: from config)')
    args = parser.parse_args()

    config = load_config(args.config)
    # CLI args override config if provided
    dt = args.dt if args.dt is not None else config['dt']
    max_t = args.max_t if args.max_t is not None else config['max_t']

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ─── Load Model ───────────────────────────────────────────────────────
    print(f"Checkpoint: {args.checkpoint}")
    model, ckpt_epoch = load_model(args.checkpoint, device)
    print(f"Loaded epoch: {ckpt_epoch}")

    # ─── Load Data ────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    test_set, transform, full_dataset = get_test_dataset()
    print(f"  Dataset size:  {len(full_dataset)}")
    print(f"  Test split:    {len(test_set)}")

    # ─── Training Curves ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" TRAINING CURVES")
    print("=" * 60)

    curves_path, train_data, val_data = plot_training_curves(args.logdir, args.output_dir)
    if curves_path and train_data and val_data:
        print(build_training_table(train_data, val_data))
        print(f"\n  Saved: {curves_path}")
    else:
        print("  No training logs found.")

    # ─── Inference & Metrics ──────────────────────────────────────────────
    num_samples = min(args.num_samples, len(test_set))
    print("\n" + "=" * 60)
    print(f" INFERENCE ({num_samples} test samples)")
    print("=" * 60)

    all_metrics = []
    for i in range(num_samples):
        sample = test_set[i]
        print(f"  [{i + 1:3d}/{num_samples}] Running rollout...", end=" ", flush=True)

        pred_history, gt_history, times = run_inference(
            model, sample, transform, device, dt, max_t
        )

        # Align lengths
        min_len = min(len(pred_history), len(gt_history))
        metrics = compute_metrics(pred_history[:min_len], gt_history[:min_len])
        all_metrics.append(metrics)

        print(f"IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
              f"MAE_arr={metrics['mae_arrival']:.4f}")

        # Save visuals
        save_arrival_comparison(pred_history, gt_history, i, args.output_dir)
        save_final_arrival_map(pred_history, gt_history, i, args.output_dir)

    # ─── Per-Sample Table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PER-SAMPLE METRICS")
    print("=" * 60)
    print(build_per_sample_table(all_metrics))

    # ─── Summary Table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY (across all samples)")
    print("=" * 60)
    print(build_summary_table(all_metrics))

    # ─── Save to Disk ─────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    save_metrics_csv(all_metrics, csv_path)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {ckpt_epoch}\n")
        f.write(f"Samples evaluated: {num_samples}\n")
        f.write(f"dt: {dt}, max_t: {max_t}\n\n")
        f.write("SUMMARY\n")
        f.write(build_summary_table(all_metrics) + "\n\n")
        f.write("PER-SAMPLE\n")
        f.write(build_per_sample_table(all_metrics) + "\n")

    print(f"\n  Results saved:")
    print(f"    Metrics CSV:    {csv_path}")
    print(f"    Summary:        {summary_path}")
    print(f"    Visuals:        {args.output_dir}/")


if __name__ == "__main__":
    main()
