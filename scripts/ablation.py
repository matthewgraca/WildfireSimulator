"""
ablation.py — Mean ablation study on input features.

For each feature channel, replaces it with its training set mean and measures
the impact on model performance. A large IoU drop indicates the feature is 
important; no change means the model ignores it.

Usage:
    python scripts/ablation.py /mnt/wildfire/surrogate-model/experiment
    python scripts/ablation.py /mnt/wildfire/surrogate-model/experiment --num_samples 20
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step
from wildfire_simulator.dataloader import TrialCollection, TrialFileLoader, WildfireDataLoader
from wildfire_simulator.datasets import WildfireDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.config import load_config


CHANNEL_NAMES = [
    "Fire Mask",          # 0
    "Fire Arrival Time",  # 1
    "Elevation",          # 2
    "Slope",              # 3
    "Aspect",             # 4
    "Fuel Model (FBFM40)",# 5
    "Canopy Cover",       # 6
    "Stand Height",       # 7
    "Canopy Base Height", # 8
    "Canopy Bulk Density",# 9
    "Wind U",             # 10
    "Wind V",             # 11
    "Foliar Moisture",    # 12
]

# Channels to ablate (all 13 including mask and FAT)
ABLATION_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    model = MK_UNet_Regression(
        in_channels=14, out_channels=1,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def compute_channel_means(dataset, train_indices):
    """Compute per-channel mean from the training set."""
    means = torch.zeros(13)
    count = 0
    for idx in train_indices:
        sample = dataset[idx]
        means += sample.mean(dim=(1, 2))
        count += 1
    means /= count
    return means


def run_inference_ablated(model, sample, transform, device, dt, max_t, ablate_channel=None, channel_mean=None):
    """Run inference with an optional channel ablated (replaced with its mean)."""
    if ablate_channel is not None and channel_mean is not None:
        sample = sample.clone()
        sample[ablate_channel] = channel_mean

    burner = ForwardBurnProcess()
    sample_transformed = transform(sample)

    # Ground truth
    times = np.arange(dt, max_t, dt)
    gt_history = []
    for t in times:
        burned = burner(sample_transformed, t)
        gt_history.append(transform.inverse(burned))
    gt_history.append(sample)

    # Run simulator
    sample_batched = sample.unsqueeze(0).to(device)
    simulator = ForwardBurnSimulator(
        data=sample_batched, model=model, step=fire_burn_step,
        transform=transform, dt=dt * max_t, max_t=max_t, t0=dt * max_t
    )
    pred_history = simulator.run_to(max_t, return_history=True)
    pred_history = [p.squeeze(0).cpu() for p in pred_history]

    return pred_history, gt_history


def compute_metrics(pred_frames, gt_frames):
    """Compute IoU, F1, Precision, Recall averaged across time steps."""
    ious, f1s, precisions, recalls = [], [], [], []

    for pred, gt in zip(pred_frames, gt_frames):
        pred_mask = pred[0].numpy() if isinstance(pred, torch.Tensor) else pred[0]
        gt_mask = gt[0].numpy() if isinstance(gt, torch.Tensor) else gt[0]

        pred_binary = (pred_mask > 0.5).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)

        intersection = float((pred_binary * gt_binary).sum())
        union = float(((pred_binary + gt_binary) > 0).sum())
        pred_sum = float(pred_binary.sum())
        gt_sum = float(gt_binary.sum())

        iou = intersection / (union + 1e-8)
        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        ious.append(iou)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    return {
        'iou': float(np.mean(ious)),
        'f1': float(np.mean(f1s)),
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
    }


def evaluate_condition(model, test_set, transform, device, dt, max_t, num_samples,
                       ablate_channel=None, channel_mean=None, label=""):
    """Evaluate model on test set with optional ablation."""
    all_metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': []}

    for i in tqdm(range(num_samples), desc=f"  {label:<23s}", leave=True):
        sample = test_set[i]
        pred_history, gt_history = run_inference_ablated(
            model, sample, transform, device, dt, max_t,
            ablate_channel=ablate_channel, channel_mean=channel_mean
        )
        min_len = min(len(pred_history), len(gt_history))
        m = compute_metrics(pred_history[:min_len], gt_history[:min_len])
        for k in all_metrics:
            all_metrics[k].append(m[k])

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    print(f"  {'':25s} IoU={avg['iou']:.4f}  F1={avg['f1']:.4f}  Prec={avg['precision']:.4f}  Recall={avg['recall']:.4f}")
    return avg


def main():
    parser = argparse.ArgumentParser(description="Mean ablation study on input features")
    parser.add_argument('experiment_dir', type=str,
                        help='Experiment directory (auto-detects best checkpoint)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to evaluate')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dt = config['dt']
    max_t = config['max_t']

    # Find best checkpoint
    ckpt_dir = os.path.join(args.experiment_dir, 'checkpoints')
    ckpt_files = sorted(Path(ckpt_dir).glob("best-model-*.pt"))
    if not ckpt_files:
        print(f"Error: No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    def parse_val_loss(p):
        try:
            return float(p.stem.split('-')[-1])
        except ValueError:
            return float('inf')

    checkpoint_path = str(min(ckpt_files, key=parse_val_loss))

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Load dataset with same split as training
    dataloader = WildfireDataLoader(TrialCollection(TrialFileLoader()))
    dataset = WildfireDataset(dataloader)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)

    num_samples = min(args.num_samples, len(test_set))
    print(f"Samples: {num_samples} from test set")
    print()

    # Compute per-channel means from training set
    print("Computing training set channel means...")
    channel_means = compute_channel_means(dataset, train_set.indices)
    print()

    # --- Baseline (no ablation) ---
    print("=" * 70)
    print(" ABLATION STUDY")
    print("=" * 70)

    baseline = evaluate_condition(
        model, test_set, transform, device, dt, max_t, num_samples,
        label="Baseline (no ablation)"
    )

    # --- Ablation per channel ---
    results = [("Baseline", baseline)]

    for ch in ABLATION_CHANNELS:
        avg = evaluate_condition(
            model, test_set, transform, device, dt, max_t, num_samples,
            ablate_channel=ch, channel_mean=channel_means[ch],
            label=CHANNEL_NAMES[ch]
        )
        results.append((CHANNEL_NAMES[ch], avg))

    # --- Save CSV ---
    csv_path = os.path.join(args.experiment_dir, "ablation_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'iou', 'f1', 'precision', 'recall'])
        for name, metrics in results:
            writer.writerow([name, metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall']])
    print(f"\n  CSV saved: {csv_path}")

    # --- Bar chart (residuals relative to baseline) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Compute residuals (ablated IoU - baseline IoU); negative = feature is important
    ablated_results = [(name, metrics) for name, metrics in results if name != "Baseline"]
    # Sort by residual (most negative = most important, at top)
    ablated_sorted = sorted(ablated_results, key=lambda x: x[1]['iou'] - baseline['iou'])

    names = [r[0] for r in ablated_sorted]
    residuals = [r[1]['iou'] - baseline['iou'] for r in ablated_sorted]

    # Color: red for negative (important), green for positive (hurts performance when present)
    colors = ['#D32F2F' if r < 0 else '#388E3C' for r in residuals]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(names)), residuals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('ΔIoU (ablated − baseline)')
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_title(f'Mean Ablation Study — Feature Importance\nBaseline IoU = {baseline["iou"]:.4f} (negative = feature is important)')
    ax.grid(True, axis='x', alpha=0.3)

    # Add value labels (on baseline side to avoid clipping)
    for bar, res in zip(bars, residuals):
        if res < 0:
            # Negative bar extends left, place text on right of baseline
            ax.text(0.002, bar.get_y() + bar.get_height()/2,
                    f'{res:+.4f}', va='center', ha='left', fontsize=9)
        else:
            # Positive bar extends right, place text on left of baseline
            ax.text(-0.002, bar.get_y() + bar.get_height()/2,
                    f'{res:+.4f}', va='center', ha='right', fontsize=9)

    fig.tight_layout()
    chart_path = os.path.join(args.experiment_dir, "ablation_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {chart_path}")


if __name__ == "__main__":
    main()
