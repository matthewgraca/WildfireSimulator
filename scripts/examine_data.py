"""
examine_data.py — Visualize what the model sees at each discrete time step.

Shows the fire mask (channel 0) and arrival time (channel 1) after applying
the burn process at each t, using the normalized (transformed) data that
the model actually receives during training.

Usage:
    python scripts/examine_data.py
    python scripts/examine_data.py --sample 3 --num_steps 12
    python scripts/examine_data.py --output ./tmp/data_viz.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.dataloader import WildfireDataLoader, TrialFileLoader, TrialCollection
from wildfire_simulator.datasets import WildfireDataset
from torch.utils.data import random_split


def main():
    parser = argparse.ArgumentParser(description="Visualize burn process at each time step")
    parser.add_argument("--sample", type=int, default=0, help="Sample index from test set")
    parser.add_argument("--num_steps", type=int, default=8, help="Number of time steps to show")
    parser.add_argument("--dt", type=float, default=1/48, help="Time step size (default: 1/48)")
    parser.add_argument("--max_t", type=float, default=1.0, help="Maximum time")
    parser.add_argument("--output", type=str, default="./tmp/examine_data.png", help="Output path")
    args = parser.parse_args()

    # Load test set (same split as training)
    dataloader = WildfireDataLoader(TrialCollection(TrialFileLoader()))
    dataset = WildfireDataset(dataloader)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    burner = ForwardBurnProcess()

    # Get sample and transform
    sample = test_set[args.sample]
    sample_transformed = transform(sample)

    print(f"Sample {args.sample} from test set")
    print(f"  Raw arrival time range: [{sample[1].min():.2f}, {sample[1].max():.2f}]")
    print(f"  Normalized arrival time range: [{sample_transformed[1].min():.4f}, {sample_transformed[1].max():.4f}]")
    print(f"  dt = {args.dt:.6f}, max_t = {args.max_t}")
    print()

    # Select time steps to visualize
    all_times = np.arange(args.dt, args.max_t, args.dt)
    if len(all_times) > args.num_steps:
        indices = np.linspace(0, len(all_times) - 1, args.num_steps, dtype=int)
        times = all_times[indices]
    else:
        times = all_times

    # Create figure: 3 rows (input arrival, target arrival, target mask) x num_steps columns
    fig, axes = plt.subplots(3, len(times), figsize=(len(times) * 3, 9), squeeze=False)

    for col, t in enumerate(times):
        # What the model sees as input at this time step
        burned_input = burner(sample_transformed, t)
        # What the model should predict (target at t + dt)
        burned_target = burner(sample_transformed, t + args.dt)

        arrival_input = burned_input[1].numpy()
        arrival_target = burned_target[1].numpy()
        mask_target = burned_target[0].numpy()

        # Count burned pixels
        n_burned_input = (burned_input[0].numpy() > 0).sum()
        n_burned_target = (mask_target > 0).sum()

        # Row 0: Input arrival time
        arrival_input_display = np.where(arrival_input == 0, np.nan, arrival_input)
        axes[0, col].imshow(arrival_input_display, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        axes[0, col].set_title(f"t={t:.3f}\n({n_burned_input} px)", fontsize=8)
        axes[0, col].axis('off')

        # Row 1: Target arrival time
        arrival_target_display = np.where(arrival_target == 0, np.nan, arrival_target)
        axes[1, col].imshow(arrival_target_display, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        axes[1, col].axis('off')

        # Row 2: Target fire mask
        target_mask_display = np.where(mask_target == 0, np.nan, mask_target)
        axes[2, col].imshow(target_mask_display, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        axes[2, col].set_title(f"target\n({n_burned_target} px)", fontsize=7)
        axes[2, col].axis('off')

        print(f"  t={t:.4f}: input={n_burned_input:6d} px burned, target={n_burned_target:6d} px burned (Δ={n_burned_target - n_burned_input})")

    fig.suptitle(f"Sample {args.sample} — Model Input/Target at Each Time Step\n(normalized space, dt={args.dt:.4f})", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(left=0.07)

    fig.text(0.025, 0.78, "Input\nArrival Time", va='center', ha='center', fontsize=11, rotation=90)
    fig.text(0.025, 0.48, "Target\nArrival Time", va='center', ha='center', fontsize=11, rotation=90)
    fig.text(0.025, 0.18, "Target\nFire Mask", va='center', ha='center', fontsize=11, rotation=90)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {output_path}")

    # --- Static channels visualization ---
    static_output = output_path.parent / (output_path.stem + "_channels" + output_path.suffix)
    plot_static_channels(sample, static_output)
    print(f"  Saved: {static_output}")


def plot_static_channels(sample, output_path):
    """Visualize static landscape channels (2-12) with a wind quiver plot."""
    # Channel mapping:
    # 2: Elevation, 3: Slope, 4: Aspect, 5: Fuel model (FBFM40)
    # 6: Canopy cover, 7: Stand height, 8: Canopy base height, 9: Canopy bulk density
    # 10: Wind speed (scalar), 11: Wind direction (scalar), 12: Foliar moisture (scalar)

    channel_info = [
        (2, "Elevation", "terrain", "m"),
        (3, "Slope", "YlOrBr", "°"),
        (4, "Aspect", "hsv", "°"),
        (5, "Fuel Model\n(FBFM40)", "Set3", ""),
        (6, "Canopy Cover", "Greens", "%"),
        (7, "Stand Height", "Greens", "m"),
        (8, "Canopy Base\nHeight", "Greens", "m"),
        (9, "Canopy Bulk\nDensity", "Greens", "kg/m³"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Plot spatial channels (2-9) in first 8 panels
    for i, (ch_idx, name, cmap, unit) in enumerate(channel_info):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        data = sample[ch_idx].numpy()

        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_title(f"{name}", fontsize=10)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cbar.set_label(unit, fontsize=8)

    # Wind quiver plot (panel at row 1, col 3)
    ax_wind = axes[1, 3]
    wind_u = sample[10, 0, 0].item()  # U component (east-west)
    wind_v = sample[11, 0, 0].item()  # V component (north-south)
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    wind_dir = (np.degrees(np.arctan2(-wind_u, -wind_v)) + 360) % 360

    # Create a grid of arrows showing uniform wind field
    grid_size = 10
    x = np.linspace(0, 500, grid_size)
    y = np.linspace(0, 500, grid_size)
    X, Y = np.meshgrid(x, y)
    # U/V are in cartesian (U=east, V=north); flip V for image coords
    U = np.full_like(X, wind_u)
    V = np.full_like(Y, -wind_v)

    ax_wind.quiver(X, Y, U, V, scale=wind_speed * 15, color='navy', alpha=0.7)
    ax_wind.set_xlim(0, 500)
    ax_wind.set_ylim(500, 0)  # flip y to match image coordinates
    ax_wind.set_aspect('equal')
    ax_wind.set_title(f"Wind\n{wind_speed:.0f} units @ {wind_dir:.0f}°", fontsize=10)
    ax_wind.axis('off')

    # Foliar moisture (panel at row 1, col 4)
    ax_fm = axes[1, 4]
    foliar_moisture = sample[12, 0, 0].item()
    ax_fm.text(0.5, 0.5, f"Foliar\nMoisture\n\n{foliar_moisture:.0f}%",
               ha='center', va='center', fontsize=14,
               transform=ax_fm.transAxes)
    ax_fm.axis('off')

    fig.suptitle("Static Input Channels (Landscape Features)", fontsize=13)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
