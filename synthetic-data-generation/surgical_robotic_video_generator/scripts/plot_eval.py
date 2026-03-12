#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to compare evaluation results from two JSON files with box plots.
"""

import argparse
import json

import matplotlib.pyplot as plt


def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, "r") as f:
        return json.load(f)


def create_comparison_plot(file_a, file_b, legend_a, legend_b, output_path=None):
    """Create comparison box plots from two evaluation JSON files."""

    # Load data
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    # Extract average MSE and std
    avg_mse_a = data_a["average_mse"]
    std_mse_a = data_a["std_mse"]
    avg_mse_b = data_b["average_mse"]
    std_mse_b = data_b["std_mse"]

    # Extract component metrics
    comp_a = data_a["average_component_metrics"]
    comp_b = data_b["average_component_metrics"]

    # Component metric names (in order for plotting)
    component_names = ["mse_xyz_arm1", "mse_xyz_arm2", "mse_rot_arm1", "mse_rot_arm2", "mse_jaw_arm1", "mse_jaw_arm2"]

    # Create figure with 2x5 grid to accommodate:
    # - Average MSE: 2x2 block (columns 0-1, rows 0-1)
    # - Components: 2x3 block (columns 2-4, rows 0-1) = 6 subplots
    # Note: Using 2x5 instead of 2x4 to fit 6 components in 2x3 layout
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)

    # ========== Average MSE plot (spans 2x2 = 4 subplot positions) ==========
    ax_avg = fig.add_subplot(gs[0:2, 0:2])

    # Prepare data for box plot (using individual trajectory MSEs)
    trajectories_a = [t["mse"] for t in data_a["trajectories"]]
    trajectories_b = [t["mse"] for t in data_b["trajectories"]]

    # Create box plot
    colors = ["#1f77b4", "#ff7f0e"]
    bp = ax_avg.boxplot([trajectories_a, trajectories_b], labels=[legend_a, legend_b], patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add error bars for std
    x_positions = [1, 2]
    means = [avg_mse_a, avg_mse_b]
    stds = [std_mse_a, std_mse_b]

    ax_avg.errorbar(
        x_positions, means, yerr=stds, fmt="o", color="black", capsize=5, capthick=2, markersize=8, label="Mean ± Std"
    )

    ax_avg.set_ylabel("MSE", fontsize=12, fontweight="bold")
    ax_avg.set_title("Average MSE Comparison", fontsize=14, fontweight="bold")
    ax_avg.grid(True, alpha=0.3, axis="y")
    ax_avg.legend()

    # ========== Component metrics plots (2x3 = 6 subplots in remaining space) ==========
    component_axes = []
    for idx, comp_name in enumerate(component_names):
        row = idx // 3
        col = 2 + (idx % 3)
        ax = fig.add_subplot(gs[row, col])
        component_axes.append(ax)

        # Get values and stds for this component
        mse_key = comp_name
        std_key = comp_name.replace("mse_", "std_")

        val_a = comp_a[mse_key]
        std_a = comp_a[std_key]
        val_b = comp_b[mse_key]
        std_b = comp_b[std_key]

        # Extract individual trajectory values for box plot
        traj_vals_a = [t[mse_key] for t in data_a["trajectories"]]
        traj_vals_b = [t[mse_key] for t in data_b["trajectories"]]

        # Create box plot
        bp = ax.boxplot([traj_vals_a, traj_vals_b], labels=[legend_a, legend_b], patch_artist=True, widths=0.6)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add error bars
        ax.errorbar(
            [1, 2], [val_a, val_b], yerr=[std_a, std_b], fmt="o", color="black", capsize=5, capthick=2, markersize=6
        )

        # Format title
        title = comp_name.replace("mse_", "").replace("_", " ").title()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if needed
        ax.tick_params(axis="x", rotation=45)

    # Add overall title
    fig.suptitle("Evaluation Metrics Comparison", fontsize=16, fontweight="bold", y=0.98)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results from two JSON files")
    parser.add_argument("file_a", type=str, help="Path to first JSON file")
    parser.add_argument("file_b", type=str, help="Path to second JSON file")
    parser.add_argument("legend_a", type=str, help="Legend name for first file (A)")
    parser.add_argument("legend_b", type=str, help="Legend name for second file (B)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path (optional)")

    args = parser.parse_args()

    create_comparison_plot(args.file_a, args.file_b, args.legend_a, args.legend_b, args.output)


if __name__ == "__main__":
    main()
