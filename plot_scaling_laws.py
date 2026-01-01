#!/usr/bin/env python3
"""
plot_scaling_laws.py
====================

Generate publication-quality scaling law plots for FP8 training research.

Recreates the canonical scaling plots from:
- Kaplan et al. "Scaling Laws for Neural Language Models" (GPT-3)
- Hoffmann et al. "Training Compute-Optimal Large Language Models" (Chinchilla)

Usage:
    python plot_scaling_laws.py
    python plot_scaling_laws.py --experiments-dir experiments --output-dir plots
"""

import argparse
import glob
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# Color palette - modern, accessible colors
COLORS = {
    'bf16': '#2E86AB',      # Blue
    'fp8': '#E94F37',       # Red/Orange
    'fp8_tensorwise': '#E94F37',
    'fp8_rowwise': '#F39237',
    '125M': '#2E86AB',
    '250M': '#28A745',
    '350M': '#F39237', 
    '760M': '#E94F37',
    '1.5B': '#9B59B6',
    '3B': '#1ABC9C',
    '7B': '#E74C3C',
}

MODEL_SIZES = {
    '125M': 125e6,
    '250M': 250e6,
    '350M': 350e6,
    '760M': 760e6,
    '1.5B': 1.5e9,
    '3B': 3e9,
    '7B': 7e9,
}


@dataclass
class ExperimentRun:
    """Represents a single training run."""
    name: str
    mode: str  # 'bf16' or 'fp8'
    model_size: str
    num_params: int
    df: pd.DataFrame
    
    @property
    def final_loss(self) -> float:
        return self.df['train_loss'].iloc[-1]
    
    @property
    def final_val_loss(self) -> float:
        val_losses = self.df['val_loss'].dropna()
        return val_losses.iloc[-1] if len(val_losses) > 0 else self.final_loss
    
    @property
    def total_tokens(self) -> int:
        return self.df['total_tokens'].iloc[-1]
    
    @property
    def tokens_billions(self) -> float:
        return self.total_tokens / 1e9
    
    @property 
    def avg_throughput(self) -> float:
        return self.df['tokens_per_sec'].mean()


def load_experiment(filepath: str) -> Optional[ExperimentRun]:
    """Load a single experiment CSV file."""
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return None
            
        filename = Path(filepath).stem
        
        # Parse mode from filename or data
        if 'mode' in df.columns:
            mode = df['mode'].iloc[0]
        elif 'fp8' in filename.lower():
            mode = 'fp8'
        elif 'bf16' in filename.lower():
            mode = 'bf16'
        else:
            mode = 'unknown'
        
        # Parse model size
        if 'model_size' in df.columns:
            model_size = df['model_size'].iloc[0]
        else:
            # Try to extract from filename
            size_match = re.search(r'(\d+\.?\d*[MB])', filename)
            model_size = size_match.group(1) if size_match else 'unknown'
        
        # Get num params
        if 'num_params' in df.columns:
            num_params = int(df['num_params'].iloc[0])
        else:
            num_params = MODEL_SIZES.get(model_size, 0)
        
        return ExperimentRun(
            name=filename,
            mode=mode,
            model_size=model_size,
            num_params=num_params,
            df=df
        )
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def load_all_experiments(experiments_dir: str) -> List[ExperimentRun]:
    """Load all experiment CSVs from directory."""
    experiments = []
    
    for filepath in glob.glob(f"{experiments_dir}/*.csv"):
        exp = load_experiment(filepath)
        if exp is not None and len(exp.df) > 10:  # Skip very short runs
            experiments.append(exp)
            print(f"Loaded: {exp.name} ({exp.mode}, {exp.model_size}, {len(exp.df)} steps)")
    
    return experiments


def filter_scaling_experiments(experiments: List[ExperimentRun]) -> List[ExperimentRun]:
    """Filter to just the scaling law experiments."""
    scaling_runs = []
    for exp in experiments:
        if 'scaling' in exp.name.lower():
            scaling_runs.append(exp)
    return scaling_runs


# =============================================================================
# Scaling Law Functions
# =============================================================================

def power_law(x, a, b):
    """Power law: L = a * x^b"""
    return a * np.power(x, b)


def chinchilla_law(N, D, A, B, alpha, beta, E):
    """
    Chinchilla scaling law:
    L(N, D) = E + A/N^alpha + B/D^beta
    
    Where:
    - N = number of parameters
    - D = number of tokens
    - E = irreducible loss
    - A, alpha = parameters scaling coefficients
    - B, beta = data scaling coefficients
    """
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit power law to data, return (a, b) coefficients."""
    # Filter out invalid values (zero, negative, inf, nan)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return 1.0, -0.1  # Default fallback
    
    # Use log-space for fitting
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Check for invalid values after log
    if not (np.all(np.isfinite(log_x)) and np.all(np.isfinite(log_y))):
        return 1.0, -0.1  # Default fallback
    
    try:
        # Linear fit in log space
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        return a, b
    except (np.linalg.LinAlgError, ValueError):
        return 1.0, -0.1  # Default fallback


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_loss_vs_tokens(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 1: Training loss vs tokens for each model size.
    Classic scaling law visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by model size and mode
    bf16_runs = [e for e in experiments if 'bf16' in e.mode.lower()]
    fp8_runs = [e for e in experiments if 'fp8' in e.mode.lower()]
    
    # Left plot: BF16 runs
    ax = axes[0]
    for exp in sorted(bf16_runs, key=lambda x: x.num_params):
        color = COLORS.get(exp.model_size, '#666666')
        tokens_b = exp.df['total_tokens'] / 1e9
        ax.plot(tokens_b, exp.df['train_loss'], 
                color=color, label=f"{exp.model_size}", alpha=0.9)
    
    ax.set_xlabel('Tokens (Billions)')
    ax.set_ylabel('Training Loss')
    ax.set_title('BF16 Training')
    ax.legend(title='Model Size', loc='upper right')
    ax.set_xlim(left=0)
    
    # Right plot: FP8 runs
    ax = axes[1]
    for exp in sorted(fp8_runs, key=lambda x: x.num_params):
        color = COLORS.get(exp.model_size, '#666666')
        tokens_b = exp.df['total_tokens'] / 1e9
        ax.plot(tokens_b, exp.df['train_loss'],
                color=color, label=f"{exp.model_size}", alpha=0.9)
    
    ax.set_xlabel('Tokens (Billions)')
    ax.set_ylabel('Training Loss')
    ax.set_title('FP8 Training')
    ax.legend(title='Model Size', loc='upper right')
    ax.set_xlim(left=0)
    
    fig.suptitle('Training Loss vs Token Count by Model Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_vs_tokens_loglog(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 2: Log-log plot of loss vs tokens - reveals power law relationship.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Group by model size
    by_size = {}
    for exp in experiments:
        if exp.model_size not in by_size:
            by_size[exp.model_size] = []
        by_size[exp.model_size].append(exp)
    
    # Plot each model size
    for model_size in sorted(by_size.keys(), key=lambda x: MODEL_SIZES.get(x, 0)):
        exps = by_size[model_size]
        color = COLORS.get(model_size, '#666666')
        
        for exp in exps:
            # Subsample for cleaner plot
            df = exp.df.iloc[::max(1, len(exp.df)//200)]
            tokens = df['total_tokens'].values
            loss = df['train_loss'].values
            
            # Filter out early noisy points
            mask = tokens > tokens.max() * 0.01
            tokens = tokens[mask]
            loss = loss[mask]
            
            linestyle = '-' if 'bf16' in exp.mode else '--'
            label = f"{model_size} ({exp.mode.upper().replace('_TENSORWISE', '')})"
            ax.plot(tokens, loss, color=color, linestyle=linestyle, 
                   label=label, alpha=0.8, linewidth=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Training Loss')
    ax.set_title('Scaling Laws: Loss vs Tokens (Log-Log)', fontweight='bold')
    
    # Format tick labels nicely
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fp8_vs_bf16_comparison(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 3: Direct FP8 vs BF16 comparison for same model sizes.
    Key plot for demonstrating FP8 parity.
    """
    # Find matching pairs
    bf16_runs = {e.model_size: e for e in experiments if 'bf16' in e.mode.lower()}
    fp8_runs = {e.model_size: e for e in experiments if 'fp8' in e.mode.lower()}
    
    common_sizes = set(bf16_runs.keys()) & set(fp8_runs.keys())
    
    if len(common_sizes) == 0:
        print("No matching BF16/FP8 pairs found for comparison")
        return
    
    n_plots = len(common_sizes)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for ax, model_size in zip(axes, sorted(common_sizes, key=lambda x: MODEL_SIZES.get(x, 0))):
        bf16 = bf16_runs[model_size]
        fp8 = fp8_runs[model_size]
        
        # Align by tokens
        bf16_tokens = bf16.df['total_tokens'] / 1e9
        fp8_tokens = fp8.df['total_tokens'] / 1e9
        
        ax.plot(bf16_tokens, bf16.df['train_loss'], 
                color=COLORS['bf16'], label='BF16', linewidth=2.5)
        ax.plot(fp8_tokens, fp8.df['train_loss'],
                color=COLORS['fp8'], label='FP8', linewidth=2.5, linestyle='--')
        
        ax.set_xlabel('Tokens (Billions)')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'{model_size} Model', fontweight='bold')
        ax.legend()
        
        # Add final loss annotation
        bf16_final = bf16.final_loss
        fp8_final = fp8.final_loss
        diff_pct = (fp8_final - bf16_final) / bf16_final * 100
        
        ax.annotate(f'Δ = {diff_pct:+.2f}%', 
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('FP8 vs BF16 Training Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_throughput_comparison(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 4: Throughput comparison FP8 vs BF16.
    """
    # Find matching pairs
    bf16_runs = {e.model_size: e for e in experiments if 'bf16' in e.mode.lower()}
    fp8_runs = {e.model_size: e for e in experiments if 'fp8' in e.mode.lower()}
    
    common_sizes = sorted(set(bf16_runs.keys()) & set(fp8_runs.keys()),
                         key=lambda x: MODEL_SIZES.get(x, 0))
    
    if len(common_sizes) == 0:
        print("No matching BF16/FP8 pairs found for throughput comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(common_sizes))
    width = 0.35
    
    bf16_throughputs = [bf16_runs[s].avg_throughput / 1e6 for s in common_sizes]  # M tok/s
    fp8_throughputs = [fp8_runs[s].avg_throughput / 1e6 for s in common_sizes]
    
    bars1 = ax.bar(x - width/2, bf16_throughputs, width, label='BF16', color=COLORS['bf16'])
    bars2 = ax.bar(x + width/2, fp8_throughputs, width, label='FP8', color=COLORS['fp8'])
    
    # Add speedup labels
    for i, (bf16_t, fp8_t) in enumerate(zip(bf16_throughputs, fp8_throughputs)):
        speedup = fp8_t / bf16_t
        ax.annotate(f'{speedup:.2f}x',
                   xy=(i + width/2, fp8_t),
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Throughput (M tokens/sec)')
    ax.set_title('Training Throughput: FP8 vs BF16', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_sizes)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_loss_vs_params(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 5: Final loss vs model parameters - classic scaling plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bf16_data = [(e.num_params, e.final_loss) for e in experiments if 'bf16' in e.mode.lower()]
    fp8_data = [(e.num_params, e.final_loss) for e in experiments if 'fp8' in e.mode.lower()]
    
    if bf16_data:
        params, losses = zip(*sorted(bf16_data))
        ax.scatter(params, losses, s=150, c=COLORS['bf16'], label='BF16', 
                  marker='o', edgecolors='black', linewidths=1, zorder=5)
        ax.plot(params, losses, c=COLORS['bf16'], alpha=0.5, linestyle='-', linewidth=2)
        
        # Fit power law
        if len(params) >= 2:
            a, b = fit_power_law(np.array(params), np.array(losses))
            x_fit = np.logspace(np.log10(min(params)*0.8), np.log10(max(params)*1.2), 100)
            ax.plot(x_fit, power_law(x_fit, a, b), 
                   c=COLORS['bf16'], alpha=0.3, linestyle='--', linewidth=2,
                   label=f'BF16 fit: $L \propto N^{{{b:.3f}}}$')
    
    if fp8_data:
        params, losses = zip(*sorted(fp8_data))
        ax.scatter(params, losses, s=150, c=COLORS['fp8'], label='FP8',
                  marker='s', edgecolors='black', linewidths=1, zorder=5)
        ax.plot(params, losses, c=COLORS['fp8'], alpha=0.5, linestyle='-', linewidth=2)
        
        if len(params) >= 2:
            a, b = fit_power_law(np.array(params), np.array(losses))
            x_fit = np.logspace(np.log10(min(params)*0.8), np.log10(max(params)*1.2), 100)
            ax.plot(x_fit, power_law(x_fit, a, b),
                   c=COLORS['fp8'], alpha=0.3, linestyle='--', linewidth=2,
                   label=f'FP8 fit: $L \propto N^{{{b:.3f}}}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (N)')
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Scaling Law: Loss vs Model Size', fontweight='bold')
    
    # Nice tick formatting
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_validation_loss_trajectory(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 6: Validation loss over training for different model sizes.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for exp in sorted(experiments, key=lambda x: x.num_params):
        # Get validation loss points
        val_df = exp.df[exp.df['val_loss'].notna()]
        if len(val_df) < 2:
            continue
            
        tokens_b = val_df['total_tokens'] / 1e9
        val_loss = val_df['val_loss']
        
        color = COLORS.get(exp.model_size, '#666666')
        linestyle = '-' if 'bf16' in exp.mode else '--'
        label = f"{exp.model_size} ({exp.mode.split('_')[0].upper()})"
        
        ax.plot(tokens_b, val_loss, color=color, linestyle=linestyle,
               label=label, marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Tokens (Billions)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss During Training', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def compute_flops(num_params: int, num_tokens: int) -> float:
    """
    Estimate training FLOPs using the standard approximation:
    FLOPs ≈ 6 * N * D
    
    Where N = parameters, D = tokens
    Factor of 6 comes from: 2 (multiply-add) * 3 (forward + backward + gradient)
    
    Returns FLOPs in PetaFLOP/s-days for comparability with papers.
    """
    # Use float to avoid integer overflow
    flops = 6.0 * float(num_params) * float(num_tokens)
    # Convert to PetaFLOP/s-days: 1 PF-day = 8.64e19 FLOPs
    pf_days = flops / 8.64e19
    return pf_days


def plot_loss_vs_compute_canonical(experiments: List[ExperimentRun], output_path: str):
    """
    Plot: Loss vs Compute (PetaFLOP/s-days) - THE canonical scaling plot.
    
    Shows all training runs with curves colored by model size,
    revealing the compute-optimal frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colormap for model sizes
    all_params = sorted(set(e.num_params for e in experiments))
    param_to_color = {}
    cmap = plt.cm.viridis
    for i, p in enumerate(all_params):
        param_to_color[p] = cmap(i / max(1, len(all_params) - 1))
    
    # Track frontier points (minimum loss at each compute level)
    frontier_points = []
    
    for exp in sorted(experiments, key=lambda x: x.num_params):
        # Compute FLOPs for each step
        df = exp.df.copy()
        df['compute_pf_days'] = df['total_tokens'].apply(
            lambda t: compute_flops(exp.num_params, t)
        )
        
        # Subsample for cleaner plot
        df = df.iloc[::max(1, len(df)//200)]
        
        color = param_to_color[exp.num_params]
        alpha = 0.7 if 'bf16' in exp.mode else 0.5
        linestyle = '-' if 'bf16' in exp.mode else '--'
        
        ax.plot(df['compute_pf_days'], df['train_loss'],
               color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5)
        
        # Track frontier (final point of each run)
        frontier_points.append((
            compute_flops(exp.num_params, exp.total_tokens),
            exp.final_loss,
            exp.num_params
        ))
    
    # Fit power law to frontier
    frontier_points = sorted(frontier_points, key=lambda x: x[0])
    compute_vals = np.array([p[0] for p in frontier_points])
    loss_vals = np.array([p[1] for p in frontier_points])
    
    if len(compute_vals) >= 2:
        a, b = fit_power_law(compute_vals, loss_vals)
        x_fit = np.logspace(np.log10(compute_vals.min()*0.5), 
                           np.log10(compute_vals.max()*2), 100)
        ax.plot(x_fit, power_law(x_fit, a, b),
               'orange', linestyle='--', linewidth=2.5, alpha=0.8,
               label=f'L = {a:.2f} · C^{{{b:.3f}}}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (PetaFLOP/s-days)', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Scaling Law: Loss vs Compute', fontweight='bold', fontsize=14)
    
    # Add colorbar for model size
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=min(all_params)/1e6, 
                                                  vmax=max(all_params)/1e6))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Parameters (M)')
    
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_kaplan_triple(experiments: List[ExperimentRun], output_path: str):
    """
    The classic 3-panel figure from Kaplan et al.:
    1. Loss vs Compute (all runs, colored by size)
    2. Loss vs Dataset Size (converged runs)
    3. Loss vs Parameters (converged runs)
    
    With fitted power law equations on each.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    all_params = sorted(set(e.num_params for e in experiments))
    cmap = plt.cm.viridis
    param_to_color = {p: cmap(i / max(1, len(all_params) - 1)) 
                      for i, p in enumerate(all_params)}
    
    # ===================
    # Panel 1: Loss vs Compute
    # ===================
    ax = axes[0]
    
    frontier_points = []
    
    for exp in sorted(experiments, key=lambda x: x.num_params):
        df = exp.df.copy()
        df['compute_pf_days'] = df['total_tokens'].apply(
            lambda t: compute_flops(exp.num_params, t)
        )
        df = df.iloc[::max(1, len(df)//150)]  # Subsample
        
        color = param_to_color[exp.num_params]
        ax.plot(df['compute_pf_days'], df['train_loss'],
               color=color, alpha=0.6, linewidth=1.2)
        
        frontier_points.append((
            compute_flops(exp.num_params, exp.total_tokens),
            exp.final_loss
        ))
    
    # Fit frontier
    compute_vals = np.array([p[0] for p in frontier_points])
    loss_vals = np.array([p[1] for p in frontier_points])
    
    if len(compute_vals) >= 2:
        a, b = fit_power_law(compute_vals, loss_vals)
        x_fit = np.logspace(np.log10(compute_vals.min()*0.3), 
                           np.log10(compute_vals.max()*3), 100)
        ax.plot(x_fit, power_law(x_fit, a, b),
               color='orange', linestyle='--', linewidth=2.5)
        
        # Add equation in box
        ax.text(0.05, 0.12, f'$L = {a:.2f} \\cdot C^{{{b:.3f}}}$',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute\n(PF-days)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Compute', fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=min(all_params)/1e6,
                                                  vmax=max(all_params)/1e6))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Params (M)', fontsize=10)
    
    # ===================
    # Panel 2: Loss vs Dataset Size
    # ===================
    ax = axes[1]
    
    # Use final loss vs tokens for each run
    data_points = [(e.total_tokens, e.final_loss) for e in experiments]
    data_points = sorted(data_points, key=lambda x: x[0])
    
    tokens = np.array([p[0] for p in data_points])
    losses = np.array([p[1] for p in data_points])
    
    ax.scatter(tokens, losses, s=80, c='#2E86AB', edgecolors='black', 
              linewidths=0.5, zorder=5, alpha=0.8)
    
    # Fit power law
    if len(tokens) >= 2:
        a, b = fit_power_law(tokens, losses)
        x_fit = np.logspace(np.log10(tokens.min()*0.5),
                           np.log10(tokens.max()*2), 100)
        ax.plot(x_fit, power_law(x_fit, a, b),
               color='#2E86AB', linewidth=2)
        
        # Format equation nicely
        exp_d = int(np.log10(1/a**(1/b)))
        ax.text(0.95, 0.95, f'$L = (D / {1/a**(1/b):.1e})^{{{b:.3f}}}$',
               transform=ax.transAxes, fontsize=11, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size\n(tokens)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Data', fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.0f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    
    # ===================
    # Panel 3: Loss vs Parameters
    # ===================
    ax = axes[2]
    
    # Use final loss for each model size (average if multiple runs)
    param_loss = {}
    for exp in experiments:
        if exp.num_params not in param_loss:
            param_loss[exp.num_params] = []
        param_loss[exp.num_params].append(exp.final_loss)
    
    params = np.array(sorted(param_loss.keys()))
    losses = np.array([np.mean(param_loss[p]) for p in params])
    
    ax.scatter(params, losses, s=80, c='#2E86AB', edgecolors='black',
              linewidths=0.5, zorder=5, alpha=0.8)
    
    # Fit power law
    if len(params) >= 2:
        a, b = fit_power_law(params, losses)
        x_fit = np.logspace(np.log10(params.min()*0.5),
                           np.log10(params.max()*2), 100)
        ax.plot(x_fit, power_law(x_fit, a, b),
               color='#2E86AB', linewidth=2)
        
        ax.text(0.95, 0.95, f'$L = (N / {1/a**(1/b):.1e})^{{{b:.3f}}}$',
               transform=ax.transAxes, fontsize=11, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters\n(non-embedding)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Parameters', fontweight='bold')
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_chinchilla_style(experiments: List[ExperimentRun], output_path: str):
    """
    Chinchilla-style plot: Loss vs Compute with iso-loss contours
    and optimal frontier highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colormap based on parameters
    all_params = sorted(set(e.num_params for e in experiments))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.log10(min(all_params)), 
                        vmax=np.log10(max(all_params)))
    
    # Plot each training run
    for exp in sorted(experiments, key=lambda x: x.num_params):
        df = exp.df.copy()
        df['compute_pf_days'] = df['total_tokens'].apply(
            lambda t: compute_flops(exp.num_params, t)
        )
        
        # Subsample
        df = df.iloc[::max(1, len(df)//200)]
        
        color = cmap(norm(np.log10(exp.num_params)))
        linewidth = 1.5 if 'bf16' in exp.mode else 1.0
        alpha = 0.8 if 'bf16' in exp.mode else 0.5
        
        ax.plot(df['compute_pf_days'], df['train_loss'],
               color=color, linewidth=linewidth, alpha=alpha)
    
    # Compute and plot the optimal frontier
    # For each compute budget, find minimum loss achieved
    all_points = []
    for exp in experiments:
        df = exp.df.copy()
        for _, row in df.iterrows():
            compute = compute_flops(exp.num_params, row['total_tokens'])
            all_points.append((compute, row['train_loss'], exp.num_params))
    
    # Bin by compute and find minimum loss
    all_points = sorted(all_points, key=lambda x: x[0])
    compute_bins = np.logspace(np.log10(min(p[0] for p in all_points)),
                               np.log10(max(p[0] for p in all_points)), 30)
    
    frontier_compute = []
    frontier_loss = []
    
    for i in range(len(compute_bins)-1):
        points_in_bin = [p for p in all_points 
                        if compute_bins[i] <= p[0] < compute_bins[i+1]]
        if points_in_bin:
            min_point = min(points_in_bin, key=lambda x: x[1])
            frontier_compute.append(min_point[0])
            frontier_loss.append(min_point[1])
    
    if frontier_compute:
        ax.plot(frontier_compute, frontier_loss, 
               'k--', linewidth=2.5, alpha=0.8, label='Optimal frontier')
    
    # Fit power law to frontier
    if len(frontier_compute) >= 2:
        a, b = fit_power_law(np.array(frontier_compute), np.array(frontier_loss))
        x_fit = np.logspace(np.log10(min(frontier_compute)*0.5),
                           np.log10(max(frontier_compute)*2), 100)
        ax.plot(x_fit, power_law(x_fit, a, b),
               color='orange', linestyle='--', linewidth=2,
               label=f'$L = {a:.2f} \\cdot C^{{{b:.3f}}}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (PetaFLOP/s-days)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Compute-Optimal Scaling (Chinchilla-style)', fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('log₁₀(Parameters)', fontsize=10)
    # Add nice tick labels
    cbar.set_ticks([np.log10(p) for p in all_params])
    cbar.set_ticklabels([f'{p/1e6:.0f}M' if p < 1e9 else f'{p/1e9:.1f}B' 
                        for p in all_params])
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fp8_bf16_scaling_comparison(experiments: List[ExperimentRun], output_path: str):
    """
    Key plot for the FP8 paper: Show that FP8 follows the SAME scaling law as BF16.
    
    Two panels:
    1. Loss vs Compute for both precisions (overlay)
    2. Loss vs Parameters with fitted curves showing identical exponents
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    bf16_exps = [e for e in experiments if 'bf16' in e.mode.lower()]
    fp8_exps = [e for e in experiments if 'fp8' in e.mode.lower()]
    
    # ===================
    # Panel 1: Loss vs Compute (overlay)
    # ===================
    ax = axes[0]
    
    bf16_frontier = []
    fp8_frontier = []
    
    for exp in bf16_exps:
        df = exp.df.iloc[::max(1, len(exp.df)//150)]
        compute = df['total_tokens'].apply(lambda t: compute_flops(exp.num_params, t))
        ax.plot(compute, df['train_loss'], color=COLORS['bf16'], alpha=0.6, linewidth=1.2)
        bf16_frontier.append((compute_flops(exp.num_params, exp.total_tokens), exp.final_loss))
    
    for exp in fp8_exps:
        df = exp.df.iloc[::max(1, len(exp.df)//150)]
        compute = df['total_tokens'].apply(lambda t: compute_flops(exp.num_params, t))
        ax.plot(compute, df['train_loss'], color=COLORS['fp8'], alpha=0.6, 
               linewidth=1.2, linestyle='--')
        fp8_frontier.append((compute_flops(exp.num_params, exp.total_tokens), exp.final_loss))
    
    # Fit and plot frontiers
    for frontier, color, label, ls in [(bf16_frontier, COLORS['bf16'], 'BF16', '-'),
                                        (fp8_frontier, COLORS['fp8'], 'FP8', '--')]:
        if len(frontier) >= 2:
            compute_vals = np.array([p[0] for p in frontier])
            loss_vals = np.array([p[1] for p in frontier])
            a, b = fit_power_law(compute_vals, loss_vals)
            x_fit = np.logspace(np.log10(compute_vals.min()*0.3),
                               np.log10(compute_vals.max()*3), 100)
            ax.plot(x_fit, power_law(x_fit, a, b), color=color, linestyle=ls,
                   linewidth=3, label=f'{label}: $L \propto C^{{{b:.3f}}}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (PF-days)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Scaling Laws: FP8 vs BF16', fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    
    # ===================
    # Panel 2: Loss vs Params
    # ===================
    ax = axes[1]
    
    # BF16
    bf16_param_loss = {}
    for exp in bf16_exps:
        if exp.num_params not in bf16_param_loss:
            bf16_param_loss[exp.num_params] = []
        bf16_param_loss[exp.num_params].append(exp.final_loss)
    
    if bf16_param_loss:
        params = np.array(sorted(bf16_param_loss.keys()))
        losses = np.array([np.mean(bf16_param_loss[p]) for p in params])
        ax.scatter(params, losses, s=120, c=COLORS['bf16'], marker='o',
                  edgecolors='black', linewidths=1, label='BF16', zorder=5)
        
        if len(params) >= 2:
            a, b = fit_power_law(params, losses)
            x_fit = np.logspace(np.log10(params.min()*0.5), np.log10(params.max()*2), 100)
            ax.plot(x_fit, power_law(x_fit, a, b), color=COLORS['bf16'],
                   linewidth=2, label=f'BF16: $L \\propto N^{{{b:.3f}}}$')
    
    # FP8
    fp8_param_loss = {}
    for exp in fp8_exps:
        if exp.num_params not in fp8_param_loss:
            fp8_param_loss[exp.num_params] = []
        fp8_param_loss[exp.num_params].append(exp.final_loss)
    
    if fp8_param_loss:
        params = np.array(sorted(fp8_param_loss.keys()))
        losses = np.array([np.mean(fp8_param_loss[p]) for p in params])
        ax.scatter(params, losses, s=120, c=COLORS['fp8'], marker='s',
                  edgecolors='black', linewidths=1, label='FP8', zorder=5)
        
        if len(params) >= 2:
            a, b = fit_power_law(params, losses)
            x_fit = np.logspace(np.log10(params.min()*0.5), np.log10(params.max()*2), 100)
            ax.plot(x_fit, power_law(x_fit, a, b), color=COLORS['fp8'],
                   linewidth=2, linestyle='--', label=f'FP8: $L \\propto N^{{{b:.3f}}}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Parameter Scaling: FP8 vs BF16', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    ax.legend(loc='upper right', fontsize=10)
    
    fig.suptitle('FP8 Achieves Identical Scaling Laws to BF16', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_scaling_summary(experiments: List[ExperimentRun], output_path: str):
    """
    Plot 7: Combined 2x2 summary figure for paper/blog.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Top Left: Loss vs Tokens ---
    ax = axes[0, 0]
    for exp in sorted(experiments, key=lambda x: x.num_params):
        color = COLORS.get(exp.model_size, '#666666')
        linestyle = '-' if 'bf16' in exp.mode else '--'
        tokens_b = exp.df['total_tokens'] / 1e9
        
        # Subsample
        df = exp.df.iloc[::max(1, len(exp.df)//100)]
        tokens_b = df['total_tokens'] / 1e9
        
        label = f"{exp.model_size} ({exp.mode.split('_')[0].upper()})"
        ax.plot(tokens_b, df['train_loss'], color=color, linestyle=linestyle,
               label=label, alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Tokens (Billions)')
    ax.set_ylabel('Training Loss')
    ax.set_title('A. Loss vs Tokens', fontweight='bold', loc='left')
    ax.legend(fontsize=8, ncol=2)
    
    # --- Top Right: Final Loss vs Params (log-log) ---
    ax = axes[0, 1]
    
    bf16_data = [(e.num_params, e.final_loss, e.model_size) for e in experiments if 'bf16' in e.mode.lower()]
    fp8_data = [(e.num_params, e.final_loss, e.model_size) for e in experiments if 'fp8' in e.mode.lower()]
    
    if bf16_data:
        params, losses, sizes = zip(*sorted(bf16_data))
        ax.scatter(params, losses, s=120, c=COLORS['bf16'], label='BF16',
                  marker='o', edgecolors='black', linewidths=1, zorder=5)
        for p, l, s in zip(params, losses, sizes):
            ax.annotate(s, (p, l), textcoords='offset points', xytext=(5, 5), fontsize=8)
    
    if fp8_data:
        params, losses, sizes = zip(*sorted(fp8_data))
        ax.scatter(params, losses, s=120, c=COLORS['fp8'], label='FP8',
                  marker='s', edgecolors='black', linewidths=1, zorder=5)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (N)')
    ax.set_ylabel('Final Loss')
    ax.set_title('B. Scaling Law: Loss vs Parameters', fontweight='bold', loc='left')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    ax.legend()
    
    # --- Bottom Left: FP8 vs BF16 Direct Comparison ---
    ax = axes[1, 0]
    
    bf16_runs = {e.model_size: e for e in experiments if 'bf16' in e.mode.lower()}
    fp8_runs = {e.model_size: e for e in experiments if 'fp8' in e.mode.lower()}
    common_sizes = sorted(set(bf16_runs.keys()) & set(fp8_runs.keys()),
                         key=lambda x: MODEL_SIZES.get(x, 0))
    
    if common_sizes:
        # Plot loss difference over training for first common size
        size = common_sizes[0]
        bf16 = bf16_runs[size]
        fp8 = fp8_runs[size]
        
        # Interpolate to common token points
        min_tokens = max(bf16.df['total_tokens'].min(), fp8.df['total_tokens'].min())
        max_tokens = min(bf16.df['total_tokens'].max(), fp8.df['total_tokens'].max())
        
        bf16_interp = np.interp(
            np.linspace(min_tokens, max_tokens, 100),
            bf16.df['total_tokens'],
            bf16.df['train_loss']
        )
        fp8_interp = np.interp(
            np.linspace(min_tokens, max_tokens, 100),
            fp8.df['total_tokens'],
            fp8.df['train_loss']
        )
        
        tokens_b = np.linspace(min_tokens, max_tokens, 100) / 1e9
        ax.plot(tokens_b, bf16_interp, color=COLORS['bf16'], label=f'BF16', linewidth=2)
        ax.plot(tokens_b, fp8_interp, color=COLORS['fp8'], label=f'FP8', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Tokens (Billions)')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'C. FP8 vs BF16 Comparison ({size})', fontweight='bold', loc='left')
        ax.legend()
        
        # Add loss difference annotation
        final_diff = (fp8_interp[-1] - bf16_interp[-1]) / bf16_interp[-1] * 100
        ax.annotate(f'Final Δ: {final_diff:+.2f}%',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- Bottom Right: Throughput ---
    ax = axes[1, 1]
    
    if common_sizes:
        x = np.arange(len(common_sizes))
        width = 0.35
        
        bf16_throughputs = [bf16_runs[s].avg_throughput / 1e6 for s in common_sizes]
        fp8_throughputs = [fp8_runs[s].avg_throughput / 1e6 for s in common_sizes]
        
        bars1 = ax.bar(x - width/2, bf16_throughputs, width, label='BF16', color=COLORS['bf16'])
        bars2 = ax.bar(x + width/2, fp8_throughputs, width, label='FP8', color=COLORS['fp8'])
        
        for i, (bf16_t, fp8_t) in enumerate(zip(bf16_throughputs, fp8_throughputs)):
            if bf16_t > 0:
                speedup = fp8_t / bf16_t
                ax.annotate(f'{speedup:.1f}x',
                           xy=(i + width/2, fp8_t),
                           ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Throughput (M tokens/sec)')
        ax.set_title('D. Training Throughput', fontweight='bold', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels(common_sizes)
        ax.legend()
    
    fig.suptitle('FP8 Training Scaling Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(experiments: List[ExperimentRun]) -> pd.DataFrame:
    """Generate summary table of all experiments."""
    rows = []
    for exp in sorted(experiments, key=lambda x: (x.mode, x.num_params)):
        rows.append({
            'Model': exp.model_size,
            'Mode': exp.mode.upper().replace('_TENSORWISE', ''),
            'Params': f"{exp.num_params/1e6:.0f}M",
            'Tokens (B)': f"{exp.tokens_billions:.2f}",
            'Final Loss': f"{exp.final_loss:.4f}",
            'Val Loss': f"{exp.final_val_loss:.4f}",
            'Throughput (k tok/s)': f"{exp.avg_throughput/1e3:.0f}",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate scaling law plots')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory containing experiment CSVs')
    parser.add_argument('--output-dir', type=str, default='plots/scaling',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load experiments
    print(f"\nLoading experiments from {args.experiments_dir}...")
    all_experiments = load_all_experiments(args.experiments_dir)
    
    if len(all_experiments) == 0:
        print("No experiments found!")
        return
    
    print(f"\nLoaded {len(all_experiments)} experiments")
    
    # Filter to scaling experiments
    scaling_experiments = filter_scaling_experiments(all_experiments)
    print(f"Found {len(scaling_experiments)} scaling experiments")
    
    # Use scaling experiments if available, otherwise all
    experiments = scaling_experiments if scaling_experiments else all_experiments
    
    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    summary_df = generate_summary_table(experiments)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Generate all plots
    print("Generating plots...")
    
    # Basic plots
    plot_loss_vs_tokens(experiments, f"{args.output_dir}/01_loss_vs_tokens.png")
    plot_loss_vs_tokens_loglog(experiments, f"{args.output_dir}/02_loss_vs_tokens_loglog.png")
    plot_fp8_vs_bf16_comparison(experiments, f"{args.output_dir}/03_fp8_vs_bf16_comparison.png")
    plot_throughput_comparison(experiments, f"{args.output_dir}/04_throughput_comparison.png")
    plot_final_loss_vs_params(experiments, f"{args.output_dir}/05_loss_vs_params.png")
    plot_validation_loss_trajectory(experiments, f"{args.output_dir}/06_validation_loss.png")
    
    # Canonical scaling law plots (Kaplan/Chinchilla style)
    plot_loss_vs_compute_canonical(experiments, f"{args.output_dir}/07_loss_vs_compute.png")
    plot_kaplan_triple(experiments, f"{args.output_dir}/08_kaplan_triple.png")
    plot_chinchilla_style(experiments, f"{args.output_dir}/09_chinchilla_style.png")
    plot_fp8_bf16_scaling_comparison(experiments, f"{args.output_dir}/10_fp8_bf16_scaling_laws.png")
    
    # Summary figure
    plot_combined_scaling_summary(experiments, f"{args.output_dir}/11_scaling_summary.png")
    
    # Save summary table
    summary_df.to_csv(f"{args.output_dir}/experiment_summary.csv", index=False)
    print(f"\nSaved summary to {args.output_dir}/experiment_summary.csv")
    
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()