#!/usr/bin/env python3
"""
FP8 Training Visualization
==========================

Beautiful plots for comparing FP8 vs BF16 training runs.
Uses seaborn for publication-quality figures.

Usage:
    python plot_training.py train_bf16_1_5B.csv train_fp8_1_5B.csv
    python plot_training.py train_bf16_1_5B.csv train_fp8_1_5B.csv --output plots/
    python plot_training.py *.csv --combined  # Plot all files together
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette - distinct colors for FP8 vs BF16
COLORS = {
    'bf16': '#2E86AB',      # Blue
    'fp8_tensorwise': '#E94F37',  # Red-orange
    'fp8_rowwise': '#F39C12',     # Orange
    'fp8': '#E94F37',       # Alias
}


def load_and_prepare_data(filepaths):
    """Load CSV files and prepare for plotting."""
    dfs = []
    for fp in filepaths:
        df = pd.read_csv(fp)
        df['source_file'] = Path(fp).stem
        
        # Handle column name variations between old and new formats
        # Old format: 'loss', 'elapsed_time', 'step_time_ms'
        # New format: 'train_loss', 'train_time_total', 'train_time_ms'
        
        # Standardize loss column
        if 'train_loss' in df.columns:
            df['loss'] = pd.to_numeric(df['train_loss'], errors='coerce')
        elif 'loss' not in df.columns:
            print(f"Warning: No loss column found in {fp}")
            continue
        else:
            df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
        
        # Clean up mode names for display
        df['mode_display'] = df['mode'].replace({
            'bf16': 'BF16',
            'fp8_tensorwise': 'FP8 (tensorwise)',
            'fp8_rowwise': 'FP8 (rowwise)',
        })
        
        # Convert tokens to millions
        df['total_tokens_M'] = df['total_tokens'] / 1e6
        
        # Handle timing columns - prioritize train time over wall time
        # For efficiency plots, we want pure training time (excludes validation)
        if 'train_time_total' in df.columns:
            # New format: has separate train time (excludes validation)
            df['train_hours'] = pd.to_numeric(df['train_time_total'], errors='coerce') / 3600
            df['wall_hours'] = pd.to_numeric(df.get('elapsed_time', df['train_time_total']), errors='coerce') / 3600
        elif 'elapsed_time' in df.columns:
            # Old format: only has wall time
            df['train_hours'] = pd.to_numeric(df['elapsed_time'], errors='coerce') / 3600
            df['wall_hours'] = df['train_hours']
        else:
            print(f"Warning: No timing columns in {fp}")
            df['train_hours'] = 0
            df['wall_hours'] = 0
        
        # Keep legacy column name for backwards compatibility
        df['elapsed_hours'] = df['wall_hours']
        
        # Handle validation loss (may be empty strings or missing)
        if 'val_loss' in df.columns:
            df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')
            if 'val_perplexity' in df.columns:
                df['val_perplexity'] = pd.to_numeric(df['val_perplexity'], errors='coerce')
        
        # Handle gradient norm
        if 'grad_norm' in df.columns:
            df['grad_norm'] = pd.to_numeric(df['grad_norm'], errors='coerce')
        
        # Handle learning rate
        if 'lr' in df.columns:
            df['lr'] = pd.to_numeric(df['lr'], errors='coerce')
        
        # Ensure perplexity is numeric
        df['perplexity'] = pd.to_numeric(df['perplexity'], errors='coerce')
        
        # Handle tokens_per_sec
        if 'tokens_per_sec' not in df.columns:
            print(f"Warning: No tokens_per_sec in {fp}")
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid data loaded from any file")
    
    return pd.concat(dfs, ignore_index=True)


def smooth(values, window=50):
    """Apply exponential moving average smoothing."""
    return pd.Series(values).ewm(span=window, adjust=False).mean()


def plot_loss_curves(df, output_dir, smooth_window=50):
    """Plot training loss over tokens."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Loss vs Tokens
    ax1 = axes[0]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        # Raw data (light)
        ax1.plot(mode_df['total_tokens_M'], mode_df['loss'], 
                 alpha=0.2, color=color, linewidth=0.5)
        
        # Smoothed (bold)
        smoothed_loss = smooth(mode_df['loss'].values, smooth_window)
        ax1.plot(mode_df['total_tokens_M'], smoothed_loss, 
                 color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Tokens (Millions)')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs Tokens Processed')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=2, top=12)
    
    # Right: Loss vs Wall Time
    ax2 = axes[1]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        ax2.plot(mode_df['elapsed_hours'], mode_df['loss'], 
                 alpha=0.2, color=color, linewidth=0.5)
        
        smoothed_loss = smooth(mode_df['loss'].values, smooth_window)
        ax2.plot(mode_df['elapsed_hours'], smoothed_loss, 
                 color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('Wall Time (Hours)')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss vs Wall Time')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(bottom=2, top=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: loss_curves.png/pdf")


def plot_perplexity(df, output_dir, smooth_window=50):
    """Plot perplexity over training."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        # Cap perplexity for visualization (early steps can be huge)
        ppl = mode_df['perplexity'].clip(upper=1000)
        
        ax.plot(mode_df['total_tokens_M'], ppl, 
                alpha=0.15, color=color, linewidth=0.5)
        
        smoothed_ppl = smooth(ppl.values, smooth_window)
        ax.plot(mode_df['total_tokens_M'], smoothed_ppl, 
                color=color, linewidth=2, label=label)
    
    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Training Perplexity')
    ax.set_yscale('log')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(bottom=50, top=1000)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: perplexity.png")


def plot_throughput(df, output_dir, smooth_window=20):
    """Plot throughput comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Throughput over time
    ax1 = axes[0]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].iloc[5:].reset_index(drop=True)  # Skip first few (compilation)
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        smoothed = smooth(mode_df['tokens_per_sec'].values, smooth_window)
        ax1.plot(mode_df['step'], smoothed, 
                 color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Tokens / Second')
    ax1.set_title('Training Throughput Over Time')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    
    # Right: Throughput distribution (box plot)
    ax2 = axes[1]
    
    # Prepare data for box plot (skip first 10 steps - compilation)
    plot_data = df[df['step'] > 10].copy()
    
    # Get colors in order of modes
    mode_order = plot_data['mode_display'].unique()
    palette = {m: COLORS.get(df[df['mode_display'] == m]['mode'].iloc[0], '#888888') for m in mode_order}
    
    sns.boxplot(data=plot_data, x='mode_display', y='tokens_per_sec', 
                ax=ax2, palette=palette, order=mode_order)
    
    ax2.set_xlabel('')
    ax2.set_ylabel('Tokens / Second')
    ax2.set_title('Throughput Distribution')
    
    # Add mean values as text
    for i, mode in enumerate(mode_order):
        mode_data = plot_data[plot_data['mode_display'] == mode]['tokens_per_sec']
        mean_val = mode_data.mean()
        ax2.annotate(f'μ = {mean_val:.0f}', 
                     xy=(i, mean_val), 
                     xytext=(i + 0.25, mean_val),
                     fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: throughput.png")


def plot_efficiency(df, output_dir):
    """Plot tokens processed per unit train time - the key efficiency metric."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use train_hours (excludes validation) for fair throughput comparison
    time_col = 'train_hours' if 'train_hours' in df.columns else 'elapsed_hours'
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        ax.plot(mode_df[time_col], mode_df['total_tokens_M'], 
                color=color, linewidth=2.5, label=label)
    
    ax.set_xlabel('Training Time (Hours)' + (' [excludes validation]' if time_col == 'train_hours' else ''))
    ax.set_ylabel('Tokens Processed (Millions)')
    ax.set_title('Training Efficiency: Tokens vs Training Time')
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower right')
    
    # Add efficiency annotation
    modes = df['mode'].unique()
    if len(modes) == 2:
        final_times = {mode: df[df['mode'] == mode][time_col].iloc[-1] for mode in modes}
        final_tokens = {mode: df[df['mode'] == mode]['total_tokens_M'].iloc[-1] for mode in modes}
        
        # Calculate speedup
        bf16_mode = [m for m in modes if 'bf16' in m.lower()][0] if any('bf16' in m.lower() for m in modes) else modes[0]
        fp8_mode = [m for m in modes if 'fp8' in m.lower()][0] if any('fp8' in m.lower() for m in modes) else modes[1]
        
        bf16_rate = final_tokens[bf16_mode] / final_times[bf16_mode]
        fp8_rate = final_tokens[fp8_mode] / final_times[fp8_mode]
        speedup = fp8_rate / bf16_rate
        
        ax.annotate(f'FP8 Speedup: {speedup:.2f}x', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: efficiency.png")


def plot_summary_dashboard(df, output_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Loss vs Tokens (large, top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        smoothed = smooth(mode_df['loss'].values, 50)
        ax1.plot(mode_df['total_tokens_M'], smoothed, color=color, linewidth=2, label=label)
    ax1.set_xlabel('Tokens (M)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(4, 12)
    
    # 2. Stats box (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = "Training Summary\n" + "="*30 + "\n\n"
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        mode_display = mode_df['mode_display'].iloc[0]
        final_loss = mode_df['loss'].iloc[-1]
        final_ppl = mode_df['perplexity'].iloc[-1]
        total_tokens = mode_df['total_tokens'].iloc[-1] / 1e6
        total_time = mode_df['elapsed_time'].iloc[-1] / 3600
        avg_throughput = mode_df['tokens_per_sec'].iloc[10:].mean()
        
        stats_text += f"{mode_display}\n"
        stats_text += f"  Final Loss: {final_loss:.3f}\n"
        stats_text += f"  Final PPL: {final_ppl:.1f}\n"
        stats_text += f"  Tokens: {total_tokens:.1f}M\n"
        stats_text += f"  Time: {total_time:.2f}h\n"
        stats_text += f"  Avg tok/s: {avg_throughput:.0f}\n\n"
    
    # Calculate speedup
    modes = df['mode'].unique()
    if len(modes) == 2:
        throughputs = {mode: df[df['mode'] == mode]['tokens_per_sec'].iloc[10:].mean() for mode in modes}
        bf16_tp = min(throughputs.values())
        fp8_tp = max(throughputs.values())
        speedup = fp8_tp / bf16_tp
        stats_text += f"FP8 Speedup: {speedup:.2f}x"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # 3. Throughput over time (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].iloc[5:]
        color = COLORS.get(mode, '#888888')
        smoothed = smooth(mode_df['tokens_per_sec'].values, 20)
        ax3.plot(mode_df['step'], smoothed, color=color, linewidth=1.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('tok/s')
    ax3.set_title('Throughput', fontweight='bold')
    
    # 4. Throughput histogram (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].iloc[10:]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        ax4.hist(mode_df['tokens_per_sec'], bins=30, alpha=0.6, 
                 color=color, label=label, edgecolor='white')
    ax4.set_xlabel('tok/s')
    ax4.set_ylabel('Count')
    ax4.set_title('Throughput Distribution', fontweight='bold')
    ax4.legend(fontsize=9)
    
    # 5. Efficiency (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        ax5.plot(mode_df['elapsed_hours'], mode_df['total_tokens_M'], 
                 color=color, linewidth=2)
    ax5.set_xlabel('Hours')
    ax5.set_ylabel('Tokens (M)')
    ax5.set_title('Tokens vs Time', fontweight='bold')
    
    # 6. Loss vs Time (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        smoothed = smooth(mode_df['loss'].values, 50)
        ax6.plot(mode_df['elapsed_hours'], smoothed, color=color, linewidth=1.5)
    ax6.set_xlabel('Hours')
    ax6.set_ylabel('Loss')
    ax6.set_title('Loss vs Wall Time', fontweight='bold')
    ax6.set_ylim(4, 12)
    
    # 7. Perplexity (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        ppl = mode_df['perplexity'].clip(upper=500)
        smoothed = smooth(ppl.values, 50)
        ax7.plot(mode_df['total_tokens_M'], smoothed, color=color, linewidth=1.5)
    ax7.set_xlabel('Tokens (M)')
    ax7.set_ylabel('Perplexity')
    ax7.set_title('Perplexity', fontweight='bold')
    ax7.set_yscale('log')
    
    # 8. Step time (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].iloc[5:]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        smoothed = smooth(mode_df['step_time_ms'].values, 20)
        ax8.plot(mode_df['step'], smoothed, color=color, linewidth=1.5, label=label)
    ax8.set_xlabel('Step')
    ax8.set_ylabel('Step Time (ms)')
    ax8.set_title('Step Time', fontweight='bold')
    ax8.legend(fontsize=9)
    
    # Main title
    fig.suptitle('FP8 vs BF16 Training Comparison - 1.5B Parameter Model on NVIDIA GB10', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'dashboard.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: dashboard.png/pdf")


def plot_loss_convergence_comparison(df, output_dir):
    """Plot loss at same token count to show FP8 matches BF16 numerically."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = df['mode'].unique()
    if len(modes) < 2:
        print("Skipping convergence comparison (need at least 2 modes)")
        return
    
    # Resample both to same token counts
    token_points = np.linspace(1e6, df['total_tokens'].max() * 0.95, 50)
    
    losses = {}
    for mode in modes:
        mode_df = df[df['mode'] == mode].sort_values('total_tokens')
        losses[mode] = np.interp(token_points, mode_df['total_tokens'], mode_df['loss'])
    
    # Compare each FP8 mode to BF16 if present
    bf16_mode = [m for m in modes if 'bf16' in m.lower()]
    fp8_modes = [m for m in modes if 'fp8' in m.lower()]
    
    if bf16_mode and fp8_modes:
        bf16_mode = bf16_mode[0]
        for fp8_mode in fp8_modes:
            loss_diff = np.array(losses[bf16_mode]) - np.array(losses[fp8_mode])
            color = COLORS.get(fp8_mode, '#888888')
            label = fp8_mode.replace('fp8_', 'FP8 ')
            ax.plot(token_points / 1e6, loss_diff, linewidth=1.5, 
                    color=color, label=f'BF16 - {label}')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Loss Difference (BF16 - FP8)')
    ax.set_title('Numerical Convergence: BF16 vs FP8 Loss Difference')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: convergence_comparison.png")


def plot_gradient_norms(df, output_dir, smooth_window=50):
    """Plot gradient norms over training."""
    if 'grad_norm' not in df.columns or df['grad_norm'].isna().all():
        print("Skipping gradient norm plot (no data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Gradient norm over steps
    ax1 = axes[0]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].dropna(subset=['grad_norm'])
        if len(mode_df) == 0:
            continue
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        ax1.plot(mode_df['step'], mode_df['grad_norm'], 
                 alpha=0.2, color=color, linewidth=0.5)
        smoothed = smooth(mode_df['grad_norm'].values, smooth_window)
        ax1.plot(mode_df['step'], smoothed, 
                 color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Norm Over Training')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_yscale('log')
    
    # Right: Gradient norm distribution
    ax2 = axes[1]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].dropna(subset=['grad_norm'])
        if len(mode_df) == 0:
            continue
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        ax2.hist(mode_df['grad_norm'], bins=50, alpha=0.5, 
                 color=color, label=label, edgecolor='white')
    
    ax2.set_xlabel('Gradient Norm')
    ax2.set_ylabel('Count')
    ax2.set_title('Gradient Norm Distribution')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: gradient_norms.png")


def plot_validation_loss(df, output_dir, smooth_window=10):
    """Plot validation loss if available."""
    if 'val_loss' not in df.columns or df['val_loss'].isna().all():
        print("Skipping validation loss plot (no data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Training vs Validation loss
    ax1 = axes[0]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        
        # Training loss (smoothed)
        train_smooth = smooth(mode_df['loss'].values, smooth_window * 5)
        ax1.plot(mode_df['total_tokens_M'], train_smooth, 
                 color=color, linewidth=1.5, linestyle='-', alpha=0.7)
        
        # Validation loss (points where available)
        val_df = mode_df.dropna(subset=['val_loss'])
        if len(val_df) > 0:
            ax1.scatter(val_df['total_tokens_M'], val_df['val_loss'], 
                       color=color, s=30, marker='o', label=f'{label} (val)',
                       edgecolors='white', linewidths=0.5)
    
    ax1.set_xlabel('Tokens (Millions)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=2)
    
    # Right: Validation loss comparison
    ax2 = axes[1]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].dropna(subset=['val_loss'])
        if len(mode_df) == 0:
            continue
        color = COLORS.get(mode, '#888888')
        label = mode_df['mode_display'].iloc[0]
        ax2.plot(mode_df['total_tokens_M'], mode_df['val_loss'], 
                 color=color, linewidth=2, marker='o', markersize=4, label=label)
    
    ax2.set_xlabel('Tokens (Millions)')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: validation_loss.png")


def plot_learning_rate(df, output_dir):
    """Plot learning rate schedule."""
    if 'lr' not in df.columns or df['lr'].isna().all():
        print("Skipping learning rate plot (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Just plot one mode since LR schedule should be same
    mode = df['mode'].unique()[0]
    mode_df = df[df['mode'] == mode].dropna(subset=['lr'])
    
    if len(mode_df) > 0:
        ax.plot(mode_df['step'], mode_df['lr'], linewidth=2, color='#2E86AB')
        ax.fill_between(mode_df['step'], 0, mode_df['lr'], alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Warmup + Cosine Decay)')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: learning_rate.png")


def main():
    parser = argparse.ArgumentParser(description='Plot FP8 training metrics')
    parser.add_argument('files', nargs='+', help='CSV log files to plot')
    parser.add_argument('--output', '-o', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--smooth', type=int, default=50, help='Smoothing window size')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {len(args.files)} file(s)...")
    df = load_and_prepare_data(args.files)
    
    print(f"Data loaded: {len(df)} rows, modes: {df['mode'].unique()}")
    print(f"Generating plots in {output_dir}/\n")
    
    # Generate all plots
    plot_loss_curves(df, output_dir, args.smooth)
    plot_perplexity(df, output_dir, args.smooth)
    plot_throughput(df, output_dir)
    plot_efficiency(df, output_dir)
    plot_loss_convergence_comparison(df, output_dir)
    plot_gradient_norms(df, output_dir, args.smooth)
    plot_validation_loss(df, output_dir)
    plot_learning_rate(df, output_dir)
    plot_summary_dashboard(df, output_dir)
    
    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
