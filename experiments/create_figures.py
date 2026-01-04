"""Create figures for VSAR paper from experimental results."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
})

results_dir = Path(__file__).parent / "results"
figures_dir = Path(__file__).parent.parent / "paper" / "figures"
figures_dir.mkdir(exist_ok=True)

# Figure 1: Encoding Comparison (Binary Predicates)
print("Creating Figure 1: Encoding Comparison...")
df = pd.read_csv(results_dir / "encoding_comparison.csv")
binary_data = df[df['arity'] == 2]

encodings = ['role_filler', 'shift', 'hybrid', 'hybrid_cancel']
labels = ['Role-Filler', 'Shift', 'Hybrid', 'Hybrid+Cancel']
similarities = [binary_data[binary_data['encoding_method'] == enc]['mean'].values[0]
                for enc in encodings]

fig, ax = plt.subplots(figsize=(3.3, 2.2))
bars = ax.bar(range(len(labels)), similarities, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
ax.set_ylabel('Similarity')
ax.set_xlabel('Encoding Method')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, similarities)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(figures_dir / "encoding_comparison.pdf", bbox_inches='tight')
plt.savefig(figures_dir / "encoding_comparison.png", bbox_inches='tight', dpi=300)
print(f"  Saved to {figures_dir / 'encoding_comparison.pdf'}")

# Figure 2: Arity Scaling
print("\nCreating Figure 2: Arity Scaling...")
arity_data = []
for arity in [2, 3, 4]:
    arity_df = df[df['arity'] == arity]
    for enc in encodings:
        enc_data = arity_df[arity_df['encoding_method'] == enc]
        if not enc_data.empty:
            arity_data.append({
                'arity': arity,
                'encoding': enc,
                'similarity': enc_data['mean'].values[0]
            })

arity_df = pd.DataFrame(arity_data)

fig, ax = plt.subplots(figsize=(3.3, 2.2))

colors = {'role_filler': '#d62728', 'shift': '#ff7f0e', 'hybrid': '#2ca02c', 'hybrid_cancel': '#1f77b4'}
markers = {'role_filler': 'o', 'shift': 's', 'hybrid': '^', 'hybrid_cancel': 'D'}
labels_map = {'role_filler': 'Role-Filler', 'shift': 'Shift',
              'hybrid': 'Hybrid', 'hybrid_cancel': 'Hybrid+Cancel'}

for enc in encodings:
    enc_df = arity_df[arity_df['encoding'] == enc]
    ax.plot(enc_df['arity'], enc_df['similarity'],
            marker=markers[enc], color=colors[enc],
            label=labels_map[enc], linewidth=1.5, markersize=5)

ax.set_xlabel('Predicate Arity')
ax.set_ylabel('Similarity')
ax.set_xticks([2, 3, 4])
ax.set_ylim([0, 1.0])
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(figures_dir / "arity_scaling.pdf", bbox_inches='tight')
plt.savefig(figures_dir / "arity_scaling.png", bbox_inches='tight', dpi=300)
print(f"  Saved to {figures_dir / 'arity_scaling.pdf'}")

# Figure 3: Multi-hop Reasoning
print("\nCreating Figure 3: Multi-hop Reasoning...")
multihop_df = pd.read_csv(results_dir / "multihop_reasoning.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 1.5))

# Recall plot
ax1.plot(multihop_df['hop'], multihop_df['recall'],
         marker='o', color='#1f77b4', linewidth=2, markersize=6)
ax1.set_xlabel('Number of Hops')
ax1.set_ylabel('Recall')
ax1.set_xticks([1, 2, 3])
ax1.set_ylim([0, 1.05])
ax1.grid(True, alpha=0.3, linestyle='--')

# Similarity plot
ax2.plot(multihop_df['hop'], multihop_df['avg_similarity'],
         marker='s', color='#2ca02c', linewidth=2, markersize=6)
ax2.set_xlabel('Number of Hops')
ax2.set_ylabel('Similarity')
ax2.set_xticks([1, 2, 3])
ax2.set_ylim([0.85, 0.95])
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(figures_dir / "multihop_reasoning.pdf", bbox_inches='tight')
plt.savefig(figures_dir / "multihop_reasoning.png", bbox_inches='tight', dpi=300)
print(f"  Saved to {figures_dir / 'multihop_reasoning.pdf'}")

print("\nAll figures created successfully!")
print(f"\nFigures saved to: {figures_dir}")
