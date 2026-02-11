import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

ROC_PATH = "centralized_model/centralized_out/roc_points.csv"

# Read ROC data
df = pd.read_csv(ROC_PATH)

# Get unique scenarios and races
scenarios = sorted(df["scenario"].unique())
races = sorted(df["race"].unique())

# Create color map for races
colors = cm.get_cmap('tab10', len(races))
race_colors = {race: colors(i) for i, race in enumerate(races)}

# Create a figure with subplots for each scenario
fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5))

# If only one scenario, axes won't be an array
if len(scenarios) == 1:
    axes = [axes]

ax = axes[0]
scenario_data = df[df["scenario"] == scenarios[1]]

# Plot ROC for each race
for race in races:
    race_data = scenario_data[scenario_data["race"] == race].sort_values("FPR")
    if len(race_data) == 0:
        continue
    auc_val = race_data["auc"].iloc[0] if len(race_data) > 0 else float("nan")
    ax.plot(race_data["FPR"], race_data["TPR"], 
            color=race_colors[race],
            label=f"{race} (AUC={auc_val:.3f})", 
            linewidth=2)

# Diagonal baseline (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random", alpha=0.5)

ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
ax.set_title(f"ROC Curve - {scenarios[1].capitalize()}", fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])


for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    scenario_data = df[df["scenario"] == scenario]
    
    # Plot ROC for each race
    for race in races:
        race_data = scenario_data[scenario_data["race"] == race].sort_values("FPR")
        if len(race_data) == 0:
            continue
        auc_val = race_data["auc"].iloc[0] if len(race_data) > 0 else float("nan")
        ax.plot(race_data["FPR"], race_data["TPR"], 
                color=race_colors[race],
                label=f"{race} (AUC={auc_val:.3f})", 
                linewidth=2)
    
    # Diagonal baseline (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random", alpha=0.5)
    
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax.set_title(f"ROC Curve - {scenario.capitalize()}", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.tight_layout()

# Save figure
out_path = Path(ROC_PATH).parent / "roc_curves.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"ROC curves saved to: {out_path}")

plt.show()