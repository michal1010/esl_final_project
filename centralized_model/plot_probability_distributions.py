import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import joblib
from scipy import stats

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "centralized_dataset.csv"
MODEL_PATH = REPO_ROOT / "centralized_model" / "centralized_out" / "logreg_model.pkl"
OUT_DIR = REPO_ROOT / "centralized_model" / "centralized_out"

from fair_centralized import preprocess, create_logreg_model
from fairlearn.postprocessing import ThresholdOptimizer

SEED = 42
N_CLIENTS = 5


def plot_probability_distributions(
    y_test, 
    probs_raw, 
    probs_postprocessed, 
    race_test, 
    constraint_name="Demographic Parity",
    output_path=None
):
    """
    Plot probability distributions before and after fairness postprocessing.
    
    Parameters:
    -----------
    y_test : array
        True labels
    probs_raw : array
        Raw model probabilities P(Y=1)
    probs_postprocessed : array
        Post-processed probabilities or predictions
    race_test : pandas Series or array
        Race/group labels for each sample
    constraint_name : str
        Name of fairness constraint applied
    output_path : Path or str, optional
        Where to save the plot
    """
    
    # Get unique races
    unique_races = sorted(race_test.unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette for races
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_races)))
    
    # Plot for each race
    for idx, race in enumerate(unique_races):
        mask = race_test == race
        
        # Raw probabilities for this race
        probs_race_raw = probs_raw[mask]
        
        # Postprocessed probabilities for this race
        probs_race_post = probs_postprocessed[mask]
        
        # Convert postprocessed predictions to probabilities if they're binary
        # (ThresholdOptimizer returns 0/1, we'll add small noise to visualize)
        if len(np.unique(probs_race_post)) == 2:
            # Binary predictions - add small jitter for visualization
            probs_race_post = probs_race_post.astype(float)
            jitter = np.random.normal(0, 0.02, size=len(probs_race_post))
            probs_race_post = np.clip(probs_race_post + jitter, 0, 1)
        
        color = colors[idx]
        
        # Plot raw distribution (dashed)
        if len(probs_race_raw) > 10:  # Only plot if enough samples
            try:
                density_raw = stats.gaussian_kde(probs_race_raw)
                x_range = np.linspace(-0.2, 1.2, 300)
                y_raw = density_raw(x_range)
                ax.plot(x_range, y_raw, 
                       linestyle='--', 
                       color=color, 
                       alpha=0.7,
                       linewidth=2,
                       label=f"{race} (raw)")
            except:
                # Fallback to histogram if KDE fails
                ax.hist(probs_race_raw, bins=30, alpha=0.3, 
                       color=color, density=True, histtype='step',
                       linestyle='--', linewidth=2, label=f"{race} (raw)")
        
        # Plot postprocessed distribution (solid)
        if len(probs_race_post) > 10:  # Only plot if enough samples
            try:
                density_post = stats.gaussian_kde(probs_race_post)
                x_range = np.linspace(-0.2, 1.2, 300)
                y_post = density_post(x_range)
                ax.plot(x_range, y_post, 
                       linestyle='-', 
                       color=color, 
                       alpha=0.9,
                       linewidth=2.5,
                       label=f"{race} (postprocessed)")
            except:
                # Fallback to histogram if KDE fails
                ax.hist(probs_race_post, bins=30, alpha=0.5, 
                       color=color, density=True, histtype='step',
                       linestyle='-', linewidth=2.5, label=f"{race} (postprocessed)")
    
    ax.set_xlabel("Predicted probability P(Ŷ = 1)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Raw vs Postprocessed probabilities per race ({constraint_name})", 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.3, 1.3])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved to: {output_path}")
    
    return fig


def main():
    """Generate probability distribution plots for both fairness constraints."""
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    df_raw = pd.read_csv(DATA_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
    X_all, y_all, df_used = preprocess(df_raw)
    
    # Split data
    idx = np.arange(len(y_all))
    parts = np.array_split(idx, N_CLIENTS + 1)
    test_idx, train_idx = parts[0], np.concatenate(parts[1:])
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    df_test = df_used.iloc[test_idx].reset_index(drop=True)
    
    # Load or train model
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    else:
        print("Training model...")
        model = create_logreg_model()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
    
    # Get raw probabilities
    probs_raw = model.predict_proba(X_test)[:, 1]
    
    # Get race information
    race_test = df_test["race"].astype(str).str.strip()
    
    print(f"\nGenerating probability distribution plots...")
    
    # =====================================================
    # 1) Demographic Parity (Independence)
    # =====================================================
    print("\n1. Applying Demographic Parity constraint...")
    try:
        thresh_indep = ThresholdOptimizer(
            estimator=model,
            constraints="demographic_parity",
            predict_method="predict_proba",
            prefit=True,
            grid_size=1000,
        )
        thresh_indep.fit(X_test, y_test, sensitive_features=race_test)
        y_pred_indep = thresh_indep.predict(X_test, sensitive_features=race_test)
        
        # Create plot
        fig1 = plot_probability_distributions(
            y_test=y_test,
            probs_raw=probs_raw,
            probs_postprocessed=y_pred_indep,
            race_test=race_test,
            constraint_name="Demographic Parity",
            output_path=OUT_DIR / "probability_distributions_demographic_parity.png"
        )
        plt.close(fig1)
        
    except Exception as e:
        print(f"Error with demographic parity: {e}")
    
    # =====================================================
    # 2) Equalized Odds (Separation)
    # =====================================================
    print("\n2. Applying Equalized Odds constraint...")
    try:
        thresh_sep = ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",
            predict_method="predict_proba",
            prefit=True,
            grid_size=1000,
        )
        thresh_sep.fit(X_test, y_test, sensitive_features=race_test)
        y_pred_sep = thresh_sep.predict(X_test, sensitive_features=race_test)
        
        # Create plot
        fig2 = plot_probability_distributions(
            y_test=y_test,
            probs_raw=probs_raw,
            probs_postprocessed=y_pred_sep,
            race_test=race_test,
            constraint_name="Equalized Odds",
            output_path=OUT_DIR / "probability_distributions_equalized_odds.png"
        )
        plt.close(fig2)
        
    except Exception as e:
        print(f"Error with equalized odds: {e}")
    
    # =====================================================
    # 3) Combined plot with both constraints
    # =====================================================
    print("\n3. Creating combined comparison plot...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Demographic Parity subplot
        ax1 = axes[0]
        unique_races = sorted(race_test.unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_races)))
        
        for idx, race in enumerate(unique_races):
            mask = race_test == race
            probs_race_raw = probs_raw[mask]
            probs_race_indep = y_pred_indep[mask].astype(float)
            
            # Add jitter to binary predictions
            if len(np.unique(probs_race_indep)) == 2:
                jitter = np.random.normal(0, 0.02, size=len(probs_race_indep))
                probs_race_indep = np.clip(probs_race_indep + jitter, 0, 1)
            
            color = colors[idx]
            
            if len(probs_race_raw) > 10:
                try:
                    density_raw = stats.gaussian_kde(probs_race_raw)
                    x_range = np.linspace(-0.2, 1.2, 300)
                    ax1.plot(x_range, density_raw(x_range), '--', color=color, alpha=0.7, linewidth=2, label=f"{race} (raw)")
                    
                    density_post = stats.gaussian_kde(probs_race_indep)
                    ax1.plot(x_range, density_post(x_range), '-', color=color, alpha=0.9, linewidth=2.5, label=f"{race} (post)")
                except:
                    pass
        
        ax1.set_xlabel("Predicted probability P(Ŷ = 1)", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.set_title("Demographic Parity", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.3, 1.3])
        
        # Equalized Odds subplot
        ax2 = axes[1]
        for idx, race in enumerate(unique_races):
            mask = race_test == race
            probs_race_raw = probs_raw[mask]
            probs_race_sep = y_pred_sep[mask].astype(float)
            
            # Add jitter to binary predictions
            if len(np.unique(probs_race_sep)) == 2:
                jitter = np.random.normal(0, 0.02, size=len(probs_race_sep))
                probs_race_sep = np.clip(probs_race_sep + jitter, 0, 1)
            
            color = colors[idx]
            
            if len(probs_race_raw) > 10:
                try:
                    density_raw = stats.gaussian_kde(probs_race_raw)
                    x_range = np.linspace(-0.2, 1.2, 300)
                    ax2.plot(x_range, density_raw(x_range), '--', color=color, alpha=0.7, linewidth=2, label=f"{race} (raw)")
                    
                    density_post = stats.gaussian_kde(probs_race_sep)
                    ax2.plot(x_range, density_post(x_range), '-', color=color, alpha=0.9, linewidth=2.5, label=f"{race} (post)")
                except:
                    pass
        
        ax2.set_xlabel("Predicted probability P(Ŷ = 1)", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title("Equalized Odds", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.3, 1.3])
        
        plt.tight_layout()
        combined_path = OUT_DIR / "probability_distributions_combined.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {combined_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating combined plot: {e}")
    
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
