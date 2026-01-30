import pandas as pd
import matplotlib.pyplot as plt

ROC_PATH = "centralized_model/centralized_out/roc_points.csv"

df = pd.read_csv(ROC_PATH)

plt.figure()
for race in df["race"].unique():
    sub = df[df["race"] == race].sort_values("FPR")
    auc = sub["auc"].iloc[0] if "auc" in sub.columns and len(sub) else float("nan")
    plt.plot(sub["FPR"], sub["TPR"], label=f"{race} (AUC={auc:.3f})")

# diagonal baseline
plt.plot([0, 1], [0, 1], linestyle="--", label="random")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC by Race (Centralized)")
plt.legend()
plt.grid(True)
plt.show()