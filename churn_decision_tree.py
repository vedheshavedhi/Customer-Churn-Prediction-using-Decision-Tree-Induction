# Customer Churn Prediction using Decision Tree Induction
# Dataset: IBM Telco Customer Churn (7043 customers, 21 columns)
import os
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)

os.makedirs("outputs", exist_ok=True)

# ---------- 1. Data collection ----------
df = pd.read_csv("Telco-Customer-Churn.csv")
print("Shape:", df.shape)
print(df["Churn"].value_counts(), "\n")

# ---------- 2. Preprocessing ----------
df = df.drop(columns="customerID")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # blanks -> NaN
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True)             # one-hot encoding
y = df["Churn"]
print("Features after encoding:", X.shape[1])
# Decision trees are scale-invariant, so no scaling is applied.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", Xtr.shape, "Test:", Xte.shape, "\n")

# ---------- 3. Model development ----------
# Model 1: Decision Tree (Gini, unpruned) - baseline
m1 = DecisionTreeClassifier(criterion="gini", random_state=42).fit(Xtr, ytr)

# Model 2: Decision Tree (Entropy / information gain) tuned with GridSearchCV = pruned tree
grid = {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 20, 50],
        "ccp_alpha": [0, 0.0005, 0.001, 0.002]}
gs = GridSearchCV(DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=42),
                  grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
m2 = gs.best_estimator_
print("Best DT params:", gs.best_params_, " CV AUC: %.4f\n" % gs.best_score_)

# Model 3: Random Forest (ensemble of trees) - comparison
m3 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                            class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr, ytr)

models = {"Decision Tree (Gini, unpruned)": m1,
          "Decision Tree (Entropy, tuned)": m2,
          "Random Forest": m3}

# ---------- 4. Evaluation ----------
rows, probs = [], {}
for name, m in models.items():
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]; probs[name] = pr
    rows.append({"Model": name,
                 "Train Acc": round(m.score(Xtr, ytr), 4),
                 "Accuracy": round(accuracy_score(yte, p), 4),
                 "Precision": round(precision_score(yte, p), 4),
                 "Recall": round(recall_score(yte, p), 4),
                 "F1": round(f1_score(yte, p), 4),
                 "ROC-AUC": round(roc_auc_score(yte, pr), 4),
                 "CM": confusion_matrix(yte, p).tolist()})
res = pd.DataFrame(rows)
print(res.drop(columns="CM").to_string(index=False))
res.to_json("outputs/metrics.json", orient="records", indent=1)

# ---------- 5. Plots ----------
plt.rcParams.update({"font.size": 10})
# churn distribution
fig, ax = plt.subplots(figsize=(4.5, 3.4))
c = df["Churn"].map({0: "No churn", 1: "Churn"}).value_counts()
ax.bar(c.index, c.values, color=["#4C78A8", "#E45756"])
for i, v in enumerate(c.values): ax.text(i, v + 60, f"{v} ({v/len(df):.1%})", ha="center")
ax.set_ylabel("Customers"); ax.set_title("Target class distribution"); ax.set_ylim(0, 6000)
plt.tight_layout(); plt.savefig("outputs/fig_class_distribution.png", dpi=200); plt.close()

# churn by contract
fig, ax = plt.subplots(figsize=(4.8, 3.4))
r = df.groupby("Contract")["Churn"].mean() * 100
ax.bar(r.index, r.values, color="#F58518")
for i, v in enumerate(r.values): ax.text(i, v + 1, f"{v:.1f}%", ha="center")
ax.set_ylabel("Churn rate (%)"); ax.set_title("Churn rate by contract type")
plt.tight_layout(); plt.savefig("outputs/fig_churn_by_contract.png", dpi=200); plt.close()

# tree plot (top levels of tuned tree)
fig, ax = plt.subplots(figsize=(13, 5.5))
plot_tree(m2, max_depth=2, feature_names=X.columns, class_names=["Stay", "Churn"],
          filled=True, rounded=True, fontsize=9, ax=ax, proportion=True)
plt.tight_layout(); plt.savefig("outputs/fig_decision_tree.png", dpi=170); plt.close()
open("outputs/tree_rules.txt", "w").write(export_text(m2, feature_names=list(X.columns), max_depth=4))

# confusion matrices
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, m) in zip(axs, models.items()):
    ConfusionMatrixDisplay.from_estimator(m, Xte, yte, display_labels=["Stay", "Churn"],
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(name, fontsize=9)
plt.tight_layout(); plt.savefig("outputs/fig_confusion_matrices.png", dpi=170); plt.close()

# ROC
fig, ax = plt.subplots(figsize=(5.2, 4.2))
for name, pr in probs.items():
    f, t, _ = roc_curve(yte, pr); ax.plot(f, t, label=f"{name} (AUC={roc_auc_score(yte, pr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves"); ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("outputs/fig_roc.png", dpi=200); plt.close()

# feature importance (tuned tree)
imp = pd.Series(m2.feature_importances_, X.columns).sort_values(ascending=False).head(8)[::-1]
fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.barh(imp.index, imp.values, color="#54A24B"); ax.set_title("Top features - tuned decision tree")
ax.set_xlabel("Importance (information gain)")
plt.tight_layout(); plt.savefig("outputs/fig_feature_importance.png", dpi=200); plt.close()
print("\nTop features:\n", imp[::-1].round(4).to_string())
imp[::-1].round(4).to_json("outputs/top_features.json")
print("Tree depth/leaves (tuned):", m2.get_depth(), m2.get_n_leaves(), "| unpruned:", m1.get_depth(), m1.get_n_leaves())
