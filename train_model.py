import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- NEW: XGBoost + SHAP ----------
from xgboost import XGBClassifier
import shap

# ---------- 1. Load dataset ----------
data = pd.read_csv("uploads/student_data.csv")

# Target and features
y = data["passed"].map({"no": 0, "yes": 1})
X = data.drop(columns=["passed"])

n_samples = len(data)
print(f"Total samples in dataset: {n_samples}")
print(y.value_counts().rename({0: "Fail (0)", 1: "Pass (1)"}))

# ---------- Small dataset warning ----------
if n_samples < 100:
    print("\n⚠ WARNING: Dataset is small "
          f"({n_samples} samples). Model accuracy may not generalize well.\n")

# ---------- 2. Column types ----------
categorical_features = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# ---------- 3. Handle class imbalance ----------
classes = np.array([0, 1], dtype=int)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)
cw_dict = {int(cls): float(w) for cls, w in zip(classes, class_weights)}
print("Class weights:", cw_dict)

# ---------- 4. Define algorithms ----------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight=cw_dict,
        solver="lbfgs"
    ),
    "SVM (linear kernel)": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    ),
    "SVM (RBF kernel)": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight=cw_dict
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=cw_dict[0] / cw_dict[1]
    ),
}

# ---------- 5. Train/test split ----------
# You can increase test_size if you want more test samples (e.g. 0.3 or 0.4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5b. Prepare Stratified K-Fold ----------
# Use min(class_count, 5) to avoid errors on tiny datasets
class_counts = y.value_counts()
min_class_count = class_counts.min()
cv_splits = min(5, int(min_class_count))

if cv_splits < 2:
    print("\n⚠ Not enough samples per class for cross-validation. "
          "Skipping CV and using only train/test split.\n")
    use_cv = False
else:
    use_cv = True
    print(f"\nUsing StratifiedKFold cross-validation with {cv_splits} splits.\n")
    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

# ---------- 6. Train & evaluate ----------
metrics = {}
best_name = None
best_score_for_selection = -1.0  # we will use CV mean if available, else test accuracy
best_pipeline = None

for name, clf in models.items():
    print(f"\n=== Training {name} ===")

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ])

    # ---------- Cross-Validation ----------
    cv_mean = None
    cv_std = None
    if use_cv:
        print("Performing cross-validation...")
        cv_scores = cross_val_score(
            pipe,
            X,
            y,
            cv=skf,
            scoring="accuracy",
            n_jobs=None
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        print(f"{name} CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # ---------- Train on training set ----------
    pipe.fit(X_train, y_train)

    # ---------- Test set evaluation ----------
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], average="binary", pos_label=1
    )

    cm_algo = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix (rows = true [Fail, Pass], "
          "cols = predicted [Fail, Pass]):")
    print(cm_algo)

    print(f"{name} Test Accuracy: {acc:.4f}")
    print(f"{name} Precision (pass): {precision:.4f}")
    print(f"{name} Recall (pass): {recall:.4f}")
    print(f"{name} F1-score (pass): {f1:.4f}")
    print(classification_report(y_test, y_pred))

    metrics[name] = {
        "test_accuracy": acc,
        "precision_pass": precision,
        "recall_pass": recall,
        "f1_pass": f1,
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
    }

    # ---------- Model selection (prefer CV mean if available) ----------
    if cv_mean is not None:
        score_for_selection = cv_mean
    else:
        score_for_selection = acc

    if score_for_selection > best_score_for_selection:
        best_score_for_selection = score_for_selection
        best_name = name
        best_pipeline = pipe

print("\n=== Best Model ===")
if use_cv:
    print(f"Best algorithm (by CV mean accuracy): {best_name} "
          f"with score {best_score_for_selection:.4f}")
else:
    print(f"Best algorithm (by test accuracy): {best_name} "
          f"with score {best_score_for_selection:.4f}")

# ---------- 7. Save best model ----------
joblib.dump(best_pipeline, "model.pkl")
print("✅ Best model saved as model.pkl")

# ---------- 8. Charts folder ----------
CHART_FOLDER = os.path.join("static", "charts")
os.makedirs(CHART_FOLDER, exist_ok=True)

# ---------- 9. Confusion matrices for best model ----------
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])

plt.figure(figsize=(4, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap="Oranges", values_format="d", colorbar=True)
plt.title("Confusion Matrix (Counts)")
cm_raw_path = os.path.join(CHART_FOLDER, "confusion_matrix_raw.png")
plt.tight_layout()
plt.savefig(cm_raw_path)
plt.close()
print(f"✅ Raw confusion matrix saved to {cm_raw_path}")

cm_norm = confusion_matrix(y_test, y_pred_best, labels=[0, 1], normalize="true")
plt.figure(figsize=(4, 4))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Fail", "Pass"])
disp_norm.plot(cmap="Blues", values_format=".2f", colorbar=True)
plt.title("Confusion Matrix (Normalized by True Class)")
cm_norm_path = os.path.join(CHART_FOLDER, "confusion_matrix_norm.png")
plt.tight_layout()
plt.savefig(cm_norm_path)
plt.close()
print(f"✅ Normalized confusion matrix saved to {cm_norm_path}")

# ---------- 10. Save metrics ----------
algo_info = {
    "best_model": best_name,
    "selection_metric": "cv_mean_accuracy" if use_cv else "test_accuracy",
    "metrics": metrics,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized": cm_norm.tolist(),
    "confusion_labels": ["Fail", "Pass"],
}
with open("algo_metrics.json", "w") as f:
    json.dump(algo_info, f, indent=4)
print("✅ Algorithm metrics + confusion matrix saved in algo_metrics.json")

# ---------- 11. Accuracy comparison chart ----------
names = list(metrics.keys())
chart_scores = []
for n in names:
    m = metrics[n]
    if use_cv and m["cv_mean_accuracy"] is not None:
        chart_scores.append(m["cv_mean_accuracy"] * 100.0)
    else:
        chart_scores.append(m["test_accuracy"] * 100.0)

plt.figure(figsize=(8, 5))
plt.bar(names, chart_scores)
plt.ylabel("Accuracy (%)")
if use_cv:
    plt.title("Algorithm Accuracy Comparison (CV mean or Test Accuracy)")
else:
    plt.title("Algorithm Accuracy Comparison (Test Accuracy)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

algo_chart_path = os.path.join(CHART_FOLDER, "algo_comparison.png")
plt.savefig(algo_chart_path)
plt.close()
print(f"✅ Algorithm comparison chart saved to {algo_chart_path}")

# ---------- 12. Feature importance + SHAP for best tree-based model ----------
if best_name in ["Random Forest", "XGBoost"]:
    feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = best_pipeline.named_steps["model"].feature_importances_

    # Plot top 15 features
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1][:15]
    plt.barh(range(len(indices)), importances[indices][::-1], align="center")
    plt.yticks(range(len(indices)), feature_names[indices][::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top Features ({best_name})")
    plt.tight_layout()

    feat_imp_path = os.path.join(CHART_FOLDER, "feature_importance.png")
    plt.savefig(feat_imp_path)
    plt.close()
    print(f"✅ Feature importance chart saved to {feat_imp_path}")

    # SHAP summary
    print("Computing SHAP values (may take a bit on larger datasets)...")
    explainer = shap.TreeExplainer(best_pipeline.named_steps["model"])
    X_test_transformed = best_pipeline.named_steps["preprocess"].transform(X_test)
    shap_values = explainer.shap_values(X_test_transformed)
    shap.summary_plot(shap_values, feature_names, show=False)
    shap_path = os.path.join(CHART_FOLDER, "shap_summary.png")
    plt.savefig(shap_path)
    plt.close()
    print(f"✅ SHAP summary plot saved to {shap_path}")
else:
    print(f"\nℹ Best model is {best_name}, which is not tree-based. "
          "Skipping feature importance and SHAP plots.\n")
