import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
data = pd.read_csv("student_data.csv")

# Target and features
y = data["passed"].map({"no": 0, "yes": 1})
X = data.drop(columns=["passed"])

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 6. Train & evaluate ----------
metrics = {}
best_name = None
best_acc = -1.0
best_pipeline = None

for name, clf in models.items():
    print(f"\n=== Training {name} ===")
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], average="binary", pos_label=1
    )

    cm_algo = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix (rows = true [Fail, Pass], cols = predicted [Fail, Pass]):")
    print(cm_algo)

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Precision (pass): {precision:.4f}")
    print(f"{name} Recall (pass): {recall:.4f}")
    print(f"{name} F1-score (pass): {f1:.4f}")
    print(classification_report(y_test, y_pred))

    metrics[name] = {
        "accuracy": acc,
        "precision_pass": precision,
        "recall_pass": recall,
        "f1_pass": f1
    }

    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_pipeline = pipe

print("\n=== Best Model ===")
print(f"Best algorithm: {best_name} with accuracy {best_acc:.4f}")

# ---------- 7. Save best model ----------
joblib.dump(best_pipeline, "model.pkl")
print("✅ Best model saved as model.pkl")

# ---------- 8. Charts folder ----------
CHART_FOLDER = os.path.join("static", "charts")
os.makedirs(CHART_FOLDER, exist_ok=True)

# ---------- 9. Confusion matrices ----------
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
    "metrics": metrics,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized": cm_norm.tolist(),
    "confusion_labels": ["Fail", "Pass"]
}
with open("algo_metrics.json", "w") as f:
    json.dump(algo_info, f, indent=4)
print("✅ Algorithm metrics + confusion matrix saved in algo_metrics.json")

# ---------- 11. Accuracy comparison chart ----------
names = list(metrics.keys())
accs = [metrics[n]["accuracy"] * 100 for n in names]

plt.figure(figsize=(8, 5))
plt.bar(names, accs)
plt.ylabel("Accuracy (%)")
plt.title("Algorithm Accuracy Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

algo_chart_path = os.path.join(CHART_FOLDER, "algo_comparison.png")
plt.savefig(algo_chart_path)
plt.close()
print(f"✅ Algorithm comparison chart saved to {algo_chart_path}")

# ---------- 12. Feature importance + SHAP ----------
if best_name in ["Random Forest", "XGBoost"]:
    # Get feature names automatically
    feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()

    # Extract importance
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
    explainer = shap.TreeExplainer(best_pipeline.named_steps["model"])
    shap_values = explainer.shap_values(
        best_pipeline.named_steps["preprocess"].transform(X_test)
    )
    shap.summary_plot(shap_values, feature_names, show=False)
    shap_path = os.path.join(CHART_FOLDER, "shap_summary.png")
    plt.savefig(shap_path)
    plt.close()
    print(f"✅ SHAP summary plot saved to {shap_path}")