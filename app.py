import os
import json
from datetime import datetime

from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------- PATH SETUP (ABSOLUTE) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
CHART_FOLDER = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# ---------- MODEL + METRICS LOAD ----------
MODEL = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

try:
    with open(os.path.join(BASE_DIR, "algo_metrics.json"), "r") as f:
        ALGO_METRICS = json.load(f)
except Exception:
    ALGO_METRICS = None

# Default dataset path (used for default predictions on GET)
DEFAULT_DATA_FILE = os.path.join(UPLOAD_FOLDER, "student_data.csv")

# ---------- TRAINING DATA PREVIEW ----------
TRAIN_TABLE_HTML = None
try:
    if os.path.exists(DEFAULT_DATA_FILE):
        if DEFAULT_DATA_FILE.endswith(".csv"):
            _train_df = pd.read_csv(DEFAULT_DATA_FILE)
        elif DEFAULT_DATA_FILE.endswith((".xlsx", ".xls")):
            _train_df = pd.read_excel(DEFAULT_DATA_FILE)
        else:
            _train_df = None

        if _train_df is not None:
            TRAIN_TABLE_HTML = _train_df.head(20).to_html(
                classes="table table-striped table-sm table-bordered align-middle mb-0",
                index=False,
            )
except Exception as e:
    print("Error loading training dataset preview:", e)
    TRAIN_TABLE_HTML = None


def create_charts(df):
    """
    Create charts from dataframe with predictions and
    return dict of chart metadata (filename + title + description).
    """
    charts = {}

    # -------- 1) Pass vs Fail count --------
    if "Predicted" in df.columns:
        plt.figure()
        counts = df["Predicted"].value_counts()
        counts = counts.reindex(["yes", "no"]).fillna(0)
        counts.plot(kind="bar")
        plt.title("Count of Predicted Pass vs Fail")
        plt.xlabel("Prediction")
        plt.ylabel("Number of Students")
        file1 = os.path.join(CHART_FOLDER, "predicted_counts.png")
        plt.tight_layout()
        plt.savefig(file1)
        plt.close()
        charts["predicted_counts"] = {
            "file": "charts/predicted_counts.png",
            "title": "Predicted Pass/Fail Distribution",
            "desc": (
                "This bar chart shows how many students are predicted to pass and how many are "
                "predicted to fail. It gives a quick overview of overall performance in the dataset."
            ),
        }

    # -------- 2) Average study time by prediction --------
    if "studytime" in df.columns and "Predicted" in df.columns:
        try:
            plt.figure()
            df.groupby("Predicted")["studytime"].mean().plot(kind="bar")
            plt.title("Average Study Time by Prediction")
            plt.xlabel("Prediction")
            plt.ylabel("Average Study Time (scale 1–4)")
            file2 = os.path.join(CHART_FOLDER, "studytime_by_pred.png")
            plt.tight_layout()
            plt.savefig(file2)
            plt.close()
            charts["studytime_by_pred"] = {
                "file": "charts/studytime_by_pred.png",
                "title": "Study Time vs Performance",
                "desc": (
                    "This chart compares the average study time (1–4 scale) for students predicted "
                    "to pass vs those predicted to fail. Higher bars for 'pass' usually indicate "
                    "that more study time is linked with better performance."
                ),
            }
        except Exception:
            pass

    # -------- 3) Absences histogram --------
    if "absences" in df.columns:
        try:
            plt.figure()
            df["absences"].plot(kind="hist", bins=15)
            plt.title("Distribution of Absences")
            plt.xlabel("Number of Absences")
            plt.ylabel("Number of Students")
            file3 = os.path.join(CHART_FOLDER, "absences_hist.png")
            plt.tight_layout()
            plt.savefig(file3)
            plt.close()
            charts["absences_hist"] = {
                "file": "charts/absences_hist.png",
                "title": "Absence Distribution",
                "desc": (
                    "This histogram shows how often students are absent from school. "
                    "A long tail to the right means some students have very high absence counts, "
                    "which can negatively impact their chance of passing."
                ),
            }
        except Exception:
            pass

    # -------- 4) Average failures by prediction --------
    if "failures" in df.columns and "Predicted" in df.columns:
        try:
            plt.figure()
            df.groupby("Predicted")["failures"].mean().plot(kind="bar")
            plt.title("Average Past Failures by Prediction")
            plt.xlabel("Prediction")
            plt.ylabel("Average Number of Past Failures")
            file4 = os.path.join(CHART_FOLDER, "failures_by_pred.png")
            plt.tight_layout()
            plt.savefig(file4)
            plt.close()
            charts["failures_by_pred"] = {
                "file": "charts/failures_by_pred.png",
                "title": "Past Failures vs Performance",
                "desc": (
                    "This chart shows the average number of previous subject failures for students "
                    "predicted to pass vs fail. Higher average failures for the 'fail' bar indicate "
                    "that past academic history is strongly related to current risk."
                ),
            }
        except Exception:
            pass

    # -------- 5) Correlation heatmap --------
    try:
        df_heat = df.copy()
        num_cols = [
            "age", "Medu", "Fedu", "traveltime", "studytime",
            "failures", "famrel", "freetime", "goout",
            "Dalc", "Walc", "health", "absences",
        ]
        if "Predicted" in df_heat.columns:
            df_heat["Predicted_num"] = (df_heat["Predicted"] == "yes").astype(int)
        else:
            df_heat["Predicted_num"] = 0

        num_cols_for_corr = [c for c in num_cols if c in df_heat.columns] + ["Predicted_num"]

        if len(num_cols_for_corr) >= 2:
            corr = df_heat[num_cols_for_corr].corr()

            plt.figure(figsize=(8, 6))
            plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
            plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    plt.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                    )
            plt.title("Correlation Heatmap (Features vs Prediction)")
            plt.tight_layout()
            file5 = os.path.join(CHART_FOLDER, "corr_heatmap.png")
            plt.savefig(file5, dpi=200)
            plt.close()
            charts["corr_heatmap"] = {
                "file": "charts/corr_heatmap.png",
                "title": "Correlation Heatmap",
                "desc": (
                    "The heatmap shows correlations between numeric features (age, study time, "
                    "absences, etc.) and the predicted result. Values close to +1 or -1 indicate a "
                    "strong relationship with the pass/fail prediction."
                ),
            }
    except Exception:
        pass

    # -------- 6) Confusion matrices (raw counts + normalized) --------
    cm_raw_path = os.path.join(CHART_FOLDER, "confusion_matrix_raw.png")
    if os.path.exists(cm_raw_path):
        charts["confusion_matrix_raw"] = {
            "file": "charts/confusion_matrix_raw.png",
            "title": "Confusion Matrix (Counts)",
            "desc": (
                "This confusion matrix shows the number of students correctly and incorrectly "
                "classified as pass or fail. The diagonal cells are correct predictions; "
                "off-diagonal cells show misclassifications (e.g., predicted pass but actually failed)."
            ),
        }

    cm_norm_path = os.path.join(CHART_FOLDER, "confusion_matrix_norm.png")
    if os.path.exists(cm_norm_path):
        charts["confusion_matrix_norm"] = {
            "file": "charts/confusion_matrix_norm.png",
            "title": "Confusion Matrix (Normalized)",
            "desc": (
                "This matrix shows the percentage of correct and incorrect predictions within each "
                "true class (pass/fail). Each row sums to 100%, so you can easily see how well the "
                "model performs for failing students vs passing students."
            ),
        }

    # -------- 7) Algorithm comparison chart --------
    algo_chart_path = os.path.join(CHART_FOLDER, "algo_comparison.png")
    if os.path.exists(algo_chart_path):
        charts["algo_comparison"] = {
            "file": "charts/algo_comparison.png",
            "title": "Algorithm Accuracy Comparison",
            "desc": (
                "This bar chart compares the accuracy of different machine learning algorithms. "
                "Depending on training, it may show cross-validation mean accuracy or test accuracy. "
                "The highest bar represents the algorithm selected as the best model."
            ),
        }

    # -------- 8) Feature importance (Random Forest / XGBoost) --------
    feat_imp_path = os.path.join(CHART_FOLDER, "feature_importance.png")
    if os.path.exists(feat_imp_path):
        charts["feature_importance"] = {
            "file": "charts/feature_importance.png",
            "title": "Top Features (Best Tree-Based Model)",
            "desc": (
                "This chart ranks the most important input features for the best tree-based model "
                "(Random Forest or XGBoost). Features at the top contribute the most to the model's "
                "decisions about whether a student will pass or fail."
            ),
        }

    # -------- 9) SHAP summary plot --------
    shap_path = os.path.join(CHART_FOLDER, "shap_summary.png")
    if os.path.exists(shap_path):
        charts["shap_summary"] = {
            "file": "charts/shap_summary.png",
            "title": "SHAP Summary Plot",
            "desc": (
                "The SHAP plot explains how each feature influences individual predictions for the "
                "tree-based model. It shows which features push predictions towards 'pass' or 'fail' "
                "and how strong that effect is across all students."
            ),
        }

    return charts


def generate_feedback(df):
    """
    Create human-readable feedback/reasons for pass/fail
    based on averages and overall distribution.
    """
    feedback = []

    total = len(df)
    if total == 0 or "Predicted" not in df.columns:
        return feedback

    # Overall pass rate
    pass_rate = (df["Predicted"] == "yes").mean() * 100.0
    fail_rate = 100.0 - pass_rate
    feedback.append({
        "title": "Overall Pass vs Fail",
        "text": (
            f"In this dataset, about {pass_rate:.1f}% of students are predicted to pass and "
            f"{fail_rate:.1f}% are predicted to fail. This gives a high-level view of how strong "
            "the overall academic performance is for the selected group of students."
        ),
    })

    passed = df[df["Predicted"] == "yes"]
    failed = df[df["Predicted"] == "no"]

    def mean_safe(sub_df, col):
        return sub_df[col].mean() if col in sub_df.columns and not sub_df.empty else None

    # Study time
    mp = mean_safe(passed, "studytime")
    mf = mean_safe(failed, "studytime")
    if mp is not None and mf is not None:
        feedback.append({
            "title": "Study Time Impact",
            "text": (
                f"Students predicted to pass study on average {mp:.2f} (on a 1–4 scale), "
                f"while students predicted to fail study about {mf:.2f}. "
                "This suggests that increasing regular study time can significantly improve "
                "the chances of passing."
            ),
        })

    # Absences
    mp = mean_safe(passed, "absences")
    mf = mean_safe(failed, "absences")
    if mp is not None and mf is not None:
        feedback.append({
            "title": "Absences Impact",
            "text": (
                f"Students predicted to pass have around {mp:.1f} absences on average, "
                f"compared to {mf:.1f} for students predicted to fail. "
                "Higher absence counts are associated with a higher risk of failing, "
                "highlighting the importance of regular attendance."
            ),
        })

    # Previous failures
    mp = mean_safe(passed, "failures")
    mf = mean_safe(failed, "failures")
    if mp is not None and mf is not None:
        feedback.append({
            "title": "Previous Failures",
            "text": (
                f"On average, students predicted to pass have {mp:.2f} previous failures, "
                f"whereas those predicted to fail have {mf:.2f}. "
                "This indicates that students with more past failures are more likely to be at risk, "
                "and may need additional support or intervention."
            ),
        })

    # Going out / socialising
    mp = mean_safe(passed, "goout")
    mf = mean_safe(failed, "goout")
    if mp is not None and mf is not None:
        feedback.append({
            "title": "Social/Going Out Behaviour",
            "text": (
                f"The average 'going out' score (1–5) is {mp:.2f} for students predicted to pass "
                f"and {mf:.2f} for those predicted to fail. Moderate social activity is normal, "
                "but very high levels can reduce study time and lower academic performance."
            ),
        })

    # Alcohol consumption
    mp_d = mean_safe(passed, "Dalc")
    mf_d = mean_safe(failed, "Dalc")
    mp_w = mean_safe(passed, "Walc")
    mf_w = mean_safe(failed, "Walc")
    if mp_d is not None and mf_d is not None and mp_w is not None and mf_w is not None:
        feedback.append({
            "title": "Alcohol Consumption",
            "text": (
                f"Workday alcohol use (1–5) is about {mp_d:.2f} for predicted pass and "
                f"{mf_d:.2f} for predicted fail. Weekend alcohol use is {mp_w:.2f} (pass) vs "
                f"{mf_w:.2f} (fail). Higher alcohol consumption is linked to poorer performance, "
                "especially when combined with low study time and high absence rates."
            ),
        })

    return feedback


def get_model_info():
    """
    Returns info about which algorithm is best and how others performed.
    Supports new metrics from train_model.py (CV + test accuracy).
    """
    if not ALGO_METRICS:
        return {
            "name": "Unknown Model",
            "details": "Model information is not available.",
            "note": "Run train_model.py again to generate and save algorithm comparison results.",
            "algos": [],
        }

    best_name = ALGO_METRICS.get("best_model", "Unknown")
    selection_metric = ALGO_METRICS.get("selection_metric", "test_accuracy")
    metrics = ALGO_METRICS.get("metrics", {})

    algo_rows = []
    text_lines = []

    for name, m in metrics.items():
        cv_mean = m.get("cv_mean_accuracy", None)
        test_acc = m.get("test_accuracy", 0.0)

        if selection_metric == "cv_mean_accuracy" and cv_mean is not None:
            acc_used = cv_mean
            acc_label = "CV mean accuracy"
        else:
            acc_used = test_acc
            acc_label = "Test accuracy"

        acc_percent = float(acc_used) * 100.0

        prec = float(m.get("precision_pass", 0.0))
        rec = float(m.get("recall_pass", 0.0))
        f1 = float(m.get("f1_pass", 0.0))

        text_lines.append(
            f"{name}: {acc_percent:.2f}% ({acc_label}), "
            f"Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}"
        )

        algo_rows.append({
            "name": name,
            "accuracy": round(acc_percent, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "is_best": (name == best_name),
        })

    if selection_metric == "cv_mean_accuracy":
        metric_text = (
            "The best model is selected using cross-validation mean accuracy "
            "(StratifiedKFold), which is more reliable for small datasets."
        )
    else:
        metric_text = (
            "The best model is selected based on its accuracy on the held-out test set."
        )

    details = (
        "The system evaluates multiple supervised learning algorithms on the student performance "
        "dataset: Logistic Regression, Support Vector Machines (with linear and RBF kernels), "
        "Random Forest, and XGBoost. These models are trained on the same preprocessed data and "
        "their performance is compared using accuracy, precision, recall and F1-score. "
        f"For this dataset, the best-performing algorithm is <b>{best_name}</b>. "
        + metric_text
    )

    note = (
        "Detailed performance of each algorithm:\n"
        + "\n".join(text_lines)
        + "\n\n"
          "• Logistic Regression: A simple and interpretable linear baseline model.\n"
          "• SVM (linear / RBF): Handles both linear and non-linear decision boundaries.\n"
          "• Random Forest: An ensemble of decision trees, very effective on tabular data.\n"
          "• XGBoost: A powerful gradient boosting algorithm, often strong on structured data."
    )

    return {
        "name": best_name,
        "details": details,
        "note": note,
        "algos": algo_rows,
    }


def run_predictions_on_dataframe(df):
    """
    Helper to run predictions on a dataframe and return:
    df_with_preds, out_path, charts, feedback
    """
    required_cols = list(MODEL.feature_names_in_)
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    preds = MODEL.predict(df[required_cols])
    probs = MODEL.predict_proba(df[required_cols])[:, 1]

    df = df.copy()
    df["Predicted"] = ["yes" if p == 1 else "no" for p in preds]
    df["Probability(%)"] = (probs * 100).round(2)

    out_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    df.to_csv(out_path, index=False)

    charts = create_charts(df)
    feedback = generate_feedback(df)

    return df, out_path, charts, feedback


def load_default_dataset_results():
    """
    Load the default student dataset, run predictions,
    and return (table_html, out_path, charts, feedback, error).
    This is used for the first view when the page loads.
    """
    if not os.path.exists(DEFAULT_DATA_FILE):
        return None, None, None, None, "Default dataset file not found on server."

    try:
        if DEFAULT_DATA_FILE.endswith(".csv"):
            df = pd.read_csv(DEFAULT_DATA_FILE)
        elif DEFAULT_DATA_FILE.endswith((".xlsx", ".xls")):
            df = pd.read_excel(DEFAULT_DATA_FILE)
        else:
            return None, None, None, None, "Default dataset must be CSV or Excel."
    except Exception as e:
        return None, None, None, None, f"Error reading default dataset: {e}"

    try:
        df_pred, out_path, charts, feedback = run_predictions_on_dataframe(df)
    except Exception as e:
        return None, None, None, None, str(e)

    table_html = df_pred.head(20).to_html(
        classes="table table-bordered table-striped table-sm",
        index=False,
    )
    return table_html, out_path, charts, feedback, None


@app.route("/", methods=["GET", "POST"])
def upload_file():
    result_table = None
    download_file = None
    charts = None
    feedback = None
    model_info = get_model_info()
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Please select a CSV or Excel file to upload."
            return render_template(
                "upload.html",
                error=error,
                charts=charts,
                table=result_table,
                download_link=download_file,
                feedback=feedback,
                model_info=model_info,
                train_table=TRAIN_TABLE_HTML,
            )

        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # Read CSV or Excel
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(path)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                error = "Only CSV (.csv) or Excel (.xlsx, .xls) files are allowed."
                return render_template(
                    "upload.html",
                    error=error,
                    charts=charts,
                    table=result_table,
                    download_link=download_file,
                    feedback=feedback,
                    model_info=model_info,
                    train_table=TRAIN_TABLE_HTML,
                )
        except Exception as e:
            error = f"Error reading file: {e}"
            return render_template(
                "upload.html",
                error=error,
                charts=charts,
                table=result_table,
                download_link=download_file,
                feedback=feedback,
                model_info=model_info,
                train_table=TRAIN_TABLE_HTML,
            )

        # Predict using helper
        try:
            df_pred, out_path, charts, feedback = run_predictions_on_dataframe(df)
        except Exception as e:
            error = str(e)
            return render_template(
                "upload.html",
                error=error,
                charts=None,
                table=None,
                download_link=None,
                feedback=None,
                model_info=model_info,
                train_table=TRAIN_TABLE_HTML,
            )

        result_table = df_pred.head(20).to_html(
            classes="table table-bordered table-striped table-sm",
            index=False,
        )
        download_file = out_path

        return render_template(
            "upload.html",
            table=result_table,
            download_link=download_file,
            charts=charts,
            feedback=feedback,
            model_info=model_info,
            error=error,
            train_table=TRAIN_TABLE_HTML,
        )

    # GET request -> show default dataset and its results if available
    result_table, download_file, charts, feedback, error_default = load_default_dataset_results()
    if error_default and not error:
        error = error_default

    return render_template(
        "upload.html",
        error=error,
        charts=charts,
        table=result_table,
        download_link=download_file,
        feedback=feedback,
        model_info=model_info,
        train_table=TRAIN_TABLE_HTML,
    )


@app.route("/download", methods=["GET"])
def download():
    filepath = request.args.get("file")
    return send_file(filepath, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
