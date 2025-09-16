import numpy as np
import warnings
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = [
    ("Logistic Regression", LogisticRegression(C=1, solver='liblinear')),
    ("Random Forest", RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)),
    ("SVM", SVC(kernel="linear", probability=True, random_state=42))
]

reports = []
predictions = {}

# Train and evaluate models
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report["accuracy"] = acc
    reports.append(report)
    predictions[model_name] = y_pred

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris_Model_Comparison")

# Log to MLflow
for i, (model_name, model) in enumerate(models):
    report = reports[i]
    y_pred = predictions[model_name]

    with mlflow.start_run(run_name=model_name):
        # Log model name
        mlflow.log_param("model", model_name)

        # Log metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        # Log per-class recall
        for cls in iris.target_names:
            cls_idx = list(iris.target_names).index(cls)
            mlflow.log_metric(f"recall_{cls}", report[str(cls_idx)]["recall"])

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Model performance bar plot
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        values = [report["accuracy"], report["macro avg"]["precision"],
                  report["macro avg"]["recall"], report["macro avg"]["f1-score"]]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=metrics, y=values, palette="viridis")
        plt.title(f"Performance Metrics - {model_name}")
        plt.ylim(0, 1)
        for idx, v in enumerate(values):
            plt.text(idx, v + 0.02, f"{v:.2f}", ha='center')
        perf_path = f"performance_{model_name.replace(' ', '_')}.png"
        plt.savefig(perf_path)
        plt.close()
        mlflow.log_artifact(perf_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")
