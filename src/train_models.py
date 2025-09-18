import os
import numpy as np
import warnings
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = [
    ("Logistic Regression", LogisticRegression(C=1, solver='liblinear')),
    ("Random Forest", RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)),
    ("SVM", SVC(kernel="linear", probability=True, random_state=42))
]

results = []

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Model_Comparison")

for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model", model_name)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        for cls_idx, cls_name in enumerate(iris.target_names):
            mlflow.log_metric(f"recall_{cls_name}", report[str(cls_idx)]["recall"])

        cm = confusion_matrix(y_test, y_pred)
        cm_path = f"results/confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        perf_path = f"results/performance_{model_name.replace(' ', '_')}.png"
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        values = [acc, report["macro avg"]["precision"], report["macro avg"]["recall"], report["macro avg"]["f1-score"]]
        plt.bar(metrics, values)
        plt.title(f"Performance of {model_name}")
        plt.ylim(0, 1)
        plt.savefig(perf_path)
        plt.close()
        mlflow.log_artifact(perf_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({
            "model_name": model_name,
            "run_id": run.info.run_id,
            "accuracy": acc,
            "f1": report["macro avg"]["f1-score"],
            "model_obj": model
        })

results_df = pd.DataFrame(results)
best_row = results_df.sort_values(by="accuracy", ascending=False).iloc[0]

best_model_name = best_row["model_name"]
best_model = best_row["model_obj"]
best_run_id = best_row["run_id"]

print(f"\nBest Model: {best_model_name} (Accuracy={best_row['accuracy']:.3f})")

model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "IrisBestModel")

print("\nBest model registered in MLflow Model Registry as 'IrisBestModel'")
