import numpy as np
import warnings
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# ---------------- Load dataset ----------------
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Define models ----------------
models = [
    ("Logistic Regression", LogisticRegression(C=1, solver='liblinear')),
    ("Random Forest", RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)),
    ("SVM", SVC(kernel="linear", probability=True, random_state=42))
]

reports = []

# Train and evaluate models
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report["accuracy"] = acc  # add accuracy manually
    reports.append(report)

# ---------------- MLflow Setup ----------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris_Model_Comparison")

# ---------------- Log to MLflow ----------------
for i, (model_name, model) in enumerate(models):
    report = reports[i]

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
            mlflow.log_metric(f"recall_{cls}", report[str(list(iris.target_names).index(cls))]["recall"])

        # Log model
        mlflow.sklearn.log_model(model, "model")
