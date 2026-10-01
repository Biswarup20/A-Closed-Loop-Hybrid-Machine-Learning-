import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("student_learning_interaction_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# 2. Define target
# -----------------------------

target = "success_label"

# Remove identifiers and variables not used as predictors
drop_columns = [
    "student_id",
    "session_id",
    "timestamp",
    "success_label",
    "next_module_prediction"
]

X = df.drop(columns=drop_columns)
y = df[target]

# -----------------------------
# 3. Detect feature types
# -----------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = [
    c for c in X.columns
    if c not in categorical_features
]

# -----------------------------
# 4. Preprocessing
# -----------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# -----------------------------
# 5. Models
# -----------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

    "SVM":
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
}

# -----------------------------
# 6. Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

results = []

trained_models = {}

# -----------------------------
# 7. Train and evaluate
# -----------------------------

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions
        ),
        "Recall": recall_score(
            y_test, predictions
        ),
        "F1": f1_score(
            y_test, predictions
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        )
    })

    trained_models[name] = pipeline

results_df = pd.DataFrame(results)

print(results_df)
# ============================================================
# 8. Research Paper Graphs - Matplotlib
# ============================================================

import matplotlib.pyplot as plt

# Display complete results table
print("\n==========================================")
print("MODEL PERFORMANCE RESULTS")
print("==========================================")
print(results_df.to_string(index=False))

# 8.1 Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8.2 Balanced Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Balanced Accuracy"])
plt.title("Balanced Accuracy Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("Balanced Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8.3 Precision Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Precision"])
plt.title("Precision Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8.4 Recall Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Recall"])
plt.title("Recall Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8.5 F1 Score Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1"])
plt.title("F1 Score Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8.6 ROC-AUC Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["ROC-AUC"])
plt.title("ROC-AUC Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8.7 Combined Performance Comparison
metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Overall Performance Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
