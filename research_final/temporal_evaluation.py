import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("student_learning_interaction_dataset.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Target
target = "success_label"

# Columns to exclude
drop_columns = [
    "student_id",
    "session_id",
    "timestamp",
    "success_label",
    "next_module_prediction"
]

# Separate features
X = df.drop(columns=drop_columns)
y = df[target]

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = [
    c for c in X.columns
    if c not in categorical_features
]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# =====================================================
# TEMPORAL DATASET SHIFT EVALUATION
# =====================================================

print("\nSorting dataset by timestamp...")

df = df.sort_values("timestamp").reset_index(drop=True)

split_point = int(len(df) * 0.70)

train_df = df.iloc[:split_point]
future_df = df.iloc[split_point:]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["success_label"]

X_future = future_df.drop(columns=drop_columns)
y_future = future_df["success_label"]

print("\nTraining samples:", len(X_train))
print("Future test samples:", len(X_future))

# Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("\nTesting on future data...")

future_predictions = model.predict(X_future)

print("\n==========================================")
print("TEMPORAL DATASET SHIFT RESULTS")
print("==========================================")

print(
    classification_report(
        y_future,
        future_predictions
    )
)
# ============================================================
# TEMPORAL EVALUATION GRAPHS - Matplotlib
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate classification report as dictionary for plotting
report = classification_report(
    y_future,
    future_predictions,
    output_dict=True
)

# 1. Temporal Classification Metrics
class_names = [str(c) for c in sorted(y_future.unique())]
precision_values = [report[c]["precision"] for c in class_names]
recall_values = [report[c]["recall"] for c in class_names]
f1_values = [report[c]["f1-score"] for c in class_names]

x = np.arange(len(class_names)) if 'np' in globals() else range(len(class_names))

plt.figure(figsize=(8, 5))
width = 0.25

plt.bar([i - width for i in x], precision_values, width=width, label="Precision")
plt.bar(x, recall_values, width=width, label="Recall")
plt.bar([i + width for i in x], f1_values, width=width, label="F1-score")

plt.title("Temporal Dataset Shift: Classification Performance")
plt.xlabel("Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(list(x), class_names)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Confusion Matrix for Future Data
cm = confusion_matrix(y_future, future_predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Temporal Dataset Shift")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(class_names)), class_names)
plt.yticks(range(len(class_names)), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()
