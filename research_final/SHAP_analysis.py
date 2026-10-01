# ============================================================
# SHAP Explainable AI Analysis
# Research:
# A Closed-Loop Hybrid Machine Learning and Generative AI
# Framework for Adaptive Personalized Learning through
# Learner State Prediction and Intelligent Intervention Selection
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from Algorithms import trained_models, X_test, y_test


# ============================================================
# 1. Get trained Random Forest model
# ============================================================

print("\nLoading trained Random Forest model...")

rf_pipeline = trained_models["Random Forest"]

rf_model = rf_pipeline.named_steps["model"]
preprocessor = rf_pipeline.named_steps["preprocessor"]

print("Random Forest model loaded successfully.")


# ============================================================
# 2. Transform test data
# ============================================================

print("\nTransforming test data...")

X_test_transformed = preprocessor.transform(X_test)

# Convert sparse matrix to NumPy array if necessary
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()


# ============================================================
# 3. Get feature names
# ============================================================

feature_names = preprocessor.get_feature_names_out()

X_shap = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

print("Number of features:", len(feature_names))
print("Test samples:", len(X_shap))


# ============================================================
# 4. Create SHAP Tree Explainer
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_shap)

print("SHAP values calculated successfully.")


# ============================================================
# Handle SHAP output shape
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions:
    # [class_0_values, class_1_values]
    shap_values_plot = shap_values[1]

else:

    # Newer SHAP versions may return:
    # (samples, features, classes)

    if shap_values.ndim == 3:

        # Select positive class (class 1)
        shap_values_plot = shap_values[:, :, 1]

    else:

        # Already (samples, features)
        shap_values_plot = shap_values


print("Original SHAP shape:", np.shape(shap_values))
print("SHAP shape used for analysis:", np.shape(shap_values_plot))


# ============================================================
# 6. Create output folder
# ============================================================

os.makedirs("shap_results", exist_ok=True)


# ============================================================
# 7. SHAP Summary Plot
# ============================================================

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values_plot,
    X_shap,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_results/shap_summary.png")


# ============================================================
# 8. SHAP Bar Plot
# ============================================================

print("\nGenerating SHAP feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values_plot,
    X_shap,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_results/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_results/shap_bar.png")


# ============================================================
# 9. Calculate Global SHAP Importance
# ============================================================

print("\nCalculating global SHAP importance...")

print("\nCalculating global SHAP importance...")

# Convert SHAP values to NumPy array
shap_array = np.asarray(shap_values_plot)

print("Original SHAP array shape:", shap_array.shape)
print("Feature names:", len(feature_names))

# Handle SHAP output dimensions
if shap_array.ndim == 3:
    # Shape: (samples, features, classes)
    shap_array = shap_array[:, :, 1]

elif shap_array.ndim != 2:
    raise ValueError(
        f"Unexpected SHAP array shape: {shap_array.shape}"
    )

print("SHAP array used for importance:", shap_array.shape)

# Calculate mean absolute SHAP value for each feature
mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

# Force to 1-dimensional
mean_abs_shap = np.asarray(mean_abs_shap).flatten()

print("Mean absolute SHAP shape:", mean_abs_shap.shape)

# Check that both arrays have the same length
if len(feature_names) != len(mean_abs_shap):
    raise ValueError(
        f"Mismatch: {len(feature_names)} feature names "
        f"but {len(mean_abs_shap)} SHAP values"
    )

# Create importance table
shap_importance = pd.DataFrame({
    "Feature": list(feature_names),
    "Mean_Absolute_SHAP": mean_abs_shap.tolist()
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

shap_importance.to_csv(
    "shap_results/shap_global_importance.csv",
    index=False
)

print("\n==========================================")
print("TOP 20 SHAP FEATURES")
print("==========================================")
print(shap_importance.head(20).to_string(index=False))
shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

# Save CSV
shap_importance.to_csv(
    "shap_results/shap_global_importance.csv",
    index=False
)

print(
    "Saved: shap_results/shap_global_importance.csv"
)


# ============================================================
# 10. Print Top 20 Features
# ============================================================

print("\n==========================================")
print("TOP 20 SHAP FEATURES")
print("==========================================")

print(
    shap_importance.head(20).to_string(index=False)
)


# ============================================================
# 11. Individual Learner Explanation
# ============================================================

sample_index = 0

print("\n==========================================")
print("INDIVIDUAL LEARNER EXPLANATION")
print("==========================================")

sample = X_shap.iloc[[sample_index]]

prediction = rf_pipeline.predict(
    X_test.iloc[[sample_index]]
)[0]

print("Predicted learner outcome:", prediction)

print("\nImportant factors for this learner:")

sample_shap = shap_values_plot[sample_index]

individual_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": sample_shap
})

individual_explanation["Absolute_SHAP"] = (
    individual_explanation["SHAP_Value"].abs()
)

individual_explanation = individual_explanation.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

print(
    individual_explanation.head(10).to_string(
        index=False
    )
)

individual_explanation.to_csv(
    "shap_results/individual_learner_explanation.csv",
    index=False
)


# ============================================================
# 12. Save complete SHAP results
# ============================================================

print("\nSaving complete SHAP matrix...")

shap_matrix = pd.DataFrame(
    shap_values_plot,
    columns=feature_names
)

shap_matrix.to_csv(
    "shap_results/shap_values_all_test_samples.csv",
    index=False
)

print(
    "Saved: shap_results/shap_values_all_test_samples.csv"
)


# ============================================================
# 13. Final message
# ============================================================

print("\n==========================================")
print("SHAP ANALYSIS COMPLETED")
print("==========================================")

print("\nGenerated files:")

print("1. shap_results/shap_summary.png")
print("2. shap_results/shap_bar.png")
print("3. shap_results/shap_global_importance.csv")
print("4. shap_results/individual_learner_explanation.csv")
print("5. shap_results/shap_values_all_test_samples.csv")
# ============================================================
# 14. Display SHAP graphs using plt.show()
# ============================================================

# SHAP Summary Plot - display interactively
plt.figure()
shap.summary_plot(
    shap_values_plot,
    X_shap,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.show()

# SHAP Bar Plot - display interactively
plt.figure()
shap.summary_plot(
    shap_values_plot,
    X_shap,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.tight_layout()
plt.show()
