"""
Supervised Machine Learning Regression Analysis
Dataset: cw1data.csv
With Image Embedding for Reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Configure matplotlib backend for saving images
import os
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Images will be saved in: {os.getcwd()}")

# Create output folder for images
output_folder = "ML_Results"
os.makedirs(output_folder, exist_ok=True)
print(f"All results will be saved in: {output_folder}/")

print("="*70)
print("SUPERVISED MACHINE LEARNING - REGRESSION ANALYSIS")
print("="*70)

# List to store all figure objects for PDF export
all_figures = []

# ============================================================================
# STEP 1: IMPORT DATASET
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv('cw1data.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Information:")
print(df.info())

print(f"\nStatistical Summary:")
print(df.describe())

print(f"\nMissing Values:")
print(df.isnull().sum())

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.dropna()
    print(f"Dataset shape after removing missing values: {df.shape}")

# ============================================================================
# STEP 2: DATA VISUALISATION
# ============================================================================
print("\n[STEP 2] Data Visualisation...")

# Automatically identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns found: {numeric_cols}")

# Check if we need to convert columns to numeric
if len(numeric_cols) == 0:
    print("\nNo numeric columns detected. Attempting to convert...")
    print(f"Current data types:\n{df.dtypes}\n")

    # Try to convert all columns to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted '{col}' to numeric")
        except:
            print(f"Could not convert '{col}' to numeric")

    # Re-identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns after conversion: {numeric_cols}")

    # Remove rows with NaN values created during conversion
    if df.isnull().sum().sum() > 0:
        print(f"\nRemoving rows with invalid values...")
        df = df.dropna()
        print(f"Dataset shape after cleaning: {df.shape}")

# If still no numeric columns, exit with error message
if len(numeric_cols) == 0:
    print("\n" + "="*70)
    print("ERROR: No numeric columns found in the dataset!")
    print("="*70)
    print("\nPlease check your CSV file. Make sure it contains numeric data.")
    print("Current columns and types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    exit()

# Correlation Heatmap
fig1 = plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap of Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_folder}/1_correlation_heatmap.png', dpi=300, bbox_inches='tight')
all_figures.append(fig1)
print("✓ Saved: 1_correlation_heatmap.png")
plt.show()

# Distribution plots for all numeric features
n_cols = len(numeric_cols)
n_rows = max(1, (n_cols + 2) // 3)
fig2, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows*4))

# Convert axes to a flat list for easier handling
if n_rows == 1 and n_cols <= 3:
    # Single row case
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = [axes]
else:
    # Multiple rows case
    axes_list = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [axes]

# Plot distributions
for idx, col in enumerate(numeric_cols):
    if idx < len(axes_list):
        data_to_plot = df[col].dropna().values
        axes_list[idx].hist(data_to_plot, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes_list[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes_list[idx].set_xlabel(col)
        axes_list[idx].set_ylabel('Frequency')

# Hide extra subplots
for idx in range(len(numeric_cols), len(axes_list)):
    axes_list[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{output_folder}/2_feature_distributions.png', dpi=300, bbox_inches='tight')
all_figures.append(fig2)
print("✓ Saved: 2_feature_distributions.png")
plt.show()

# Pairplot for key relationships (limited to first 5 features for clarity)
if len(numeric_cols) > 1:
    cols_to_plot = numeric_cols[:min(5, len(numeric_cols))]
    fig3 = sns.pairplot(df[cols_to_plot], diag_kind='kde', corner=True)
    fig3.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16, fontweight='bold')
    plt.savefig(f'{output_folder}/3_pairplot.png', dpi=300, bbox_inches='tight')
    all_figures.append(fig3.fig)
    print("✓ Saved: 3_pairplot.png")
    plt.show()

# ============================================================================
# STEP 3: DATA PREPARATION
# ============================================================================
print("\n[STEP 3] Data Preparation...")

# Show all available columns
print(f"\nAvailable columns: {list(df.columns)}")
print(f"Total columns: {len(df.columns)}")

# Identify target variable
# Method 1: Assume last column is target (most common)
target_column = df.columns[-1]
print(f"\n→ Auto-detected target variable (last column): '{target_column}'")

# Verify target is numeric
if target_column not in numeric_cols:
    print(f"\nWARNING: Target column '{target_column}' is not numeric!")
    print("Attempting to convert to numeric...")
    try:
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        df = df.dropna(subset=[target_column])
        print(f"✓ Converted successfully. Rows remaining: {df.shape[0]}")
    except:
        print("ERROR: Could not convert target to numeric. Please check your data.")
        exit()

print(f"\nTarget variable: {target_column}")
print(f"Target statistics:\n{df[target_column].describe()}")

# Define features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Remove non-numeric columns from X if any
X = X.select_dtypes(include=[np.number])

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures: {list(X.columns)}")

# Check if we have enough data
if X.shape[0] == 0 or X.shape[1] == 0:
    print("\n" + "="*70)
    print("ERROR: No valid data remaining after preprocessing!")
    print("="*70)
    print(f"Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    print("\nPossible reasons:")
    print("1. All data was removed during cleaning (too many missing values)")
    print("2. Target column contains all the data")
    print("3. No numeric features available")
    print(f"\nDataset info:")
    print(f"Total rows in original dataset: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Target column: {target_column}")
    print("\nPlease check your CSV file and ensure:")
    print("- It has multiple numeric columns")
    print("- The last column is your target variable")
    print("- There are not too many missing values")
    exit()

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature Scaling (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 4: MODEL BUILDING
# ============================================================================
print("\n[STEP 4] Building Multiple Regression Models...")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train models and store results
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

    print(f"{name} trained successfully!")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\n[STEP 5] Model Evaluation and Comparison...")

# Create comparison DataFrame
comparison_data = []
for name, res in results.items():
    comparison_data.append({
        'Model': name,
        'Train MSE': res['train_mse'],
        'Test MSE': res['test_mse'],
        'Train RMSE': res['train_rmse'],
        'Test RMSE': res['test_rmse'],
        'Train R²': res['train_r2'],
        'Test R²': res['test_r2'],
        'Train MAE': res['train_mae'],
        'Test MAE': res['test_mae']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(f'{output_folder}/model_comparison.csv', index=False)

# Visualization 1: R² Score Comparison
fig4, axes = plt.subplots(1, 2, figsize=(15, 6))

models_list = comparison_df['Model'].tolist()
train_r2 = comparison_df['Train R²'].tolist()
test_r2 = comparison_df['Test R²'].tolist()

x = np.arange(len(models_list))
width = 0.35

axes[0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='skyblue')
axes[0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='orange')
axes[0].set_xlabel('Models', fontweight='bold')
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('R² Score Comparison', fontweight='bold', fontsize=14)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_list, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Visualization 2: RMSE Comparison
train_rmse = comparison_df['Train RMSE'].tolist()
test_rmse = comparison_df['Test RMSE'].tolist()

axes[1].bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8, color='lightgreen')
axes[1].bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8, color='salmon')
axes[1].set_xlabel('Models', fontweight='bold')
axes[1].set_ylabel('RMSE', fontweight='bold')
axes[1].set_title('RMSE Comparison', fontweight='bold', fontsize=14)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_list, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/4_model_comparison.png', dpi=300, bbox_inches='tight')
all_figures.append(fig4)
print("✓ Saved: 4_model_comparison.png")
plt.show()

# Prediction vs Actual plots for each model
fig5, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

colors = ['blue', 'green', 'red', 'purple']
for idx, (name, res) in enumerate(results.items()):
    axes[idx].scatter(y_test, res['y_pred_test'], alpha=0.6, s=50, color=colors[idx])
    axes[idx].plot([y_test.min(), y_test.max()],
                   [y_test.min(), y_test.max()],
                   'r--', lw=2, label='Perfect Prediction')
    axes[idx].set_xlabel('Actual Values', fontweight='bold')
    axes[idx].set_ylabel('Predicted Values', fontweight='bold')
    axes[idx].set_title(f'{name}\nR² = {res["test_r2"]:.4f}, RMSE = {res["test_rmse"]:.4f}',
                       fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/5_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
all_figures.append(fig5)
print("✓ Saved: 5_prediction_vs_actual.png")
plt.show()

# ============================================================================
# OPTIONAL ENHANCEMENTS (BONUS)
# ============================================================================
print("\n[BONUS] Optional Enhancements...")

# 1. Feature Importance (for tree-based models)
print("\n1. Feature Importance Analysis:")
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

fig6 = plt.figure(figsize=(10, 6))
colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_bar)
plt.xlabel('Importance', fontweight='bold')
plt.ylabel('Features', fontweight='bold')
plt.title('Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_folder}/6_feature_importance.png', dpi=300, bbox_inches='tight')
all_figures.append(fig6)
print("✓ Saved: 6_feature_importance.png")
plt.show()

# 2. Cross-Validation
print("\n2. Cross-Validation Scores (5-fold):")
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring='r2', n_jobs=-1)
    cv_results[name] = cv_scores
    print(f"{name}: Mean R² = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 3. Residual Analysis for best model
best_model_name = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
print(f"\n3. Residual Analysis for Best Model: {best_model_name}")

best_pred = results[best_model_name]['y_pred_test']
residuals = y_test - best_pred

fig7, axes = plt.subplots(1, 2, figsize=(15, 5))

# Residual plot
axes[0].scatter(best_pred, residuals, alpha=0.6, color='purple')
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Values', fontweight='bold')
axes[0].set_ylabel('Residuals', fontweight='bold')
axes[0].set_title(f'Residual Plot - {best_model_name}', fontweight='bold', fontsize=14)
axes[0].grid(alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='teal')
axes[1].set_xlabel('Residuals', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Residual Distribution', fontweight='bold', fontsize=14)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/7_residual_analysis.png', dpi=300, bbox_inches='tight')
all_figures.append(fig7)
print("✓ Saved: 7_residual_analysis.png")
plt.show()

# ============================================================================
# EXPORT ALL IMAGES TO A SINGLE PDF (FOR EASY REPORT SUBMISSION)
# ============================================================================
print("\n[EXPORTING] Creating comprehensive PDF report...")

pdf_filename = f'{output_folder}/ML_Analysis_Complete_Report.pdf'
with PdfPages(pdf_filename) as pdf:
    for fig in all_figures:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"✓ PDF Report created: {pdf_filename}")
print("  → All images combined in a single PDF file!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nBest Model: {best_model_name}")
print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")

print(f"\nAll visualizations and results have been saved in '{output_folder}/':")
print("  - 1_correlation_heatmap.png")
print("  - 2_feature_distributions.png")
print("  - 3_pairplot.png")
print("  - 4_model_comparison.png")
print("  - 5_prediction_vs_actual.png")
print("  - 6_feature_importance.png")
print("  - 7_residual_analysis.png")
print("  - model_comparison.csv")
print("  - ML_Analysis_Complete_Report.pdf ← ALL IMAGES IN ONE FILE!")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

# ============================================================================
# HOW TO USE IMAGES IN YOUR REPORT/DOCUMENT
# ============================================================================
print("\n" + "="*70)
print("📊 HOW TO ADD IMAGES TO YOUR REPORT")
print("="*70)
print("\n1. FOR WORD DOCUMENTS:")
print("   - Open MS Word")
print("   - Go to Insert → Pictures")
print(f"   - Navigate to '{output_folder}/' folder")
print("   - Select any PNG image and insert")
print("\n2. FOR PDF SUBMISSION:")
print(f"   - Use the generated PDF: {pdf_filename}")
print("   - All charts are already included!")
print("\n3. FOR POWERPOINT:")
print("   - Drag and drop PNG files directly into slides")
print("\n4. FOR LATEX/OVERLEAF:")
print("   - Upload PNG files to your project")
print("   - Use: \\includegraphics{1_correlation_heatmap.png}")

# ============================================================================
# Q&A SECTION (COMPULSORY)
# ============================================================================
print("\n" + "="*70)
print("Q&A SECTION")
print("="*70)

print("\n--- QUESTION 1: Python Code Question ---")
print("Q: What does the train_test_split() function do and why is random_state=42 used?")
print("\nA: The train_test_split() function from sklearn.model_selection splits the dataset")
print("   into training and testing sets. In this code, we use test_size=0.2, meaning 80%")
print("   of data is used for training and 20% for testing. The random_state=42 parameter")
print("   ensures reproducibility - it sets a seed for the random number generator so that")
print("   the same split is obtained every time the code runs. This is crucial for:")
print("   - Consistent results across different runs")
print("   - Comparing model performance fairly")
print("   - Debugging and sharing reproducible results")
print("   The value 42 is arbitrary; any integer would work the same way.")

print("\n--- QUESTION 2: Lecture/Seminar Concepts Question ---")
print("Q: What is overfitting and how can we detect it in our model evaluation?")
print("\nA: Overfitting occurs when a model learns the training data too well, including")
print("   noise and outliers, resulting in poor generalization to new, unseen data.")
print("\n   Detection methods used in this code:")
print("   1. TRAIN-TEST GAP: Compare training vs testing metrics. A large gap (e.g.,")
print("      Train R²=0.99 but Test R²=0.65) indicates overfitting.")
print("   2. CROSS-VALIDATION: We use 5-fold CV which tests the model on different data")
print("      subsets. High variance in CV scores suggests overfitting.")
print("   3. LEARNING CURVES: High training accuracy with low test accuracy indicates")
print("      the model has memorized rather than learned patterns.")
print("\n   Prevention techniques:")
print("   - Use regularization (L1/L2)")
print("   - Limit model complexity (e.g., max_depth in Decision Trees)")
print("   - Use more training data")
print("   - Apply feature selection")
print("   - Use ensemble methods like Random Forest")

print("\n" + "="*70)