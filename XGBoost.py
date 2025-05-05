import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# Load hasil preprocessing
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()  # Ensure y_train is 1D
y_test  = pd.read_csv('y_test.csv').values.ravel()    # Ensure y_test is 1D

# ---------- 1. XGBoost Classifier ----------
# Calculate the scale_pos_weight for handling class imbalance
scale_pos_weight = len(y_train) / sum(y_train == 1)

# Inisialisasi dan training
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluasi XGBoost
print("=== XGBoost Classification Report ===")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix XGBoost
disp_xgb = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb, cmap='Blues', normalize=None
)
disp_xgb.ax_.set_title('Confusion Matrix - XGBoost')
plt.savefig('confusion_matrix_xgb.png')
plt.close()

# Feature Importance XGBoost
i_importances = xgb.feature_importances_
feat_names = X_train.columns  # Assuming X_train is a DataFrame with columns
imp_xgb = pd.Series(i_importances, index=feat_names).sort_values()
plt.figure(figsize=(8,6))
imp_xgb.plot(kind='barh')
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_xgb.png')
plt.close()
