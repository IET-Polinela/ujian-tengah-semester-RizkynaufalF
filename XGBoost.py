import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load hasil preprocessing dan SMOTE
X_train_smote = pd.read_csv('X_train_smote.csv')
X_test  = pd.read_csv('X_test.csv')
y_train_smote = pd.read_csv('y_train_smote.csv').values.ravel()  # Ensure y_train is 1D
y_test  = pd.read_csv('y_test.csv').values.ravel()    # Ensure y_test is 1D

# Inisialisasi dan training XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb.predict(X_test)

# Evaluasi XGBoost
print("=== XGBoost Classification Report setelah SMOTE ===")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix XGBoost
disp_xgb = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb, cmap='Blues', normalize=None
)
disp_xgb.ax_.set_title('Confusion Matrix - XGBoost setelah SMOTE')
plt.savefig('confusion_matrix_xgb_smote.png')
plt.close()

# Feature Importance XGBoost
i_importances = xgb.feature_importances_
feat_names = X_train_smote.columns  # Assuming X_train is a DataFrame with columns
imp_xgb = pd.Series(i_importances, index=feat_names).sort_values()
plt.figure(figsize=(8,6))
imp_xgb.plot(kind='barh')
plt.title('Feature Importance - XGBoost setelah SMOTE')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_xgb_smote.png')
plt.close()
