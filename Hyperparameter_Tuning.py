from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pandas as pd

# Memuat data yang telah diproses sebelumnya
X_train_smote = pd.read_csv('X_train_smote.csv')
y_train_smote = pd.read_csv('y_train_smote.csv').values.ravel()  # Pastikan y_train adalah 1D
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()  # Pastikan y_test adalah 1D

# Parameter yang ingin diuji
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Inisialisasi model XGBoost tanpa use_label_encoder
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# GridSearchCV untuk mencari kombinasi parameter terbaik
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_smote, y_train_smote)

# Menampilkan hasil pencarian grid terbaik
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Evaluasi model terbaik
y_pred_best_xgb = grid_search.best_estimator_.predict(X_test)

# Evaluasi XGBoost dengan parameter terbaik
from sklearn.metrics import classification_report
print("=== XGBoost Classification Report dengan Hyperparameter Tuning ===")
print(classification_report(y_test, y_pred_best_xgb))

# Confusion Matrix XGBoost
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp_xgb_best = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best_xgb, cmap='Blues', normalize=None
)
disp_xgb_best.ax_.set_title('Confusion Matrix - XGBoost setelah Hyperparameter Tuning')
plt.savefig('confusion_matrix_xgb_best.png')
plt.close()

# Feature Importance XGBoost terbaik
i_importances_best = grid_search.best_estimator_.feature_importances_
feat_names = X_train_smote.columns  # Assuming X_train is a DataFrame with columns
imp_xgb_best = pd.Series(i_importances_best, index=feat_names).sort_values()
plt.figure(figsize=(8,6))
imp_xgb_best.plot(kind='barh')
plt.title('Feature Importance - XGBoost setelah Hyperparameter Tuning')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_xgb_best.png')
plt.close()
