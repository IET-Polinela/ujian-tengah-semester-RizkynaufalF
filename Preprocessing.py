import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.drop(['id'], axis=1)

# Perbaikan warning: isi nilai null di 'bmi' tanpa inplace
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Encode fitur kategorikal
le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col])

# Split fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print info hasil preprocessing
print("Preprocessing selesai.")
print("Jumlah data latih:", X_train.shape[0])
print("Jumlah data uji:", X_test.shape[0])
print("Contoh data fitur:\n", X_train.head())

# Simpan hasil preprocessing ke dalam CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)
