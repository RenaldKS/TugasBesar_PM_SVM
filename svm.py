import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Membaca dataset
data = pd.read_csv('diabetes_012_health_indicators_BRFSS2021.csv')

# Memisahkan fitur dan label
X = data.drop('Diabetes_012', axis=1)  # Ganti 'target_column_name' dengan nama kolom target
y = data['Stroke']

# a. Seleksi Fitur
# Menggunakan ANOVA untuk seleksi fitur
selector = SelectKBest(f_classif, k=5)  # Ganti 'k' dengan jumlah fitur yang ingin dipilih
X_selected = selector.fit_transform(X, y)

# b. Praproses
# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c. Klasifikasi menggunakan SVM
svm_classifier = SVC(kernel='linear')  # Bisa menggunakan kernel lain sesuai kebutuhan
svm_classifier.fit(X_train_scaled, y_train)
y_pred = svm_classifier.predict(X_test_scaled)

# d. Hasil Pengujian
# Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
