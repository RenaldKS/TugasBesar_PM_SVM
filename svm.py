import pandas as pd
import numpy as np
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

selected_features = X.columns[selector.get_support()] # Menampilkan fitur yang dipilih berdasarkan ANOVA, selector.get_support() 
print("=====================================") #mengembalikan array boolean yang menunjukkan fitur yang dipilih
print("Fitur yang dipilih:")
print(selected_features)

# b. Praproses
# Memisahkan data menjadi data latih dan data uji sesuai dengan rasio yang diinginkan
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42) #"test_size" dapat diubah sesuai kebutuhan 0.2 = 20% data uji

# Mengubah tipe data 'BMI' menjadi numerik dan menangani missing values
X_train[:, 1] = pd.to_numeric(X_train[:, 1], errors='coerce')
mean_bmi = np.nanmean(X_train[:, 1])
X_train[:, 1][np.isnan(X_train[:, 1])] = mean_bmi

# Z-score normalization for 'BMI' in the training set
X_train[:, 1] = (X_train[:, 1] - np.mean(X_train[:, 1])) / np.std(X_train[:, 1])

# Assuming similar normalization for 'BMI' in the test set
X_test[:, 1] = pd.to_numeric(X_test[:, 1], errors='coerce')
X_test[:, 1][np.isnan(X_test[:, 1])] = mean_bmi
X_test[:, 1] = (X_test[:, 1] - np.mean(X_train[:, 1])) / np.std(X_train[:, 1])

# Print the normalized 'BMI' values in the training and test sets

# print('Normalized BMI in the training set:')
# print(X_train[:, 1])

# print('Normalized BMI in the test set:')
# print(X_test[:, 1])

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c. Klasifikasi menggunakan SVM
svm_classifier = SVC(kernel='rbf')  # Bisa menggunakan kernel lain sesuai kebutuhan (e.g., 'linear', 'poly', 'sigmoid') rbf yang disarankan
svm_classifier.fit(X_train_scaled, y_train)
y_pred = svm_classifier.predict(X_test_scaled)

# d. Hasil Pengujian
# Akurasi Model
accuracy = accuracy_score(y_test, y_pred)
print('=====================================')
print ("Accuracy Model :")
print(f'Akurasi: {accuracy}')

# Classification Report
print('=====================================')
print('Laporan Klasifikasi:')
print(classification_report(y_test, y_pred))
print('Presisi : mengukur ketepatan model dalam mengklasifikasikan data positif')
print('Recall : mengukur ketepatan model dalam menemukan kembali semua data positif')
print('F1-Score : mengukur keseimbangan antara presisi dan recall')
print('Support : jumlah kemunculan setiap kelas')
print('Macro Avg : rata-rata presisi, recall, dan f1-score')
print('Weighted Avg : rata-rata presisi, recall, dan f1-score dengan bobot masing-masing kelas')
print('Micro Avg : rata-rata presisi, recall, dan f1-score dengan menghitung jumlah true positive, false negative, dan false positive')
print('Accuracy : mengukur akurasi dari model')
print('=====================================')

# Confusion Matrix
print('Confusion Matrix:')
print ('TN  FP')
print ('FN  TP')
print(confusion_matrix(y_test, y_pred))
print('=====================================')
print('TN : True Negative')
print('FP : False Positive')
print('FN : False Negative')
print('TP : True Positive')
print('=====================================')

# Menghitung skor kebenaran untuk beberapa sampel pada dataset uji
decision_values = svm_classifier.decision_function(X_test_scaled[:5])
# Cetak hasil skor kebenaran
print('Decision Function Values:')
print(decision_values)
print('=====================================')
# Support Vectors
# Mendapatkan vektor pendukung dari model SVM
support_vectors = svm_classifier.support_vectors_
# Cetak jumlah dan nilai vektor pendukung
print('Jumlah Support Vectors:', len(support_vectors))
print('Support Vectors: adalah titik-titik data dari kelas yang memiliki kontribusi signifikan terhadap pembentukan batas keputusan (decision boundary) dalam model Support Vector Machine (SVM)')
print(support_vectors)
print('=====================================')

