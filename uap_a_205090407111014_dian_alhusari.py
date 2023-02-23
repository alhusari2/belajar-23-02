#import package yang digunakan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

"""2. membuka file """

ISPU = pd.read_csv('ISPUProvinsiDKIJakarta2021.csv')
ISPU

ISPU.info()

ISPU.isnull().sum()

"""3. melakukan pengisian data yang kosong dengan nilai rata-rata"""

#pengisian data yang kosong dengan nilai rata-rata
rata=ISPU['pm25'].mean()
ISPU['pm25']=ISPU['pm25'].fillna(rata)
ISPU.isnull().sum()

ISPU['critical'].unique()

ISPU['location'].unique()

ISPU['categori'].unique()


"""4. seleksi fitur yang  tidak signifikan

  karena tanggal hanya menampilkan tanggal data di ambil, critical menampilkan parameter yang hasil pengukurannya tinggi, max menampilkan nilai ukur yang tinggi, dan location tempat pengukuran. sehingga tidak berpengaruh pada faktor penyebab Indeks Standar Pencemar Udara nya. 
"""

ISPU = ISPU.drop(['tanggal','critical','max','location'],axis=1)
ISPU


"""5. transformasi data kategorikal menjadi tipe data numerik."""

#transformasi data kategorikal menjadi tipe data numerik.
labelencoder = LabelEncoder()
ISPU['categori'] = labelencoder.fit_transform(ISPU['categori'])
ISPU
#pada kolom categori : 0(baik),1(sedang),2(tidak sehat)

"""6. box-plot untuk memperlihatkan adanya outlier atau tidak"""

print('\nPersebaran data sebelum ditangani Outlier: ')
print(ISPU[['pm10','pm25','so2','co','o3','no2','categori']].describe())
# Creating Box Plot
import matplotlib.pyplot as plt
import seaborn as sns
# Masukkan variable
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['pm10'])
plt.show()
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['pm25'])
plt.show()
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['so2'])
plt.show()
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['co'])
plt.show()
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['o3'])
plt.show()
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['no2'])
plt.show()
plt.figure() # untuk membuat figure baru
sns.boxplot(x=ISPU['categori'])
plt.show()

"""7. mengatasi outlier"""

# categori outlier with IQR
Q1 = (ISPU[['pm10','pm25','so2','co','o3','no2']]).quantile(0.25)
Q3 = (ISPU[['pm10','pm25','so2','co','o3','no2']]).quantile(0.75)
IQR = Q3 - Q1
maximum = Q3 + (1.5*IQR)
print('Nilai Maximum dari masing-masing Variable adalah: ')
print(maximum)
minimum = Q1 - (1.5*IQR)
print('\nNilai Minimum dari masing-masing Variable adalah: ')
print(minimum)
more_than = (ISPU > maximum)
lower_than = (ISPU < minimum)
ISPU = ISPU.mask(more_than, maximum, axis=1)
ISPU = ISPU.mask(lower_than, minimum, axis=1)
print('\nPersebaran data setelah ditangani Outlier: ')
print(ISPU[['pm10','pm25','so2','co','o3','no2','categori']].describe())

ISPU

"""8. normalisasi data dengan menggunakan min-max normalization"""

from sklearn.preprocessing import MinMaxScaler 
# memisahkan array menjadi komponen input dan output 
X = ISPU.iloc[:, :-1].values
y = ISPU.iloc[:, 6].values
scaler = MinMaxScaler(feature_range=(0, 1)) 
rescaledX = scaler.fit_transform(X) 
# ringkasan data hasil transformasi 
np.set_printoptions(precision=3) 
print(rescaledX[0:6,:])

"""9. preprocessing diperlukan karena untuk mengantisipasi data missing, data noisy dan juga outlier2 sehinggal preprocessing sangat di perlukan agar data yang di pakai bener2 baik dan benar agar memudahkan proses mengklasifikasi

10. membagi 80% data training dan 20% data testing
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2) 
# test size = 0.2 -> 20% data testing yang di gunakan

"""11. klasifikasi metode KNN"""

from sklearn.neighbors import KNeighborsClassifier#import package KNN
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  pred_i = knn.predict(X_test)
  error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed',marker='o',

markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train) #Menginput data training pada fungsi klasifikasi.
y_predknn = classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_predknn)
ConfusionMatrixDisplay(cm,display_labels=classifier.classes_).plot()
#0(baik),1(sedang),2(tidak sehat)

print(classification_report(y_test, y_predknn))

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_predknn)))

"""k-fold Cross Validation metode KNN"""

knn_cv = KNeighborsClassifier(n_neighbors=11)
cv_scores = cross_val_score(knn_cv, X, y, cv=10)
print('Cross-validation scores:{}'.format(cv_scores))
#print('cv_scores mean:{}'.format(np.mean(cv_scores))

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(cv_scores.mean()))

"""- Menggunakan validasi silang rata-rata di metode KNN, kita dapat menyimpulkan bahwa kita mengharapkan model menjadi sekitar 92,6% akurat rata-rata.

- Jika kita melihat semua 10 skor yang dihasilkan oleh validasi silang 10 kali lipat, kita juga dapat menyimpulkan bahwa ada perbedaan yang relatif ada yang kecil ada juga yang sedang dalam akurasi antar lipatan, mulai dari akurasi 86,5% hingga akurasi 100% bisa di katakan perubahannya sedang. Jadi, kita dapat menyimpulkan bahwa model bisa juga bergantung pada lipatan tertentu yang digunakan untuk pelatihan.

- Akurasi model asli kami adalah 0,9726, tetapi akurasi validasi silang rata-rata adalah 0,9260. Tidak ada peningkatan.

12. metode Naïve Bayes
"""

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predNB = gnb.predict(X_test)
from sklearn import metrics
b=metrics.accuracy_score(y_test, y_predNB)
print("Accuracy:",b, '\n')

print(confusion_matrix(y_test, y_predNB))
#0(baik),1(sedang),2(tidak sehat)

print(classification_report(y_test, y_predNB))

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_predNB)))

"""k-fold Cross Validation metode Naive Bayes"""

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))

"""- Menggunakan validasi silang rata-rata di metode Naive Bayes, kita dapat menyimpulkan bahwa kita mengharapkan model menjadi sekitar 89,03% akurat rata-rata.

- Jika kita melihat semua 10 skor yang dihasilkan oleh validasi silang 10 kali lipat, kita juga dapat menyimpulkan bahwa ada perbedaan yang relatif kecil  dalam akurasi antar lipatan tetapi saat akurasi 75,9% itu selanjutnya perbedaannya lumayan sedang ke 89,7%. Jadi, kita dapat menyimpulkan bahwa model bisa juga bergantung pada lipatan tertentu yang digunakan untuk pelatihan.

- Akurasi model asli kami adalah 0,9041, tetapi akurasi validasi silang rata-rata adalah 0,8903. Tidak ada peningkatan.

# DESKRIPSI
di sini saya akan membangun suatu model menggunakan klasifikasi metode KNN dan metode Naive Bayes untuk mempridiksi apakah suatu lokasi memiliki udara yang baik, sedang, atau buruk/tidak sehat. data di ambil dari kaggle dengan jumlah data sebanyak 365 data

model yang dibuat menghasilkan kinerja yang sangat baik yang ditunjukkan dengan akurasi model dari metode KNN yang ditemukan sebesar 0,9726 dan dari metode Naive bayes didapat sebesar 0,9041 yang mana berarti akurasi dari metode KNN lebih baik di bandingkan metode naive bayes.

untuk akurasi rata2 di dapatkan dari metode KNN ialah 92,6% dan dari metode naive bayes adalah 89,03%.

Maka dapat disimpulkan bahwa dengan metode K-Nearest Neighbor pada data Indeks Standar Pencemar Udara di Provinsi DKI Jakarta 2021 lebih efektif dibandingkan dengan metode naive bayes karena akurasi yang di dapat lebih tinggi. hal ini di sebabkan data yang di testing banyak sehingga metode KNN lebih unggul. akan tetapi Metode Naive bayes dapat menghasilkan akurasi yang baik walaupun memiliki
data latih yang sedikit.
"""