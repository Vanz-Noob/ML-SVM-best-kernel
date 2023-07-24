import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import model_selection, preprocessing, utils

from pandas.plotting import scatter_matrix

# Membaca file CSV ke dalam DataFrame
df = pd.read_csv('DATASET.csv', encoding='utf-8')

# Menggabungkan isi dari beberapa kolom menjadi satu teks
df['teks'] = df[['satu', 'dua', 'tiga', 'empat', 'lima']].astype(str).apply(lambda x: ' '.join(x), axis=1)

# Membuat objek CountVectorizer dan TfidfTransformer
cv = CountVectorizer()
tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)

#tf-idf
X = tfidf.fit_transform(cv.fit_transform(df["teks"])).toarray()
y = df['label'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.05, random_state= 0)
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('Y_train: ', Y_train.shape)
print('Y_test: ', Y_test.shape)

#Import svm model
from sklearn import svm

svm = SVC(decision_function_shape='ovo')
hyperparameters = {'kernel':['rbf','linear','poly','sigmoid'], 'C':[0.5,0.75,1,10], 'gamma':[0.001,0.01,0.5,1,'scale','auto']}
svm_tuned = GridSearchCV(svm,hyperparameters,cv=10)
svm_tuned.fit(X_train, Y_train)

#Train the model using the training sets
svm.fit(X_train, Y_train)

#mendapatkan nilai bias
bias = svm.intercept_

#Predict the response for test dataset
y_pred = svm.predict(X_test)

#Evaluasi Hasil Prediksi
print(accuracy_score(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
print("Nilai b:", bias)
print(" ")
print('Kernel Using  :',svm_tuned.best_estimator_.kernel)
print('Best C       :',svm_tuned.best_estimator_.C)
print('Best Gamma   :',svm_tuned.best_estimator_.gamma)
print('Best Score   :',svm_tuned.best_score_)