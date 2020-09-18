#!/usr/bin/env python
# coding: utf-8

# check sklearn version
import sklearn
print(sklearn.__version__)

# load data
import pandas as pd
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
df = pd.read_csv('labelData.csv',header=None)
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

# concatenate alternating columns with pandas
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# concatenate alternating columns with pandas
df1 = df.iloc[:,:125]
df1
df2 = df.iloc[:,125:250]
df2

df_new = pd.DataFrame()
for i in range(len(df2.columns)):
    df_new[df1.columns[i]]=df1[df1.columns[i]]
    df_new[df2.columns[i]]=df2[df2.columns[i]]

df_new
df_new.shape


# make X to complex numpy array
import numpy as np
X_reshape = np.array(df_new).reshape(10657,1,125,2)
complex_df = np.apply_along_axis(lambda args: [complex(*args)], 3, X_reshape)
complex_df_reshape = complex_df.reshape(10657,125)

# Fast Fourier Transformation
from scipy.fftpack import fft
ft_df = fft(complex_df_reshape)
ft_real = pd.DataFrame(ft_df.real)
ft_imag = pd.DataFrame(ft_df.imag)
X_full = pd.concat([ft_real, ft_imag],axis=1)

X_full
X_full.to_csv('labelData_fft.csv', index=False)

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# load fft transformed data
import pandas as pd
df_fft = pd.read_csv('labelData_fft.csv',header=0)

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[5]:


# Original Data
X = df.iloc[:,:250]
y = df[250]
#print(X.describe())
# print(X.shape)

# get unique value for the label
import numpy as np
#np.unique(y)
#print(y.describe())

# FFT
# split features (X) and lables (y)
X_fft = df_fft
#print(X_fft.describe())
#print(X_fft.shape)

import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# standardize the data as z-scores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype('float64'))
X_fft_scaled = scaler.fit_transform(X_fft.astype('float64'))

# split training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
X_fft_train, X_fft_test, y_train, y_test = train_test_split(X_fft_scaled, y, test_size=0.33, random_state=42)
y_train = y_train - 1
y_test = y_test -1

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

# deep learning on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

import tensorflow as tf
from tensorflow import keras
tf.__version__
keras.__version__

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[250]))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"]) # sgd

filepath = '/Users/jiahuali1991/Dropbox/Machine Learning/Data/Ouyang/DeepNet_Model_labelData_original.h5'
# model.load_weights(filepath) #load previously trained model


history = model.fit(X_train, y_train, epochs=500, batch_size = 50, verbose = 0,
                    validation_data=(X_test, y_test),
                   callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)
'''
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
'''

y_test_pred_dl = model.predict_classes(X_test)
# model.evaluate(X_fft_test, y_test)
from sklearn.metrics import accuracy_score  
print("Deep Learning Model Accuracy: %.2f%%" % (accuracy_score(y_test, y_test_pred_dl) * 100.0))

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# deep learning on fft tranformed data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

import tensorflow as tf
from tensorflow import keras
tf.__version__
keras.__version__

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[250]))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"]) # sgd

filepath = '/Users/jiahuali1991/Dropbox/Machine Learning/Data/Ouyang/DeepNet_Model_labelData_fft.h5'
# model.load_weights(filepath) #load previously trained model


history = model.fit(X_fft_train, y_train, epochs=500, batch_size = 50, verbose = 0,
                    validation_data=(X_fft_test, y_test),
                   callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)
'''
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
'''
y_test_pred_dl = model.predict_classes(X_fft_test)
# model.evaluate(X_fft_test, y_test)
from sklearn.metrics import accuracy_score  
print("Deep Learning Model Accuracy: %.2f%%" % (accuracy_score(y_test, y_test_pred_dl) * 100.0))

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# Random Forest on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Random Forest Model')    
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_rf = rf_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
rf_accuracy = accuracy_score(y_test, y_test_pred_rf)
print("Random Forest Model Accuracy: %.2f%%" % (rf_accuracy * 100.0))

# save the model to disk
import pickle
filename = 'RandomForest_model_original.sav'
pickle.dump(rf_clf, open(filename, 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)



# Random Forest on fft transformed data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Random Forest Model')    
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_rf = rf_clf.predict(X_fft_test)
from sklearn.metrics import accuracy_score  
print("Random Forest Model Accuracy: %.2f%%" % (accuracy_score(y_test, y_test_pred_rf) * 100.0))

# save the model to disk
import pickle
filename = 'RandomForest_model_fft.sav'
pickle.dump(rf_clf, open(filename, 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")


# Classification and Regression Trees (CART) on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting CART Model')    
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_dt = dt_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
dt_accuracy = accuracy_score(y_test, y_test_pred_dt)
print("CART Model Accuracy: %.2f%%" % (dt_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'CART_model_original.sav'
pickle.dump(dt_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

# Classification and Regression Trees (CART) on fft data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting CART Model')    
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_dt = dt_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
dt_accuracy = accuracy_score(y_test, y_test_pred_dt)
print("CART Model Accuracy: %.2f%%" % (dt_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'CART_model_fft.sav'
pickle.dump(dt_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# KNN on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting KNN Model')    
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_neighbors=17, p=2,
                     weights='distance', n_jobs = -1) 
knn_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_knn = knn_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
knn_accuracy = accuracy_score(y_test, y_test_pred_knn)
print("KNN Model Accuracy: %.2f%%" % (knn_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'KNN_model_original.sav'
pickle.dump(knn_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# KNN on fft transformed data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting KNN Model')    
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_neighbors=17, p=2,
                     weights='distance', n_jobs = -1) 
knn_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_knn = knn_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
knn_accuracy = accuracy_score(y_test, y_test_pred_knn)
print("KNN Model Accuracy: %.2f%%" % (knn_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'KNN_model_fft.sav'
pickle.dump(knn_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

# Naive Bayes on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Gaussian Naive Bayes Model')    
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_gnb = gnb_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
gnb_accuracy = accuracy_score(y_test, y_test_pred_gnb)
print("Gaussian Naive Bayes Model Accuracy: %.2f%%" % (gnb_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'NB_model_original.sav'
pickle.dump(gnb_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# Naive Bayes
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Gaussian Naive Bayes Model')    
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()
gnb_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_gnb = gnb_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
gnb_accuracy = accuracy_score(y_test, y_test_pred_gnb)
print("Random Forest Model Accuracy: %.2f%%" % (gnb_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'NB_model_fft.sav'
pickle.dump(gnb_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# Logistic Regression on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Logistic Regression Model')    
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=0)
lr_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_lr = lr_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
lr_accuracy = accuracy_score(y_test, y_test_pred_lr)
print("Logistic Regression Model Accuracy: %.2f%%" % (lr_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'LR_model_original.sav'
pickle.dump(lr_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# Logistic Regression
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Logistic Regression Model')    
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=0)
lr_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_lr = lr_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
lr_accuracy = accuracy_score(y_test, y_test_pred_lr)
print("Logistic Regression Model Accuracy: %.2f%%" % (lr_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'LR_model_fft.sav'
pickle.dump(lr_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# SGD on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting SGD Model') 
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_sgd = sgd_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
sgd_accuracy = accuracy_score(y_test, y_test_pred_sgd)
print("SGD Model Accuracy: %.2f%%" % (sgd_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'SGD_model_original.sav'
pickle.dump(sgd_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[20]:


# SGD on Fourier Tranformed Data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting SGD Model') 
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_sgd = sgd_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
sgd_accuracy = accuracy_score(y_test, y_test_pred_sgd)
print("SGD Model Accuracy: %.2f%%" % (sgd_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'SGD_model_fft.sav'
pickle.dump(sgd_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[21]:


# Light GBM on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Light GBM Model') 
import lightgbm as lgb
gbm = lgb.LGBMClassifier()
gbm.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_gbm = gbm.predict(X_test)
print('start calculating accuracy score') 
gbm_accuracy = accuracy_score(y_test, y_test_pred_gbm)
print("Light GBM Model Accuracy: %.2f%%" % (gbm_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'LightGBM_model_original.sav'
pickle.dump(gbm, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[22]:


# Light GBM on TTF data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Light GBM Model') 
import lightgbm as lgb
gbm = lgb.LGBMClassifier()
gbm.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_gbm = gbm.predict(X_fft_test)
print('start calculating accuracy score') 
gbm_accuracy = accuracy_score(y_test, y_test_pred_gbm)
print("Light GBM Model Accuracy: %.2f%%" % (gbm_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'LightGBM_model_fft.sav'
pickle.dump(gbm, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# XGBoost on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting XGBoost Model') 
from xgboost import XGBClassifier
XGB_clf = XGBClassifier()
XGB_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_XGB = XGB_clf.predict(X_test)
print('start calculating accuracy score') 
XGB_accuracy = accuracy_score(y_test, y_test_pred_XGB)
print("XGBoost Model Accuracy: %.2f%%" % (XGB_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'XGBoost_model_original.sav'
pickle.dump(XGB_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[24]:


# XGBoost
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting XGBoost Model') 
from xgboost import XGBClassifier
XGB_clf = XGBClassifier()
XGB_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_XGB = XGB_clf.predict(X_fft_test)
print('start calculating accuracy score') 
XGB_accuracy = accuracy_score(y_test, y_test_pred_XGB)
print("XGBoost Model Accuracy: %.2f%%" % (XGB_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'XGBoost_model_fft.sav'
pickle.dump(XGB_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[25]:


# CatBoost on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
import time
start_time = time.time() 
print('start fitting CatBoost Model') 
from catboost import CatBoostClassifier
catb_clf = CatBoostClassifier()
catb_clf.fit(X_train, y_train, verbose=0)
print('start making prediction') 
y_test_pred_catb = catb_clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('start calculating accuracy score') 
catb_accuracy = accuracy_score(y_test, y_test_pred_catb)
print("Cat Boost Model Accuracy: %.2f%%" % (catb_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'CatBoost_model_original.sav'
pickle.dump(catb_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[26]:


# CatBoost on FFT data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
import time
start_time = time.time() 
print('start fitting CatBoost Model') 
from catboost import CatBoostClassifier
catb_clf = CatBoostClassifier()
catb_clf.fit(X_fft_train, y_train, verbose=0)
print('start making prediction') 
y_test_pred_catb = catb_clf.predict(X_fft_test)
from sklearn.metrics import accuracy_score
print('start calculating accuracy score') 
catb_accuracy = accuracy_score(y_test, y_test_pred_catb)
print("Cat Boost Model Accuracy: %.2f%%" % (catb_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'CatBoost_model_fft.sav'
pickle.dump(catb_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[32]:


# Support Vector Machine on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Support Vector Machine Model') 
from sklearn import svm
svm_clf = svm.SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svm_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_svm = svm_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
svm_accuracy = accuracy_score(y_test, y_test_pred_svm)
print("Support Vector Machine Model Accuracy: %.2f%%" % (svm_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'SVM_model_original.sav'
pickle.dump(svm_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[33]:


# Support Vector Machine on fft transformed data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting Support Vector Machine Model') 
from sklearn import svm
svm_clf = svm.SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svm_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_svm = svm_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
svm_accuracy = accuracy_score(y_test, y_test_pred_svm)
print("Support Vector Machine Model Accuracy: %.2f%%" % (svm_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'SVM_model_fft.sav'
pickle.dump(svm_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[36]:


# AdaBoost on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting AdaBoost Model') 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=200, algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_train, y_train)
print('start making prediction') 
y_test_pred_ada = ada_clf.predict(X_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
ada_accuracy = accuracy_score(y_test, y_test_pred_ada)
print("AdaBoost Model Accuracy: %.2f%%" % (ada_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'AdaBoost_model_original.sav'
pickle.dump(ada_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[37]:


# AdaBoost on original data
import time
start_time = time.time()   
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
print('start fitting AdaBoost Model') 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=200, algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_fft_train, y_train)
print('start making prediction') 
y_test_pred_ada = ada_clf.predict(X_fft_test)
print('start calculating accuracy score') 
from sklearn.metrics import accuracy_score  
ada_accuracy = accuracy_score(y_test, y_test_pred_ada)
print("AdaBoost Model Accuracy: %.2f%%" % (ada_accuracy * 100.0))
# save the model to disk
import pickle
filename = 'AdaBoost_model_fft.sav'
pickle.dump(ada_clf, open(filename, 'wb'))
print("--- %s seconds ---" % (time.time() - start_time))
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[ ]:




