# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:48:17 2023

@author: Duygu
"""

#This code is written to implement the 2LE-CEEMDAN-LSTM-SVR method. Relevant areas/sections are operated piecemeal.

import os
import pandas as pd
from PyEMD import CEEMDAN, EEMD
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hyperopt import fmin, tpe, hp
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import sqrt 
import keras
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.gridspec as gridspec
import investpy as inv



def decompose_signal(signal, method,trials, noise_std):
    if method == 'EMD':
        emd = EEMD()
        IMF = emd(signal)
    elif method == 'EEMD':
        eemd = EEMD(trials)
        IMF = eemd(signal)
    elif method == 'CEEMDAN':
        ceemdan = CEEMDAN(trials, noise_std)
        IMF = ceemdan(signal)
    return IMF


def ApEn(U, m, r) -> float:
    """Approximate_entropy."""
    "U: seri, m: embedding dimension, r: tolerance"

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def SampEn(L, m, r):
    """Sample entropy."""
    N = len(L)
    B = 0.0
    A = 0.0
    #sıfıra bölme hatasının önüne geçmek için epsilon kullanıldı
    EPSILON = 1e-12
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B + EPSILON)

def calculate_entropy_StdSapma(imf,m,r):
    approx = ApEn(imf,m,r)
    sample = SampEn(imf,m,r)
    std_sapma = np.std(imf)
    return round(approx,5), round(sample,5), round(std_sapma,5)

def calculate_entropy(imfs):
    #method = 'CEEMDAN' # Choose one of 'EMD', 'EEMD', or 'CEEMDAN'
    imf_entropy_StdDeviation = np.zeros((imfs.shape[0], 3))
    for i in range(imfs.shape[0]):
        imf_entropy_StdDeviation[i] = calculate_entropy_StdSapma(imfs[i],m,r)
    
    Entropy_StdDeviation = pd.DataFrame(imf_entropy_StdDeviation, columns = ['Approximate Entropy', 'Sample Entropy', 'Standard Deviation'])
    # Entropy_StdDeviation = Entropy_StdDeviation.fillna(method = 'ffill')
    return Entropy_StdDeviation
    # Ratio_entropy = column_ratio(imf_entropy_StdDeviation) # İlk kolon: ApEn, İkinci kolon SampEn

def selection_imfs(imfs, Ratio_entropy):
#Birinci ayrıştırmada elde edilen IMF'lerden ApEn_Ratio ve SampEn_Ratio > 20 (%20) olan IMF'ler yüksek frekanslı olarak ele alınır
#ve ikinci ayrıştırmaya tabi tutulur -> high_frequency_IMFs. Kalanları ise selected_IMFs_first dizisine atanır. 

    selected_IMFs_first = []
    high_frequency_IMFs = []
    for i in range(len(imfs)):
        # if (((Ratio_entropy.iloc[i, 0] + Ratio_entropy.iloc[i, 1] )/2) < 20):
        if ((Ratio_entropy.iloc[i, 0] < 20) and (Ratio_entropy.iloc[i, 1] < 20)):
        # if ((Ratio_entropy.iloc[i, 1] < 20)):
            selected_IMFs_first.append(imfs[i])
        else:
            high_frequency_IMFs.append(imfs[i])

    #Yüksek frekanslı olarak belirlenen IMF'lerin toplanması ve çizdirilmesi
    high_frequency_IMF_sum=0
    for num in high_frequency_IMFs:
        high_frequency_IMF_sum += num
    return selected_IMFs_first, high_frequency_IMFs, high_frequency_IMF_sum

def MAPE(Y_actual,Y_Predicted):

    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def repeat_list(n, x):
    return [x] * n


# folder_path = 'E:/9 Mayıs 2023/HSI-R-1'
folder_path = 'E:/a'
#One of the following options should be selected for the index data you want to work on.
symbol = "^GSPC" 
# symbol = "000001.SS" 
# symbol = "^GDAXI"
# symbol = "^DJI"

start_date = "2010-01-01"
end_date = "2019-10-01"


df = yf.download(symbol, start=start_date, end=end_date).reset_index()
df = df.fillna(method='bfill')
decomp_method = input("Enter decomposition method (EMD, EEMD, or CEEMDAN): ")

# Define the decomposition method to use based on user input
if decomp_method == "EMD":
    method = 'EMD'
elif decomp_method == "EEMD":
    method = 'EEMD'
elif decomp_method == "CEEMDAN":
    method = 'CEEMDAN'
else:
    print("Invalid decomposition method entered. Defaulting to EEMD.")
    method = 'EEMD'
    
if (decomp_method == "EEMD" or decomp_method == "CEEMDAN"):
    trials = int(input("Enter number of trials for EEMD or CEEMDAN decomposition: "))
if decomp_method == "CEEMDAN":
    noise_std = float(input("Enter the standard deviation of the white noise used: "))
    
m= 2
r=0.2   

   

# =============================================================================
# Implementation of The First Decomposition

imfs_first_close = decompose_signal(df['Close'].values, method, trials, noise_std)


Entropy_StdDeviation_first_close = calculate_entropy(imfs_first_close)
col_sum = np.sum(Entropy_StdDeviation_first_close, axis=0)
ratios = np.divide(Entropy_StdDeviation_first_close, col_sum)
Ratio_entropy = ratios*100
Ratio_entropy_first_close = Ratio_entropy.iloc[:,:2]
selected_IMFs_first_close, high_frequency_IMFs_first_close, high_frequency_IMF_sum_first_close = selection_imfs(imfs_first_close, Ratio_entropy_first_close)

plt.figure()
fig, axs = plt.subplots(nrows=len(imfs_first_close)+1, figsize=(10,10), sharex= True)
fig.subplots_adjust(hspace=0.2)

fig.suptitle("First Decomposition Results")

# Plot the original data in the first subplot
axs[0].plot(df['Close'].values)
# axs[0].plot(df)
axs[0].set_title('Original Data')

# Plot the IMFs in subsequent subplots
for i in range(len(imfs_first_close)):
    axs[i+1].plot(imfs_first_close[i])
    axs[i+1].set_ylabel(f'IMF {i+1}', rotation=90, labelpad = 9)
fig.text(0.5, 0.04, 'Trading Day', ha='center')

for ax in axs.flat:
    ax.set_xlabel('')
# Show the plot
plt.show()

file_name1 = 'first_decomposition_IMFs.png'
full_path1 = os.path.join(folder_path, file_name1)
fig.savefig(full_path1)

# =============================================================================
#Implemantation of The Second Decomposition

imfs_second_close = decompose_signal(high_frequency_IMF_sum_first_close, method, trials, noise_std)
Entropy_StdDeviation_second_close = calculate_entropy(imfs_second_close)
col_sum_second = np.sum(Entropy_StdDeviation_second_close, axis=0)
ratios_second = np.divide(Entropy_StdDeviation_second_close, col_sum_second)
Ratio_entropy_second = ratios_second*100
Ratio_entropy_second_close = Ratio_entropy_second.iloc[:,:2]
selected_IMFs_second_close, high_frequency_IMFs_second_close, high_frequency_IMF_sum_second_close = selection_imfs(imfs_second_close, Ratio_entropy_second_close)

#Yüksek Frekanslı IMFlerden elde edilen 2. decomposition IMF lerinin çizdirilmesi
plt.figure()
fig, axs = plt.subplots(nrows=len(imfs_second_close)+1, figsize=(10,10), sharex= True)
fig.subplots_adjust(hspace=0.2)

fig.suptitle("Second Decomposition Results")

# Plot the original data in the first subplot
axs[0].plot(high_frequency_IMF_sum_first_close)
axs[0].set_title('High Frequency Data')

# Plot the IMFs in subsequent subplots
for i in range(len(imfs_second_close)):
    axs[i+1].plot(imfs_second_close[i])
    axs[i+1].set_ylabel(f'IMF {i+1}', rotation=90, labelpad = 9)
fig.text(0.5, 0.04, 'Trading Day', ha='center')

for ax in axs.flat:
    ax.set_xlabel('')
# Show the plot
plt.show()
file_name2 = 'first_decomposition_high_frequency_data.png'
full_path2 = os.path.join(folder_path, file_name2)
fig.savefig(full_path2)


#Ploting the IMFs selected as a result of second decomposition
plt.figure()
fig, axs = plt.subplots(nrows=len(selected_IMFs_second_close), figsize=(10,10), sharex= True)
fig.subplots_adjust(hspace=0.2)
fig.suptitle("Selected IMFs at The 2nd Decomposition")

# Plot the IMFs in subsequent subplots
for i in range(len(selected_IMFs_second_close)):
    axs[i].plot(selected_IMFs_second_close[i])
    axs[i].set_ylabel(f'IMF {i+1}', rotation=90, labelpad = 9)
fig.text(0.5, 0.04, 'Trading Day', ha='center')

for ax in axs.flat:
    ax.set_xlabel('')
# Show the plot
plt.show()
file_name3 = 'selected_IMFs_second_decomp.png'
full_path3 = os.path.join(folder_path, file_name3)
fig.savefig(full_path3)

# =============================================================================
# Plotting the actual data with the data determined noiseless as a result of the 1st and 2nd Decomposition

scaler= MinMaxScaler(feature_range=(0,1))
train_size = int(len(df) * 0.9)
test_size = len(df) - train_size

selected_all_sum_imfs = sum(selected_IMFs_first_close) + sum(selected_IMFs_second_close)
denoised_test_data_hiyerarşik = selected_all_sum_imfs[train_size:len(df)]

plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'].values, label='Original Data')
plt.plot(selected_all_sum_imfs, label='Denoised Data')
plt.xlabel('Trading Day')
plt.legend()
plt.show()

file_name4 = 'actual_and_denoised_data_hiyerarşik_duygu.png'
full_path4 = os.path.join(folder_path, file_name4)
fig.savefig(full_path4)
# =============================================================================
#Selecting the time steps value
timestep = 5
# timestep = 10
# timestep = 15
time_step_path = 'timestep_5'
# time_step_path = 'timestep_10'
# time_step_path = 'timestep_15'
time_step_folder_path = os.path.join(folder_path, time_step_path)


TEST_ACTUAL_DATA = (df['Close'][train_size:len(df)]).values
#When the imfs_second_close variable is assigned to the train_IMFs variable,
# the same training process (training the last IMF with SVR and the rest with LSTM)
# is performed for the noiseless IMFs obtained as a result of the second decomposition.
train_IMFs = imfs_first_close[:-1]
imf_pred, imf_pred_descaled = [], []
rmse = np.zeros((len(train_IMFs),1))
mse = np.zeros((len(train_IMFs),1))
mae = np.zeros((len(train_IMFs),1))
mape = np.zeros((len(train_IMFs),1))
mape_yuzsuz = np.zeros((len(train_IMFs),1))
r2 = []

#Train each IMF component separately
for i in range(len(train_IMFs)):
    print("IMF", i+1, "training...")
    dataset = train_IMFs[i]
    dataset_scaled = scaler.fit_transform(dataset.reshape(-1, 1))

    X_train, y_train, X_test, y_test, y_test_orj = [], [], [], [], []
    for j in range(timestep, train_size):
        X_train.append(dataset_scaled[j-timestep:j, 0])
        y_train.append(dataset_scaled[j, 0])
    for k in range(train_size, len(dataset)):
        X_test.append(dataset_scaled[k-timestep:k, 0])
        y_test.append(dataset_scaled[k, 0])
        y_test_orj.append(dataset[k])

    X_train, y_train, X_test, y_test, y_test_orj = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(y_test_orj)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model= Sequential()
    model.add(LSTM(units=128,return_sequences=True, activation='relu', input_shape=(timestep, 1)))
    model.add(Dropout(rate=0.1))
    model.add(LSTM(units=64,return_sequences=True, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(LSTM(units=16))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode ='min', min_delta = 0.0001)
    history=model.fit(X_train, y_train,epochs=200,batch_size=16,validation_data=(X_test, y_test), callbacks=[early_stop])
    
    testpredict = model.predict(X_test)
    imf_pred.append(testpredict)
    testPredict_descaled = scaler.inverse_transform(testpredict)
    imf_pred_descaled.append(testPredict_descaled)
    
    rmse[i] = sqrt(mean_squared_error(y_test_orj, testPredict_descaled))
    mse [i] = mean_squared_error(y_test_orj, testPredict_descaled)
    mae[i] = mean_absolute_error(y_test_orj, testPredict_descaled)
    mape[i] = MAPE(y_test_orj, testPredict_descaled)
    mape_yuzsuz[i] = mape[i]/100
    r2.append(r2_score(y_test_orj, testPredict_descaled))
    
    print("IMF", i+1, "training sonuçları")
    print('Test RMSE: %.3f' % rmse[i])
    print('Test MSE: %.3f' % mse[i])
    print('Test MAE: %.3f' % mae[i])
    print('Test MAPE: %.3f' % mape[i])
    print('Test MAPE_100_çarpmasız: %.3f' % mape_yuzsuz[i])
    print('Test R2: %.3f' % r2[i])
r2 = np.array(r2). reshape(len(r2), 1)
result_metrik = np.concatenate((rmse, mae, mape, mape_yuzsuz, mse, r2), axis=1)

# =============================================================================
#SVR+hyperparameter tuning implementation for the latest IMF 

data = imfs_first_close[-1:]
dataset_knn = np.array(data, dtype = float).reshape(-1,1)
dataset_knn_scaled = scaler.fit_transform(np.array(dataset_knn).reshape(-1,1))


X_train_knn, y_train_knn, X_test_knn, y_test_knn, y_test_orj_knn = [], [], [], [], []
for j in range(timestep, train_size):
   X_train_knn.append(dataset_knn_scaled[j-timestep:j, 0])
   y_train_knn.append(dataset_knn_scaled[j, 0])
for k in range(train_size, len(dataset)):
    X_test_knn.append(dataset_knn_scaled[k-timestep:k, 0])
    y_test_knn.append(dataset_knn_scaled[k, 0])
    y_test_orj_knn.append(dataset_knn[k])

X_train_knn, y_train_knn, X_test_knn, y_test_knn, y_test_orj_knn = np.array(X_train_knn), np.array(y_train_knn), np.array(X_test_knn), np.array(y_test_knn), np.array(y_test_orj_knn)
X_train_knn = np.reshape(X_train_knn, (X_train_knn.shape[0], X_train_knn.shape[1], 1))
X_test_knn = np.reshape(X_test_knn, (X_test_knn.shape[0], X_test.shape[1], 1))

svr = SVR(kernel = 'rbf')
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'epsilon': [0.1, 0.01, 0.001], 'kernel':['rbf', 'linear']}
tscv = TimeSeriesSplit(n_splits=5)
grid_model = GridSearchCV(estimator=svr, param_grid=param_grid, cv=tscv, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error', error_score='raise')
grid_model.fit(X_train_knn.reshape(-1, timestep), y_train_knn)
print('Best hyperparameters:', grid_model.best_params_)

best_C = grid_model.best_params_['C']
best_epsilon = grid_model.best_params_['epsilon']
best_kernel = grid_model.best_params_['kernel']
best_gamma = grid_model.best_params_['gamma']

svr = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel, gamma = best_gamma)
svr.fit(X_train_knn.reshape(-1, timestep), y_train_knn)
test_predict_svr = svr.predict(X_test_knn.reshape(-1, timestep))
descaled_test_predict_svr = scaler.inverse_transform(test_predict_svr.reshape(-1,1))

rmse_svr = sqrt(mean_squared_error(y_test_orj_knn, descaled_test_predict_svr))
mse_svr = mean_squared_error(y_test_orj_knn, descaled_test_predict_svr)
mae_svr = mean_absolute_error(y_test_orj_knn, descaled_test_predict_svr)
mape_svr = MAPE(y_test_orj_knn, descaled_test_predict_svr)
mape_yuzsuz_svr = mape_svr/100
r2_svr = (r2_score(y_test_orj_knn, descaled_test_predict_svr)) 

print("SVR SONUÇLARI")
print('Test RMSE: %.3f' % rmse_svr)
print('Test MSE: %.3f' % mse_svr)
print('Test MAE: %.3f' % mae_svr)
print('Test MAPE: %.3f' % mape_svr)
print('Test MAPE_100_çarpmasız: %.3f' % mape_yuzsuz_svr)
print('Test R2: %.3f' % r2_svr)

result_metrik_svr = np.array((rmse_svr, mae_svr, mape_svr, mape_yuzsuz_svr, mse_svr, r2_svr)).reshape(1,6)

# =============================================================================

#The results (error metrics) produced by the code were combined in an external Excel file as specified in the study.
