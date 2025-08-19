#%% Import libraries and define general functions

from __future__ import absolute_import, division, print_function
import keras
import numpy as np
import h5py
from datetime import datetime
import os

def open_hdf_array(filename, variable_name): #Opens given hdf5 file and returns the given variable as a numpy array.
    if not os.path.isfile(filename):
        raise Exception('Filename: "'+filename+'" not found')        
    #print(' | Opening file: ', filename, ' ,variable: ', variable_name)
    file = h5py.File(filename)
    data = file.get(variable_name)
    data = np.array(data)
    data = data.transpose()
    file.close()
    return data

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print(' ')
        print('#', end='')

class PrintBestEpoch(keras.callbacks.Callback):
    def on_train_begin(self, epoch, logs={}):
        self.best_value = 10000000
        self.epoch = epoch
        return
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_loss') < self.best_value:
            self.best_value = logs.get('val_loss')
            self.epoch = epoch
        return
    def on_train_end(self, logs={}):
        print('Best epoch:', self.epoch + 1)
        return

#%% Load txt data

labels_binary=np.expand_dims(np.loadtxt('BinaryOutput_SpO2_cleaned4_.txt'),axis=1)
labels_count=np.expand_dims(np.loadtxt('CountOutput_SpO2_cleaned4_.txt'),axis=1)
data=np.loadtxt('DataMatrix_SpO2_cleaned4_.txt',delimiter=',')

test_prc=0.1
num_samples=np.size(labels_binary)
num_test=round(test_prc*num_samples)

# test/train division

all_inds=np.random.permutation(num_samples)

test_inds=all_inds[0:num_test]
train_inds=all_inds[num_test:num_samples]

test_data=data[test_inds,:]
train_data=data[train_inds,:]

test_labels_binary=labels_binary[test_inds]
test_labels_count=labels_count[test_inds]

train_labels_binary=labels_binary[train_inds]
train_labels_count=labels_count[train_inds]

train_labels_binary_oh=keras.utils.to_categorical(train_labels_binary, num_classes=2)

#%% Define models

val_split=0.1;
batsz=20000;
epnum=1000;
max_fail=100;
lrn_rate=0.0002;
model_name='b_'

def m_binary():
    model = keras.Sequential()
    model.add(keras.layers.Dense(30, activation='tanh', input_shape=(300,)))
    model.add(keras.layers.Dense(10, activation='tanh'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    optimizer = keras.optimizers.Adam(lrn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def m_count():
    model = keras.Sequential()
    model.add(keras.layers.Dense(30, activation='tanh', input_shape=(300,)))
    model.add(keras.layers.Dense(10, activation='tanh'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam(lrn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def run_nn(model_curr, train_data_curr, train_labels_curr, model_name_curr, weights_name_curr):
    dt_now=datetime.now()
    dt_str=dt_now.strftime("%d/%m/%Y %H:%M:%S")    
    print('Training started at: '+dt_str+'\n')
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=max_fail) 
    callbacks = [PrintBestEpoch(), early_stop, keras.callbacks.ModelCheckpoint(filepath=weights_name_curr, monitor='val_loss', save_best_only=True)]  
    history = model_curr.fit(train_data_curr, train_labels_curr, epochs=epnum, validation_split=val_split, verbose=1, batch_size=batsz, callbacks=callbacks)
    model_curr.save(model_name_curr)
    return model_curr, history

#%% Train binary

model_binary, history_binary=run_nn(m_binary(), train_data, train_labels_binary_oh, model_name+"ml_binary.keras", model_name+"wg_binary.keras")

#%% Train count

model_count, history_count=run_nn(m_count(), train_data, train_labels_count, model_name+"ml_count.keras", model_name+"wg_count.keras")


#%% Evaluate binary model

test_model=keras.models.load_model("b_ml_binary.keras")
model_output=test_model.predict(test_data)
model_output_fixed=np.expand_dims(np.argmax(model_output,1).astype(float),axis=1)

diff=model_output_fixed-test_labels_binary
errors=np.abs(diff)
eval_errorprc=np.sum(errors)/len(errors)
eval_accuracy=1-eval_errorprc

eval_summary={"Accuracy": np.round(eval_accuracy,3),"Error %": np.round(eval_errorprc*100,1)}
print('Desaturation binary model')
print(eval_summary)


#%% Evaluate count model

test_model=keras.models.load_model("b_ml_count.keras")
model_output=test_model.predict(test_data)
model_output_fixed=np.round(model_output.astype(float))

diff=model_output_fixed-test_labels_count
eval_mae=np.mean(np.abs(diff))
eval_mse=np.mean(np.abs(diff)**2)
errors=(np.abs(diff)>0).astype('float')
eval_errorprc=np.sum(errors)/len(errors)
eval_accuracy=1-eval_errorprc

eval_summary={"Accuracy": np.round(eval_accuracy,3),"MAE": np.round(eval_mae,3),"MSE": np.round(eval_mse,3)}
print('Desaturation count model')
print(eval_summary)





















