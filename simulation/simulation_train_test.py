from __future__ import absolute_import, division, print_function

import numpy as np
import os
import sklearn
from sklearn import model_selection 

import pickle
import os,sys
import numpy as np
import xgboost, os
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
import utils

import sys
sys.path.append('../Models')
from Classifier import GeneralizedFuzzyEvolveClassifier
import matplotlib.pyplot as plt

def generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=1234):
  np.random.seed(random_state)
  x0 = 2**0.5*np.random.randn(n_samples, 1)
  x1 = 3**0.5*np.random.randn(n_samples, 1) + 5
  x2 = 5**0.5*np.random.randn(n_samples, 1) - 1
  x3 = 2**0.5*np.random.randn(n_samples, 1) + 1
  x4 = np.random.randn(n_samples, 1) - 2
  x_noise = np.random.randn(n_samples, 2)
  dynamic_features = np.concatenate([x0, x1, x2, x3, x4, x_noise], axis=-1)

  x1_static = np.expand_dims(np.random.randint(2, size=n_samples), axis=-1)
  x2_static = np.expand_dims(np.random.randint(10, size=n_samples), axis=-1)
  static_features = np.concatenate([x1_static,x2_static], axis=-1)

  timestamps = np.arange(n_timestamp)
  time_series_data = np.zeros((n_samples,n_timestamp,dynamic_features.shape[-1]))
  for i in range(n_samples):
      input_i = dynamic_features[i,:]
      input_i = input_i[:, np.newaxis] + np.cos((2 * input_i[:, np.newaxis] + timestamps[np.newaxis, :]))
      time_series_data[i] = input_i.T

  x0_last = time_series_data[:,-1,0].reshape((-1,1))
  x1_last = time_series_data[:,-1,1].reshape((-1,1))
  x2_last = time_series_data[:,-1,2].reshape((-1,1))
  x3_last = time_series_data[:,-1,3].reshape((-1,1))
  x4_last = time_series_data[:,-1,4].reshape((-1,1))

  rules = [np.logical_and.reduce([x1_last<3.8, x2_last>-2,x1_static == 1], axis=0),
              np.logical_and.reduce([x1_last>6.3, x2_last>-2,x2_static > 6], axis=0),
              np.logical_and.reduce([x0_last<1, x3_last>2], axis=0),
              np.logical_and.reduce([x2_last>0, x4_last>-1,x1_static == 1], axis=0),
              np.logical_and.reduce([x0_last<1, x4_last>-1.5,x2_static > 6], axis=0)]
    
  for i in range(len(rules)):
      print('Rule {}: {:.2f}%'.format(i, np.sum(rules[i])/n_samples*100))

  labels = np.logical_or.reduce(rules, axis=0)[:,0]
  if mislabel is not None:
      one_array = labels[labels==1]
      mutated_one_array = np.where(np.random.random(one_array.shape) < mislabel, False, one_array)
      labels[labels==1] = mutated_one_array


  categorical_info = np.zeros([dynamic_features.shape[1]]) 
  print('Positive samples: {:.2f}%'.format(np.sum(labels)/dynamic_features.shape[0]*100))
  return time_series_data,static_features, labels

def split_data_into_K_fold(n_samples,n_split):
  fold_taskname = np.empty(shape=(n_split, 3), dtype=object)

  idx_all = sorted(range(n_samples))
  for i_split, idx in enumerate(model_selection.KFold(5, shuffle=False).split(idx_all)):
      fold_taskname[i_split][2] = idx[-1]
  for i_split in range(n_split):
      fold_taskname[i_split][1] = fold_taskname[(i_split + 1) % n_split][2]
      fold_taskname[i_split][0] = np.asarray(sorted(set(idx_all).difference(fold_taskname[i_split][1]).difference(fold_taskname[i_split][2])))

  print(fold_taskname[0][0].shape, fold_taskname[0][1].shape, fold_taskname[0][2].shape)
  return fold_taskname

def generagte_train_val_test_from_fold(time_series_data, labels, fold_taskname):
  train_X = np.take(time_series_data,fold_taskname[0][0],axis = 0)
  train_X = train_X.reshape((int(n_samples*0.6),-1))
  train_y = np.take(labels,fold_taskname[0][0])

  test_X = np.take(time_series_data,fold_taskname[0][2],axis = 0)
  test_X = test_X.reshape((int(n_samples*0.2),-1))
  test_y = np.take(labels,fold_taskname[0][2])

  val_X = np.take(time_series_data,fold_taskname[0][1],axis = 0)
  val_X = test_X.reshape((int(n_samples*0.2),-1))
  val_y = np.take(labels,fold_taskname[0][1])

  return train_X, train_y, test_X, test_y,val_X,val_y

n_split = 5
n_samples = 1000
n_timestamp = 10
random_state = 42
time_series_data,static_data, labels = generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=42)

split_method = 'sample_wise'
category_info = np.zeros([time_series_data.shape[-1]]).astype(np.int32)
num_classes = 2
feature_names = ['x0','x1','x2','x3','x4','x5','x6']
static_feature_names = ['static_x1','static_x2']


num_time_varying_features = len(feature_names)
num_time_invariant_features = len(static_feature_names)
static_category_info = np.zeros(num_time_invariant_features) 
static_category_info[0] = 2
static_category_info = static_category_info.astype(np.int32)

time_varying_features = time_series_data
time_invariant_features = static_data
data = np.concatenate([time_series_data.reshape((n_samples,-1)),static_data],axis =1 )

max_steps = 800

ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X, y, X_test,y_test = utils.split_dataset(ss_train_test, data, labels, index=0)
X_train, y_train, X_val, y_val = utils.split_dataset(ss_train_test, X, y, index=0)
X_train = utils.fill_in_missing_value(X_train,X_train)
X_val = utils.fill_in_missing_value(X_val,X_val)
X_train_variant,X_train_invariant = utils.handle_data(X_train,num_time_invariant_features,n_timestamp)
X_val_variant,X_val_invariant = utils.handle_data(X_val,num_time_invariant_features,n_timestamp)

X_test = utils.fill_in_missing_value(X_test,X_test)
X_test_variant,X_test_invariant = utils.handle_data(X_test,num_time_invariant_features,n_timestamp)
print('The shape of training time-varying data:',X_train_variant.shape)
print('The shape of training time-invariant data:',X_train_invariant.shape)
print('The shape of validation time-varying data:',X_val_variant.shape)
print('The shape of testing time-varying data:',X_test_variant.shape)
print(static_category_info)
device = 'cpu'
max_steps = 800
evolve_type = 'GRU'
classifier = GeneralizedFuzzyEvolveClassifier(
                evolve_type = evolve_type,
                weighted_loss=[1.0,1.1],
                n_visits = n_timestamp,
                report_freq=50,
                patience_step=500,
                max_steps=max_steps,
                learning_rate=0.1,
                batch_size = 64,
                split_method='sample_wise',
                category_info=category_info,
                static_category_info=static_category_info,
                random_state=random_state,
                verbose=2,
                min_epsilon = 0.9,
                sparse_regu=1e-3,
                corr_regu=1e-4,
    
            )
classifier.fit(X_train_variant, y_train,X_train_invariant,
          X_val_variant,X_val_invariant, y_val,)
train_metrics = utils.cal_metrics_fbeta(classifier,X_train_variant,X_train_invariant,y_train,2)
print('train')
print(train_metrics)
val_metrics = utils.cal_metrics_fbeta(classifier,X_val_variant,X_val_invariant,y_val,2)
print('val')
print(val_metrics)
test_metrics = utils.cal_metrics_fbeta(classifier,X_test_variant,X_test_invariant,y_test,2)
print(test_metrics)