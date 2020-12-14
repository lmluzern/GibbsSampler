import sampler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
import datetime

def gete2t(train_size,truth_labels):
    e2t = {}
    for example in range(train_size):
        e2t[example] = truth_labels[example]
    return e2t

epochs = 1

param = {
	'annotation_file' : '../input/multiclass_aij.csv',
    'labels_file' : '../input/multiclass_labels.csv',
    'A_0' : 8,
    'B_0' : 2,
    'gamma_0' : 8,
    'mu_0' : 2.9,
    'iters' : 1000,
    'burn_in_rate' : 0.5,
    'supervision_rate' : 0.6,
    'sampling_rate' : 0.4
    }

ground_truth = pd.factorize(pd.read_csv(param['labels_file'],sep=",")['label'],sort=True)[0] + 1
ground_truth_encoded = pd.get_dummies(ground_truth).values

l = []

for value in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: # supervision rate
	all_accuracy_test = []
	all_auc_test = []
	all_accuracy_val = []
	all_auc_val = []
	param['supervision_rate'] = value

	train_size = int(ground_truth.shape[0] * param['supervision_rate'])
	test_size = int((ground_truth.shape[0] * (1-param['supervision_rate']))/2)
	e2t = gete2t(train_size,ground_truth)

	for i in range(epochs):
		z_median,trace_conv = sampler.run(param,e2t)
		df_trace = pd.DataFrame(trace_conv[int(param['iters']*param['burn_in_rate']):],columns=['precision','mean','A_new','B_new','z','r']).describe()

		z_median_encoded = pd.get_dummies(z_median.round()).values
		accuracy_test = accuracy_score(ground_truth[-test_size:], z_median.round()[-test_size:])
		accuracy_val = accuracy_score(ground_truth[train_size:-test_size], z_median.round()[train_size:-test_size])
		auc_test = roc_auc_score(ground_truth_encoded[-test_size:], z_median_encoded[-test_size:],multi_class="ovo",average="macro")
		auc_val = roc_auc_score(ground_truth_encoded[train_size:-test_size], z_median_encoded[train_size:-test_size],multi_class="ovo",average="macro")
		all_accuracy_test.append(accuracy_test)
		all_auc_test.append(auc_test)
		all_accuracy_val.append(accuracy_val)
		all_auc_val.append(auc_val)
	print('value:',value,accuracy_val,accuracy_test)
	param['mean_accuracy_test'] = np.mean(all_accuracy_test)
	param['mean_auc_test'] = np.mean(all_auc_test)
	param['mean_accuracy_val'] = np.mean(all_accuracy_val)
	param['mean_auc_val'] = np.mean(all_auc_val)
	l.append(param.copy())

pd.DataFrame(l).to_csv('../output/sampler_'+str(datetime.datetime.now())+'.csv')
