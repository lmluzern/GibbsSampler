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

param = {
	'annotation_file' : '../input/influencer_aij.csv',
    'labels_file' : '../input/influencer_labels.csv',
    'A_0' : 8,
    'B_0' : 1,
    'gamma_0' : 1,
    'mu_0' : 2,
    'iters' : 1000,
    'burn_in_rate' : 0.2,
    'supervision_rate' : 0.6,
    'sampling_rate' : 0.0
    }

ground_truth = pd.factorize(pd.read_csv(param['labels_file'],sep=",")['label'],sort=True)[0] + 1
ground_truth_encoded = pd.get_dummies(ground_truth).values

mu_range = [0.5+x/2 for x in range(0,7)]
gamma_range = [1e-5,1e-4,1e-3,1e-2] + [x/10 for x in range(1,11)] + [x for x in range(2,11)] + [20,30,100,1000,10000,100000]
a_range = [x for x in range(0,9)]

# test
mu_range = [2.5,3]
gamma_range = [0.1,2]
a_range = [0,8]

print(a_range)
print(mu_range)
print(gamma_range)

l = []

train_size = int(ground_truth.shape[0] * param['supervision_rate'])
test_size = int((ground_truth.shape[0] * (1-param['supervision_rate']))/2)
e2t = gete2t(train_size,ground_truth)

for a_value in a_range:
	param['A_0'] = a_value
	print('a_value:',a_value)
	for mu_value in mu_range:
		param['mu_0'] = mu_value
		for gamma_value in gamma_range:
			param['gamma_0'] = gamma_value

			z_median,trace_conv = sampler.run(param,e2t)
			z_median_encoded = pd.get_dummies(z_median.round()).values
			accuracy_test = accuracy_score(ground_truth[-test_size:], z_median.round()[-test_size:])
			accuracy_val = accuracy_score(ground_truth[train_size:-test_size], z_median.round()[train_size:-test_size])
			auc_test = roc_auc_score(ground_truth_encoded[-test_size:], z_median_encoded[-test_size:],multi_class="ovo",average="macro")
			auc_val = roc_auc_score(ground_truth_encoded[train_size:-test_size], z_median_encoded[train_size:-test_size],multi_class="ovo",average="macro")

			param['mean_accuracy_test'] = accuracy_test
			param['mean_auc_test'] = auc_test
			param['mean_accuracy_val'] = accuracy_val
			param['mean_auc_val'] = auc_val
			l.append(param.copy())

pd.DataFrame(l).to_csv('../output/3d_sampler_'+str(datetime.datetime.now())+'.csv')
