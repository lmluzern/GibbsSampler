# inspired by https://kieranrcampbell.github.io/blog/2016/05/15/gibbs-sampling-bayesian-linear-regression.html (18.11.2020)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time

def sample_z_i(gamma_0,mu_0,r_j,a_ij):
    precision = r_j.sum() + gamma_0
    mean = ((r_j*a_ij).sum() + gamma_0 * mu_0)/precision
    # return np.random.normal(mean, 1 / np.sqrt(precision))
    return np.random.normal(mean,np.sqrt(precision))

def sample_r_j(A_0,B_0,z_i,a_ij):
    A_new = A_0 + a_ij.shape[0]/2.0
    B_new = B_0 + ((a_ij-z_i)**2).sum()/2
    return np.random.gamma(A_new,1 /B_new)

def gibbs(param):
    annotation_matrix = pd.read_csv(param['annotation_file'],sep=",",header=None)
    annotation_matrix['label_code'] = pd.factorize(annotation_matrix[2],sort=True)[0] + 1
    n_workers = annotation_matrix[0].unique().shape[0]
    n_items = annotation_matrix[1].unique().shape[0]
    r  = np.full((n_workers,1),param['init_r'],dtype=float)
    z = np.full((n_items,1),-1,dtype=float)
    trace = np.zeros((param['iters'],n_items))

    for it in range(param['iters']):
        # for each item label
        print('it:',it)
        for i in annotation_matrix[1].unique():
            r_j = r[annotation_matrix[annotation_matrix[1] == i][0].values]
            a_ij = annotation_matrix[annotation_matrix[1] == i]['label_code'].values.reshape((-1,1))
            z[i] = sample_z_i(param['gamma_0'],param['mu_0'],r_j,a_ij)
        trace[it,:] = z.transpose()[0].copy()

        # for each worker reliability
        for j in annotation_matrix[0].unique():
            z_i = z[annotation_matrix[annotation_matrix[0] == j][1].values]
            a_ij = annotation_matrix[annotation_matrix[0] == j]['label_code'].values.reshape((-1,1))
            r[j] = sample_r_j(param['A_0'],param['B_0'],z_i,a_ij)

    return trace

if __name__ == '__main__':
    param = {
        'annotation_file' : '../input/multiclass_aij.csv',
        'labels_file' : '../input/multiclass_labels.csv',
        'A_0' : 2,
        'B_0' : 4,
        'gamma_0' : 1,
        'mu_0' : 2,
        'init_r' : 4,
        'iters' : 10,
        'burn_in_rate' : 0.5
        }

    start_t = time.time()
    trace = gibbs(param)
    print('gibbs execution time (s):', time.time() - start_t)

    labels = pd.read_csv(param['labels_file'],sep=",")
    labels['label_code'] = pd.factorize(labels['label'],sort=True)[0] + 1
    ground_truth = labels['label_code'].values

    trace_burnt = trace[int(param['iters']*param['burn_in_rate']):]
    z_median = np.median(trace_burnt,axis=0)
    accuracy = accuracy_score(ground_truth,z_median.round())
    print('accuracy',accuracy)
