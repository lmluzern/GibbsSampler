import pandas as pd
import numpy as np
import scipy.stats as stats
import arguments
import math
from sklearn.metrics import accuracy_score

LABEL_NAMES = ['emerging', 'established', 'no_option']
NUMBER_OF_LABELS = len(LABEL_NAMES)
LABEL_INDEX = np.array(range(0,NUMBER_OF_LABELS))

def run(annotation_file,labels_file,T,gamma,mu,alpha,beta,burn_in_rate):
    labels = pd.read_csv(labels_file, sep=",")
    labels['label_code'] = pd.factorize(labels['label'],sort=True)[0] + 1

    lower_bound = 0.5
    upper_bound = labels['label_code'].max() + 0.5

    annotation_matrix = pd.read_csv(annotation_file, sep=",",header=None)
    annotation_matrix['label_code'] = pd.factorize(annotation_matrix[2],sort=True)[0] + 1

    assert annotation_matrix[1].unique().shape[0] == labels.shape[0]

    n_workers = annotation_matrix[0].unique().shape[0]
    n_items = annotation_matrix[1].unique().shape[0]

    z_i = np.random.randn(n_items ,1) * math.sqrt(1/gamma) + mu
    for i in range(z_i.shape[0]):
        while z_i[i] <= lower_bound or z_i[i] >= upper_bound:
            z_i[i] = np.random.randn(1 ,1) * math.sqrt(1/gamma) + mu

    r_j = np.random.gamma(alpha,beta,(n_workers,1))
    while np.where( r_j < 0 )[0].shape[0] > 0:
        r_j = np.random.gamma(alpha,beta,(n_workers,1))

    ground_truth = labels['label_code'].values
    true_label = []

    for t in range(T):
        print('t:',t)

        # for each item label
        for i in annotation_matrix[1].unique():
            r_j_i = r_j[annotation_matrix[annotation_matrix[1] == i][0].values]
            aij = annotation_matrix[annotation_matrix[1] == i]['label_code'].values
            aij = aij.reshape((-1,1))
            temp_gamma = r_j_i.sum() + gamma
            temp_mu = ((aij*r_j_i).sum() + gamma*mu)/temp_gamma
            z_i[i] = np.random.randn(1 ,1) * math.sqrt(1/temp_gamma) + temp_mu
            while z_i[i] <= lower_bound or  z_i[i] >= upper_bound:
                z_i[i] = np.random.randn(1 ,1) * math.sqrt(1/temp_gamma) + temp_mu

        true_label.append(z_i.copy())

        # for each worker reliability
        for j in annotation_matrix[0].unique():
            sum_aij_zi = 0
            for index, row in annotation_matrix[annotation_matrix[0] == j].iterrows():
                # row[1] is the item id
                sum_aij_zi += (row['label_code'] - z_i[row[1]])**2

            temp_alpha = alpha + annotation_matrix[annotation_matrix[0] == j].shape[0] / 2
            temp_beta = beta + 0.5 * sum_aij_zi

            r_j[j] = np.random.gamma(temp_alpha,temp_beta,1)
            while r_j[j] < 0:
                r_j[j] = np.random.gamma(temp_alpha,temp_beta,1)
            
    true_label = np.array(true_label)
    burn_in_size = int(burn_in_rate * true_label.shape[0])
    print('burn_in_size',burn_in_size)
    true_label = true_label[burn_in_size:]
    true_label = np.mean(true_label,(0,2))

    accuracy = accuracy_score(ground_truth,true_label.round())
    print('accuracy',accuracy)
    
    # tbd convergence break needed?

if __name__ == '__main__':
    # load default arguments
    args = arguments.args

    run(**args)