import os
import itertools
import pandas as pd
import urllib
import numpy as np
from sklearn.metrics import pairwise_distances, roc_auc_score
import time
import math
from numba import jit
from sklearn.metrics import roc_auc_score

#Download drug-target interaction flat file from DrugCentral
df_dti = pd.read_csv('drug.target.interaction.tsv.gz', compression='gzip', sep='\t')
#print(df_dti.head(3))


#Generate a similarity matrix using Jaccard distance
#raw_folder = r'https://github.com/luoyunan/DTINet/blob/master/data/'
raw_files = {'drug_drug': 'mat_drug_drug.txt', 'drug_disease': 'mat_drug_disease.txt','drug_sideeffect': 'mat_drug_se.txt', 'protein_disease': 'mat_protein_disease.txt','protein_protein': 'mat_protein_protein.txt'}

raw_df = dict()
from urllib.parse import urljoin
for key, val in raw_files.items():
    raw_df[key] = pd.read_csv(val, header=None, sep='\s+',engine='python')
    #print(f'{key} table shape: {raw_df[key].shape}')


# add drug code, disease name    
drug_name = pd.read_csv('drug.txt', header=None)
disease_name = pd.read_csv('disease.txt', header=None, sep='\t')

raw_df['drug_disease'].index = drug_name[0]; raw_df['drug_disease'].index.name =None
raw_df['drug_disease'].columns = disease_name[0]
#print(raw_df['drug_disease'].iloc[:5,:5])

# Merge each drug and target similarity files
similarity_df = dict()
for key, val in raw_df.items():
    _df = val.astype(bool).to_numpy()
    _df = pd.DataFrame(pairwise_distances(_df, metric='jaccard'))
    _df = 1 - _df.map(lambda x: 1 if x == 0 else x)
    similarity_df[key] = _df + np.eye(len(_df))
#print(similarity_df['drug_disease'].map(lambda x: round(x,2)))


#Reduce similarity matrix dimensions using SVD
similarity_df['drug'] = pd.read_csv('Similarity_Matrix_Drugs.txt', header=None, sep='\s+')
similarity_df['protein'] = pd.read_csv('Similarity_Matrix_Proteins.txt', header=None, sep=' ')

def calc_dca(df_mat, maxiter=20, restart_prob=0.5):
    
    mat = df_mat.div(df_mat.sum(axis=1),axis=0).to_numpy()
    restart = np.eye(len(mat))
    mat_ret = np.eye(len(mat))
    
    for i in range(maxiter):
        mat_ret_new = (1-restart_prob) * mat * mat_ret + restart_prob * restart
        delta = np.linalg.norm(mat_ret-mat_ret_new, ord='fro')
        mat_ret = mat_ret_new
        if delta < 1e-6:
            break 
            
    return mat_ret
drug_mat = ['drug_drug','drug_disease','drug_sideeffect','drug']
protein_mat = ['protein_disease','protein_protein','protein']

drug_dca = np.concatenate([calc_dca(similarity_df[key]) for key in drug_mat], axis=1)
protein_dca = np.concatenate([calc_dca(similarity_df[key]) for key in protein_mat], axis=1)

# Concatenate matrix of Drug similarity
pd.DataFrame(drug_dca).apply(lambda x: round(x, 2))


from sklearn.decomposition import TruncatedSVD

def calc_svd(dca_mtx, num_feature=100):
    
    alpha = 1 / len(dca_mtx)
    dca_mtx = np.log(dca_mtx + alpha) - np.log(alpha)
    dca_mtx = np.matmul(dca_mtx, dca_mtx.transpose())

    svd = TruncatedSVD(n_components=num_feature)
    svd.fit(dca_mtx)
    result = svd.transform(dca_mtx)
    
    return result

# number of features
num_feature = 100

drug_svd = calc_svd(drug_dca, num_feature)
protein_svd = calc_svd(protein_dca, num_feature)

pd.DataFrame(drug_svd).apply(lambda x: round(x,2))

@jit(nopython=True)
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    '''
    ORIGINAL SOURCE: https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T

R = pd.read_csv('mat_drug_protein.txt', header=None, sep='\s+').to_numpy()

# N: number of drugs
N = R.shape[0]
# M: number of proteins
M = R.shape[1]
# Number of feature
K = num_feature

start = time.time()
n_drug, n_protein = matrix_factorization(R, drug_svd, protein_svd, K)
end = time.time()
print(f'Elapse: {end-start}')

n_drug_protein = np.dot(n_drug, n_protein.T)

rhat = n_drug_protein
rhat = pd.DataFrame(rhat).stack().reset_index()
rhat.set_index(rhat.apply(lambda x: f'{int(x.level_0)}_{int(x.level_1)}', axis=1), inplace=True)
rhat = rhat[0]
rhat.sort_index(inplace=True)

r = R
r = pd.DataFrame(r).stack().reset_index()
r.set_index(r.apply(lambda x: f'{int(x.level_0)}_{int(x.level_1)}', axis=1), inplace=True)
r = r[0]
r = r[rhat.index]
r.sort_index(inplace=True)

roc_auc_score(r, rhat)