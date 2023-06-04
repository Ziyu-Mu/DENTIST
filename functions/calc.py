#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Calculating Values Needed

Created on Thur April 27 09:58:59 2023

@author: muziyu
"""

import pickle
import os
import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr

import scanpy as sc
import anndata
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

def save(var, name):
    """
    save current variable into .pkl format.
    Args:
        var: variable to be saved.
        name: path and .pkl name.
    """
    if os.path.exists(name+'.pkl'):
        os.remove(name+'.pkl')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(var, f)
        
        
def load(name):
    """
    load .pkl file.
    Args:
        name: path and .pkl name.
    Returns:
        return the variable.
    """
    if not os.path.exists(name+'.pkl'):
        raise ValueError(name)
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def flo2str(n):
    """
    transform a number into a succint string.
    Args:
        n: a number
    Returns:
        return the corresponding string
    """
    n = str(n)
    if '.' in n:
        n = n.rstrip('0') # delete extra zeros of decimal
        if n.endswith('.'):
            n = n.rstrip('.')
    return n


def deviancePoisson(X, sz=None):
    """
    X is matrix-like with observations in ROWs and features in COLs,
        sz is a 1D numpy array that is same length as rows of X
        sz are the size factors for each observation
        sz defaults to the row means of X
        set sz to 1.0 to have "no size factors" (ie Poisson for absolute counts instead of relative).

    Note that due to floating point errors the results may differ slightly
    between deviancePoisson(X) and deviancePoisson(X.todense()), use higher
    floating point precision on X to reduce this.
  """
    dtp = X.dtype
    X = X.astype("float64") #convert dtype to float64 for better numerics
    if sz is None:
        sz = np.ravel(X.mean(axis=1))
    else:
        sz = np.ravel(sz).astype("float64")
    if len(sz)==1:
        sz = np.repeat(sz,X.shape[0])
    feature_sums = np.ravel(X.sum(axis=0))
    try:
        with np.errstate(divide='raise'):
            ll_null = feature_sums*np.log(feature_sums/sz.sum())
    except FloatingPointError:
        raise ValueError("column with all zeros encountered, cannot compute deviance")
    if sps.issparse(X): #sparse scipy matrix
        LP = sps.diags(1./sz)@X
    #LP is a NxJ matrix with same sparsity pattern as X
        LP.data = np.log(LP.data) #log transform nonzero elements only
        ll_sat = np.ravel(X.multiply(LP).sum(axis=0))
    else: #dense numpy array
        X = np.asarray(X) #avoid problems with numpy matrix objects
        sz = np.expand_dims(sz,-1) #coerce to Nx1 array for broadcasting
        with np.errstate(divide='ignore',invalid='ignore'):
            ll_sat = X*np.log(X/sz) #sz broadcasts across rows of X
        ll_sat = ll_sat.sum(axis=0, where=np.isfinite(ll_sat))
    return (2*(ll_sat-ll_null)).astype(dtp)


def median_normalize(X):
    """
    median normalize for cell*gene dense matrix.
    Args:
        X: input data. Rows are cells/spots and columns are genes.
    Returns:
        return a normalized array.
    """
    library_size = X.sum(axis=1)
    if np.sum(library_size==0.) > 0:
        m,n = X.shape
        ls_se = pd.Series(library_size, index=range(m))
        zero_loc = ls_se[ls_se==0.].index #0行下标
        nonz_loc = ls_se[ls_se>0.].index #非0部分的下标
        ls_dz = ls_se[ls_se>0.].values #非0部分的值
        df = pd.DataFrame(X, index=range(m), columns=range(n)) #化为dataframe便于处理全零行
        df.drop(zero_loc, axis=0, inplace=True)
        X_dz = df.to_numpy() #去0
        med_dz = np.median(ls_dz)
        X_dz_n = ((med_dz / ls_dz) * X_dz.T).T
        df.loc[nonz_loc] = X_dz_n
        for l in zero_loc:
            df.loc[l] = np.zeros(n)
        df.sort_index(inplace=True)
        X_n = df.to_numpy()
        if np.max(X_n) >= 15:
            return np.log(X_n+1)
        else:
            return X_n
    median = np.median(library_size)
    X_n = ((median / library_size) * X.T).T
    if np.max(X_n) >= 15:
        return np.log(X_n+1)
    else:
        return X_n
    
    
def SNN(X, k=25, normalize=False):
    """
    calculate similarity matrix in SNN-cliq method.
    Args:
        X: input data. Rows are cells/spots and columns are genes.
        k: number of nearest neighbors.
        normalize: Default: False. If True, median_normalize input data.
    """
    num = X.shape[0]
    if normalize:
        X = median_normalize(X)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X)
    _, indice = nbrs.kneighbors(X)
    snn = np.zeros((num, num))
    for i in range(num):
        for j in range(i+1, num):
            nni = indice[i]
            nnj = indice[j]
            shared = np.intersect1d(nni, nnj)
            s = [0]
            for l in range(len(shared)):
                s.append(k - 1 - 0.5 * (np.where(nni == shared[l])[0][0] + np.where(nnj == shared[l])[0][0]))
            snn[i, j] = max(s)
            snn[j, i] = snn[i, j]
    return snn


def rank_w_a(X, n_pca=100, wedgebound=0.085, alracut=78, alratrshv=6):
    """
    calculate rank of a matrix using WEDGE and ALRA methods.
    Args:
        X: input data. Rows are cells/spots and columns are genes.
        n_pca: number of PCA.
        wedgebound: bound of fluctuation in WEDGE. Defualt: 0.085.
        alracut: default False. If True, median_normalize input data.
        alratrshv:
    Returns:
        rank of WEDGE, rank of ALRA
    """
    n_pca = min(n_pca, np.shape(X)[1] - 1)
    W0, single_value, _ = svds(X, k=n_pca, which='LM')
    single_value = single_value[range(n_pca - 1, -1, -1)] # s1>s2>s3>...
    s = single_value.copy()
    # wedge
    single_value = single_value[single_value > 0]
    n_svd = min(n_pca, len(single_value) - 1)
    single_value = single_value / single_value[0]
    latent_new_diff = single_value[0:n_svd] / single_value[1:n_svd + 1] - 1
    n_rank = 2
    for i in range(len(latent_new_diff) - 10):
        n_rank = i + 1
        if (latent_new_diff[i] >= wedgebound) and all(latent_new_diff[i + 1:11] < wedgebound):
            break
    n_rank = max(n_rank, 3)
    n_components_w = n_rank
    # alra
    s1 = np.delete(s,-1) # s1,s2,...sn-1
    s2 = np.delete(s, 0) # s2,s3,...sn
    s_diff = s1 - s2
    mu = np.mean(s_diff[alracut:])
    sigma = np.std(s_diff[alracut:])
    thresh = (s_diff - mu / sigma)
    n_components_a = max(np.where(thresh > alratrshv)[0]) + 1
    
    return n_components_w, n_components_a
 
    
def CMD(R1, R2):
    tr12 = np.dot(R1.ravel(), R2.ravel()) #tr(R1*R2.T)
    fro1 = np.dot(R1.ravel(), R1.ravel())
    fro2 = np.dot(R2.ravel(), R2.ravel())
    return 1 - tr12 / np.sqrt(fro1 * fro2)


def RMSE(xtrue, xpred):
    return np.sqrt(((xtrue-xpred)**2).mean())


def PCC(xtrue, xpred):
    m = xtrue.shape[0]
    pr = [pearsonr(xtrue[i,:], xpred[i,:]) for i in range(m)]
    pcc = list(map(lambda x:x[0], pr))
    pcc_p = list(map(lambda x:x[1], pr))
    pcc = np.array(pcc)
    pcc_p = np.array(pcc_p) # p_value
    return pcc, pcc_p

def ARI(xtrue, xpred):
    ari = adjusted_rand_score(xtrue, xpred)
    return ari

def Jaccard(xtrue, xpred):
    n_11 = 0
    n_10 = 0
    n_01 = 0
    n_00 = 0
    for i in range(len(xtrue)):
        for j in range(i+1,len(xtrue)):
            comember1 = (xtrue[i] == xtrue[j])
            comember2 = (xpred[i] == xpred[j])
            if comember1 and comember2:
                n_11 = n_11 + 1
            elif comember1 and (not comember2):
                n_10 = n_10 + 1
            elif (not comember1) and comember2:
                n_01 = n_01 + 1
            else:
                n_00 = n_00 + 1
    den = n_11 + n_10 + n_01
    if den == 0:
        jcrd = 0
    else:
        jcrd = n_11 / den
    return jcrd

def Silhouette(xpca, label):
    slt = silhouette_score(xpca, label)
    return slt

def umap(x, obs, var, savepth, n_comps=20, n_neigh=15):
    adumap = AnnData(sps.csr_matrix(x))
    adumap.obs = obs
    adumap.var = var
    sc.pp.pca(adumap, n_comps=n_comps)
    sc.pp.neighbors(adumap, n_neighbors=n_neigh)
    sc.tl.umap(adumap)
    sc.tl.leiden(adumap, key_added="umapClus")
    umapClus = adumap.obs['umapClus'].values.tolist()
    save(umapClus, savepth+'/umapClus')
    xumap = adumap.obsm['X_umap']
    save(xumap, savepth+'/xumap')
    xpca = adumap.obsm['X_pca']
    save(xpca, savepth+'/xpca')

def savepic(pth, name):
    if not os.path.exists(pth):
        os.makedirs(pth)
    pic = pth+'/'+name
    if os.path.exists(pic+'.png'):
        os.remove(pic+'.png')
    plt.savefig(pic, bbox_inches='tight')
    
    
def svdSIGdif_plot(wpth, dname='', signum=15, figsize=(5,4), s=13, plt_save=True, plt_show=False):
    _= plt.figure(figsize=figsize)
    _= plt.title('svdSIGdif'+dname)
    xnorm = load(wpth+'/xnorm')
    u, sig, v = np.linalg.svd(xnorm)
    sig = sorted(sig, reverse=True) # s1>s2>s3>... 
    s1 = np.delete(sig,-1) # s1,s2,...sn-1
    s2 = np.delete(sig, 0) # s2,s3,...sn
    sig = s1 - s2
    s_clip = sig[:signum]
    _= plt.scatter(range(signum), s_clip, s=s)
    if plt_save:
        savepic(wpth, 'svdSIGdif')
    if plt_show:
        plt.show()
    else:
        plt.close()