import numpy as np
from warnings import warn
from joblib import Parallel, delayed
import copy, argparse, os, math, random, time
from scipy import io,linalg
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import blas
import warnings
import pandas as pd
from numpy import dot, multiply

from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, squared_norm
from sklearn.utils.validation import check_non_negative

from math import sqrt
import warnings
import numbers

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pickle
import shutil
    
class DENTIST:
    '''
    Performs netNMF-sc with gradient descent using Tensorflow
    '''
    def __init__(self, distance="frobenius", d=None, S=None, lamb=10, n_inits=2, tol=1e-2, 
                 max_iter=20000, n_jobs=4, weight=0.5, parallel_backend='multiprocessing', 
                 normalize=False, lr=0.0001, init='nndsvd', random_state=None):
        """
            d:          number of dimensions
            S:          Similarity Matrix
            lamb:      regularization parameter
            n_inits:    number of runs to make with different random inits (in order to avoid being stuck in local minima)
            n_jobs:     number of parallel jobs to run, when n_inits > 1
            tol:        stopping criteria
            max_iter:   stopping criteria
        """
        self.X = None
        self.M = None
        self.d = d
        self.S = S
        self.lamb = lamb
        self.n_inits = n_inits
        self.rand_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.normalize = normalize
        self.weight = weight
        self.distance = distance
        self.lr = lr
        self.init = init
                 
    def _initialize(self, X, init=None, eps=1e-6, random_state=None):
        n_components = self.d
        
        check_non_negative(X, "DENTIST initialization")
        n_samples, n_features = X.shape
        if (init is not None
            and n_components > min(n_samples, n_features)):
            raise ValueError("init = '{}' can only be used when "
                             "n_components <= min(n_samples, n_features)".format(init))

        if init is None:
            init = 'wdgsvd'
        elif init == 'nndsvd':
            # NNDSVD initialization(Nonnegative Double Singular Value Decomposition)
            U, S, V = randomized_svd(X, n_components, random_state=random_state)
            W, H = np.zeros(U.shape), np.zeros(V.shape)
            # The leading singular triplet is non-negative
            # so it can be used as is for initialization.
            W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
            H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
            for j in range(1, n_components):
                x, y = U[:, j], V[j, :]
                # extract positive and negative parts of column vectors
                x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
                x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
                # and their norms
                x_p_nrm, y_p_nrm = sqrt(squared_norm(x_p)), sqrt(squared_norm(y_p))
                x_n_nrm, y_n_nrm = sqrt(squared_norm(x_n)), sqrt(squared_norm(y_n))
                m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
                # choose update
                if m_p > m_n:
                    u = x_p / x_p_nrm
                    v = y_p / y_p_nrm
                    sigma = m_p
                else:
                    u = x_n / x_n_nrm
                    v = y_n / y_n_nrm
                    sigma = m_n
                lbd = np.sqrt(S[j] * sigma)
                W[:, j] = lbd * u
                H[j, :] = lbd * v
            W[W < eps] = 0
            H[H < eps] = 0
        elif init == 'wdgsvd':
            n_pca = 100
            bound = 0.085
            n_pca = min(n_pca, np.shape(X)[1] - 1)
            W0, single_value, H0 = svds(X, k=n_pca, which='LM')
            single_value = single_value[range(n_pca - 1, -1, -1)]
            
            W0 = W0[:, range(n_pca - 1, -1, -1)]
            W0_P = np.maximum(W0, 0)
            W0_N = np.maximum(-W0, 0)
            Wn_P_big = np.sum(W0_P * W0_P, axis=0)
            Wn_N_big = np.sum(W0_N * W0_N, axis=0)
            if np.sum(Wn_P_big >= Wn_N_big) > 0:
                W0[:, Wn_P_big >= Wn_N_big] = W0_P[:, Wn_P_big >= Wn_N_big]
            if np.sum(Wn_P_big < Wn_N_big) > 0:
                W0[:, Wn_P_big < Wn_N_big] = W0_N[:, Wn_P_big < Wn_N_big]
            W = W0[:,:n_components]
            
            H0 = H0[range(n_pca - 1, -1, -1), :]
            H0_P = np.maximum(H0, 0)
            H0_N = np.maximum(-H0, 0)
            Hn_P_big = np.sum(H0_P * H0_P, axis=1)
            Hn_N_big = np.sum(H0_N * H0_N, axis=1)
            if np.sum(Hn_P_big >= Hn_N_big) > 0:
                H0[Hn_P_big >= Hn_N_big, :] = H0_P[Hn_P_big >= Hn_N_big, :]

            if np.sum(Hn_P_big < Hn_N_big) > 0:
                H0[Hn_P_big < Hn_N_big, :] = H0_N[Hn_P_big < Hn_N_big, :]
            H = H0[:n_components, :]

        else:
            raise ValueError('Invalid initialize parameter: got %r instead of one of %r' %
                             (init, (None, 'nndsvd', 'wegsvd'))) 
        return np.abs(W), np.abs(H)


    def _fit(self, X):
        init_W, init_H = self._initialize(X=X, init=self.init, random_state=self.rand_state)
        if init_W.shape[1] != self.d or init_H.shape[0] != self.d:
            raise ValueError('wrong dimension')
        temp_W = np.array(init_W, order='F').astype(np.float32)
        temp_H = np.array(init_H, order='F').astype(np.float32)
        conv = False
        
        mask = tf.constant(self.M.astype(np.float32))
        eps = tf.constant(np.float32(1e-8))
        A = tf.constant(X.astype(np.float32)) + eps
        H =  tf.Variable(temp_H.astype(np.float32))
        W = tf.Variable(temp_W.astype(np.float32))
        print(np.max(self.M), np.min(self.M), np.sum(self.M))
        WH = tf.matmul(W, H)
        if self.weight < 1:
            WH = tf.multiply(mask, WH)
        WH += eps
        L_s = tf.constant(self.L.astype(np.float32))
        lamb_s = tf.constant(np.float32(self.lamb))
        

        if self.distance == 'frobenius':
            cost0 = tf.reduce_sum(tf.pow(A - WH, 2))
            costL = lamb_s * tf.trace(tf.matmul(tf.transpose(W), tf.matmul(L_s,W)))
        elif self.distance == 'KL':
            cost0 = tf.reduce_sum(tf.multiply(A, tf.math.log(tf.div(A, WH)))- A + WH)
            costL = lamb_s * tf.trace(tf.matmul(tf.transpose(W), tf.matmul(L_s, W)))
        else:
            raise ValueError('Select frobenius or KL for distance')

        if self.lamb > 0:
            cost = cost0 + costL
        else:
            cost = cost0

        lr = self.lr
        decay = 0.95

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(lr, global_step, self.max_iter, decay, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=.1)
        train_step = optimizer.minimize(cost,global_step=global_step)

        initial = tf.global_variables_initializer()
        # Clipping operation. This ensure that W and H learnt are non-negative
        clip_W = tf.assign(W, tf.maximum(tf.zeros_like(W), W))
        clip_H = tf.assign(H, tf.maximum(tf.zeros_like(H), H))
        clip = tf.group(clip_W, clip_H)
        
        start = time.clock()
        c = np.inf
        with tf.Session() as sess:
            sess.run(initial)
            for i in range(self.max_iter):
                sess.run(train_step)
                sess.run(clip)
                if i%300==0:
                    c2 = sess.run(cost)
                    e = c-c2
                    c = c2
                    if i%10000==0:
                        print(i,c,e)
                    if e < self.tol:
                        conv = True
                        break
            learnt_W = sess.run(W)
            learnt_H = sess.run(H)
        tf.reset_default_graph()
        end = time.clock()
        runTime = end - start
        return {
            'conv': conv,
            'obj': c,
            'W': learnt_W,
            'H': learnt_H,
            'time' :runTime
        }

    def _median_normalize(self, X):
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
            #print('log-tansforming X with pseudocount 1')
                return np.log(X_n+1)
            else:
                return X_n
        median = np.median(library_size)
        X_n = ((median / library_size) * X.T).T
        if np.max(X_n) >= 15:
            #print('log-tansforming X with pseudocount 1')
            return np.log(X_n+1)
        else:
            return X_n
        
    def check_symmetric(self, L, tol=1e-8):
        return np.allclose(L, L.T, atol=tol)

    def fit_transform(self, X=None):
        if type(X) == np.ndarray:
            self.X = X
        assert type(self.X) == np.ndarray
        if self.normalize:
            print('library size normalizing...')
            self.X = self._median_normalize(self.X)
        M = np.ones_like(self.X)
        M[self.X == 0] = self.weight
        self.M = M
        if self.d is None:
            self.d = min(X.shape)
            print('rank set to:', self.d)
        if self.S is not None:
            if np.max(abs(self.S)) > 0:
                self.S = self.S / np.max(abs(self.S))
            S = self.S
            D = np.sum(abs(self.S), axis=0) * np.eye(self.S.shape[0])
            print(np.count_nonzero(S),'edges')
            self.D = D
            self.S = S
            self.L = self.D - self.S
            assert self.check_symmetric(self.L)
        else:
            self.S = np.eye(X.shape[0])
            self.D = np.eye(X.shape[0])
            self.L = self.D - self.S
        
        if self.n_inits > 1:
            self.rand_state = None
        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(delayed(self._fit)(self.X) for x in range(self.n_inits))
        best_results = {"obj": np.inf, "W": None, "H": None, "time": 0}
        runTime = 0
        for r in results:
            if r['obj'] < best_results['obj']:
                best_results = r
            runTime = runTime + r['time']
        if 'conv' not in best_results:
            warn("Did not converge after {} iterations. Error is {}. Try increasing `max_iter`.".format(self.max_iter, best_results['e']))
            res = 0
        else:
            res = 1
        return (res, best_results['obj'], best_results['W'], best_results['H'], runTime)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main function for running WEDGE2.",
    )
    parser.add_argument(
        "input_path", 
        type=str, 
        help="Input folder containing all necessary files. Must have xnorm, sspr.pkl"
    )
    parser.add_argument(
        "--nmf_w",
        "-w",
        default=0.5,
        type=float,
        help="weight of scalar matrix",
    )
    parser.add_argument(
        "--nmf_lamb",
        "-l",
        default=10,
        type=float,
        help="regularization parameter",
    )
    parser.add_argument(
        "--nmf_dim",
        "-d",
        default=0,
        type=int,
        help="regularization parameter",
    )
    parser.add_argument(
        "--nmf_initial",
        "-i",
        default='wdgsvd',
        type=str,
        help="initial value",
    )
    parser.add_argument(
        "--dim_method",
        "-c",
        default='wedge', #['wedge', 'alra']
        type=str,
        help="dim method",
    )
    parser.add_argument(
        "--dim_multiple",
        "-p",
        default=3,
        type=int,
        help="dim multiple",
    )

    
    def flo2str(n): #删除小数点后多余的0并转换为字符
        n = str(n)
        if '.' in n:
            n = n.rstrip('0')  # 删除小数点后多余的0
            if n.endswith('.'):
                n = n.rstrip('.')
        return n
    
    args = parser.parse_args()
    wpth = args.input_path
    w = args.nmf_w
    lamb = args.nmf_lamb
    setdim = args.nmf_dim
    init = args.nmf_initial
    method = args.nmf_method
    dim_method = args.dim_method
    multi = args.dim_multiple
    
    def save(var, name):
        if os.path.exists(name+'.pkl'):
            os.remove(name+'.pkl')
        with open(name+'.pkl', 'wb') as f:
            pickle.dump(var, f)
    def load(name):
        if not os.path.exists(name+'.pkl'):
            raise ValueError("file not found")
        with open(name+'.pkl', 'rb') as f:
            return pickle.load(f)
    
    def components_num(X, dim_method, n_pca=100, bound=0.085):
        '''
        method: wedge, alra
        '''
        n_pca = min(n_pca, np.shape(X)[1] - 1)
        W0, single_value, _ = svds(X, k=n_pca, which='LM')
        single_value = single_value[range(n_pca - 1, -1, -1)] # s1>s2>s3>...
        
        if dim_method == 'wedge':
            single_value = single_value[single_value > 0]
            n_svd = min(n_pca, len(single_value) - 1)
            single_value = single_value / single_value[0]
            latent_new_diff = single_value[0:n_svd] / single_value[1:n_svd + 1] - 1
            n_rank = 2
            for i in range(len(latent_new_diff) - 10):
                n_rank = i + 1
                if (latent_new_diff[i] >= bound) and all(latent_new_diff[i + 1:11] < bound):
                    break
            n_rank = max(n_rank, 3)
            n_components = n_rank
        elif dim_method == 'alra':
            s1 = np.delete(single_value,-1) # s1,s2,...sn-1
            s2 = np.delete(single_value, 0) # s2,s3,...sn
            s_diff = s1 - s2
            mu = np.mean(s_diff[78:])
            sigma = np.std(s_diff[78:])
            thresh = (s_diff - mu / sigma)
            n_components = max(np.where(thresh > 6)[0]) + 1
        else:
            raise ValueError('Invalid components_num parameter: got %r instead of one of %r' %
                             (dim_method, (None, 'wedge', 'alra')))
        return n_components
    
    if setdim == 0:
        dim = None
    else:
        dim = setdim
    
    xnorm = load(wpth+'/xnorm') #sprod_data/sim/paper_data/visium_brain_sagittal/wedge/75/xnorm.pkl
    if setdim == 0:
        dim = components_num(X=xnorm, dim_method=dim_method)
        save(dim, wpth+'/dwedge')
        dim = min(multi * dim, 50)
    sim_mat = load(wpth+'/sspr')
    operator = WEDGE2(d=dim,S=sim_mat,lamb=lamb,weight=w,init=init)
    res, obj, W, H, runTime = operator.fit_transform(xnorm)
    if res == 0:
        save(res, wpth+'/notconverge')
    save(obj, wpth+'/obj'+s)
    save(W, wpth+'/H'+s)
    save(H, wpth+'/H'+s)
    X = np.dot(W, H)
    save(X, wpth+'/X')
    save(runTime, wpth+'/time')
                