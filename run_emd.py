#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:10:16 2018

@author: mducoffe, rflammary, ncourty
"""

"""
run emd on mnist
"""

import numpy as np
import scipy as sp
import scipy.io as spio
import ot
import os

from keras.datasets import mnist

MNIST='mnist'
REPO='data'
CAT='cat'
CRAB='crab'
FACE='face'
REPO='data'
#%%

def run_emd(dataset_name='mnist', train=True, n_pairwise=1000000, n_iter=1, n_proc=None):
    
    assert dataset_name in [MNIST, CAT, CRAB, FACE], 'unknown dataset {}'.format(dataset_name)
    
    if n_proc is None:
        import multiprocessing
        n_proc = multiprocessing.cpu_count()
        
    print('number of processors used {}'.format(n_proc))
    
    if dataset_name==MNIST:
        n = 28
        if train:
            (x_train, _), _ = mnist.load_data()
            xapp=x_train.reshape((len(x_train),-1))*1.0
        else:
            _, (x_test, _) = mnist.load_data()
            xapp=x_test.reshape((len(x_test),-1))*1.0
            
    if dataset_name in [CAT, CRAB, FACE]:
        n=28
        url_path = "https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap"
        
        assert os.path.isfile(os.path.join(REPO, '{}.npy'.format(dataset_name))), \
            "file not found: please download it at '{}' and put it in './{}/{}.npy'".format(url_path, REPO, dataset_name)
            
        X = np.load(os.path.join(REPO, '{}.npy'.format(dataset_name)))
        X = X.reshape((len(X),-1))*1.0
        X /=X.sum(1).reshape((-1,1))
        X=X.reshape((-1,1,n,n))
        
        # split into train, and test
        N = len(X)
        n_test = (int)(0.2*N)
        n_train = N - 2*n_test
        x_train = X[:n_train]
        x_test = X[n_train:]

        if train:
            xapp=x_train.reshape((len(x_train), -1))
        else:
            xapp=x_test.reshape((len(x_test), -1))
    ###################################################################""
            
    N = len(xapp)
    print(N)
    print(xapp.shape)
    xapp/=xapp.sum(1).reshape((-1,1))
    xx,yy=np.meshgrid(np.arange(n),np.arange(n))
    xy=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
    
    M=ot.dist(xy, xy)
        
    for i in range(n_iter):
        
        isource=np.random.randint(0,N,n_pairwise)
        itarget=np.random.randint(0,N,n_pairwise)
        def compute_emd(i):
            return ot.emd2(xapp[isource[i],:],xapp[itarget[i],:],M)
        
        ilist=range(n_pairwise)
        D2=np.array(ot.utils.parmap(compute_emd,ilist,n_proc))
        
        spio.savemat('{}/{}_{}_{}.mat'.format(REPO, dataset_name, 'train' if train else 'test', i),{'is':isource,'it':itarget,'D':D2})
        
#%%
if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset_name', type=str, default='cat', help='dataset name')
    parser.add_argument('--n_pairwise', type=int, default=100, help='number of pairwise emd')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations')
    
    args = parser.parse_args()                                                                                                                                                                                                                             
    dataset_name=args.dataset_name
    n_pairwise=args.n_pairwise
    n_iter=args.n_iter

    run_emd(dataset_name, True, n_pairwise, n_iter)