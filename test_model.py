#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:40:34 2018

@author: mducoffe, rflammary, ncourty
"""
#%%
import numpy as np
import pylab as pl
import scipy as sp
import ot

from keras.models import load_model

from dataset import get_data, MNIST, REPO
from build_model import MODEL
#%%
def compute(method_name, dataset_name=MNIST, repo=REPO):
    method_name = method_name.upper()
    assert method_name.upper() in ['MSE', 'BARYCENTER', 'PCA', 'INTERPOLATION'], 'unknown method {}'.format(method_name)
    
    if method_name=='MSE':
        emd = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'emd'))
        get_MSE(dataset_name, emd,repo)
        
    if method_name=='BARYCENTER':
        feat = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'feat'))
        unfeat = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'unfeat'))
        plot_Barycenter(dataset_name, feat, unfeat, repo)
        
    if method_name=='PCA':
        feat = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'feat'))
        unfeat = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'unfeat'))
        get_PCA(dataset_name, feat, unfeat, repo)
        
    if method_name=="INTERPOLATION":
        feat = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'feat'))
        unfeat = load_model('{}/{}_{}.hd5'.format(MODEL, dataset_name, 'unfeat'))
        bilinear_interpolation(dataset_name, feat, unfeat, repo)
#%%
def hide_axis():
    """hides axis but let you use xlabel and ylalbels"""
    pl.gca().spines['bottom'].set_color('white')
    pl.gca().spines['top'].set_color('white') 
    pl.gca().spines['right'].set_color('white')
    pl.gca().spines['left'].set_color('white')
    pl.xticks(())
    pl.yticks(())

def get_MSE(dataset_name, emd, repo):
    _, _, test=get_data(dataset_name, repo)
    xtest1, xtest2, ytest = test
    ot.tic()
    ytest_pred=emd.predict([xtest1,xtest2])
    t_est=ot.toc()
    err=np.mean(np.square(ytest_pred.ravel()-ytest.ravel()))
    errr=np.mean(np.square(ytest_pred.ravel()-ytest.ravel()))/np.mean(np.square(ytest.ravel()))
    r=np.corrcoef(ytest.ravel(),ytest_pred.ravel())[0,1]
    # compute quantiles
    nbin=30
    yp_mean=np.zeros((nbin,))
    yp_10=np.zeros((nbin,))
    yp_90=np.zeros((nbin,))
    yp_plot=np.zeros((nbin,))

    hst,bins=np.histogram(ytest[:],nbin)
    yp_plot[:]=np.array([.5*bins[k]+.5*bins[k+1] for k in range(nbin)])
    for j in range(nbin):
        idx=np.where((ytest[:]>bins[j]) * (ytest[:]<bins[j+1]) )
        ytemp=ytest_pred[idx]
        if ytemp.any():
            yp_mean[j]=ytemp.mean()
            yp_10[j]=np.percentile(ytemp,10)
            yp_90[j]=np.percentile(ytemp,90)
        else:
            yp_mean[j]=np.nan
            yp_10[j]=np.nan
            yp_90[j]=np.nan
    print('MSE={}\nRel MSE={}\nr={}\nEMD/s={}'.format(err,errr,r,ytest_pred.shape[0]/t_est))
    
    pl.figure(1,(8,3))
    pl.clf()
    pl.plot([0,45],[0,45],'k')
    xl=pl.axis()
    pl.plot(ytest,ytest_pred,'+')
    pl.plot([0,45],[0,45],'k')
    pl.axis(xl)

    pl.xlim([0,45])
    pl.ylim([0,45])
    pl.xlabel('True Wass. distance')
    pl.ylabel('Predicted Wass. distance')
    pl.title('True and predicted Wass. distance')
    pl.legend(('Exact prediction','Model prediction'))
    pl.savefig('imgs/{}_emd_pred_true.png'.format(dataset_name),dpi=300)
    pl.savefig('imgs/{}_emd_pred_true.pdf'.format(dataset_name))

    pl.subplot(1,2,2)
    pl.plot([ytest[:].min() ,ytest[:].max() ],[ytest[:].min() ,ytest[:].max() ],'k')
    pl.plot(yp_plot[:],yp_mean[:],'r+-')
    pl.plot(yp_plot[:],yp_10[:],'g+-')
    pl.plot(yp_plot[:],yp_90[:],'b+-')
    pl.xlim([0,45])
    pl.ylim([0,45])
    pl.legend(('Exact prediction','Mean pred','10th percentile','90th precentile',))
    pl.title('{} MSE:{:3.2f}, RelMSE:{:3.3f}, Corr:{:3.3f}'.format('',err,errr,r))
    pl.grid()
    pl.xlabel('True Wass. distance')
    pl.ylabel('Predicted Wass. distance')
    pl.savefig('imgs/{}_emd_pred_true_quantile.png'.format(dataset_name),dpi=300)

    pl.savefig('imgs/{}_perf.png'.format(dataset_name),dpi=300,bbox_inches='tight')
    pl.savefig('imgs/{}_perf.pdf'.format(dataset_name),dpi=300,bbox_inches='tight')

#%%
def _bary_wdl2(index, x, feat, unfeat):
        
    f=feat.predict(x[index])
    w = np.mean(f,axis=0)[None]
    return unfeat.predict(w)
    
def plot_Barycenter(dataset_name, feat, unfeat, repo):

    if dataset_name==MNIST:
        _, _, test=get_data(dataset_name, repo, labels=True)
        xtest1,_,_, labels,_=test
    else:
        _, _, test=get_data(dataset_name, repo, labels=False)
        xtest1,_,_ =test
        labels=np.zeros((len(xtest1),))
    # get labels
    def bary_wdl2(index): return _bary_wdl2(index, xtest1, feat, unfeat)
    
    n=xtest1.shape[-1]
    
    num_class = (int)(max(labels)+1)
    barys=[bary_wdl2(np.where(labels==i)) for i in range(num_class)]
    pl.figure(1, (num_class, 1))
    for i in range(num_class):
        pl.subplot(1,10,1+i)
        pl.imshow(barys[i][0,0,:,:],cmap='Blues',interpolation='nearest')
        pl.xticks(())
        pl.yticks(())
        if i==0:
            pl.ylabel('DWE Bary.')
        if num_class >1:
            pl.title('{}'.format(i))
    pl.tight_layout(pad=0,h_pad=-2,w_pad=-2) 
    pl.savefig("imgs/{}_dwe_bary.pdf".format(dataset_name))

#%% ACP

def get_PCA(dataset_name, feat, unfeat, repo, n_components=4, n_directions=3):
    
    if dataset_name==MNIST:
        _, _, test=get_data(dataset_name, repo, labels=True)
        xtest1,_,_, labels,_=test
    else:
        _, _, test=get_data(dataset_name, repo, labels=False)
        xtest1,_,_ =test
        labels=np.zeros((len(xtest1),))
    
    def bary_wdl2(index): return _bary_wdl2(index, xtest1, feat, unfeat)
    num_class = (int)(min(max(labels)+1, n_components))
    barys=[bary_wdl2(np.where(labels==i)) for i in range(num_class)]
    
    def medoid(index):
        embeddings = feat.predict(xtest1[np.where(labels==index)])
        return np.argmin([np.sum(( embeddings- feat.predict(barys[index]))**2)])
    
    # compute the medoids
    width=1.5
    nbv=5
    xv=np.linspace(-width,width,nbv)
    medoids=[medoid(i) for i in range(num_class)]
    xtemp = feat.predict(xtest1)
    x0=np.mean(xtemp,0)
    xtemp0=xtemp-x0.reshape((1,-1))
    C=np.cov(xtemp0.T)
    w,V=np.linalg.eig(C)
    #Ii=[[[0 for j in range(nbkeep)]for i in range(nbv) ] for c in num_class]
    pca_list=[[[] for i in range(nbv)] for c in range(num_class)]
    for j in range(n_directions):
        v1 = V[:,j].real
        s=np.sqrt(w[j].real)
        
        # compute principal directions and distort the medoid given those directions
        for i in range(nbv):
            for c in range(num_class):
                embedding_medoid = xtemp[medoids[c]]
                embedding_medoid = embedding_medoid.ravel()
                var=unfeat.predict((embedding_medoid+s*xv[i]*v1)[None])
                pca_list[c][i].append(var[0,0])
              
    pl.figure(2,(nbv, n_directions*nbv+1))
    pl.clf()
    for i in range(5):
        for j in range(n_directions):
            #print((n_directions,n_directions+n_directions*i))
            #pl.subplot(nbv,n_directions,1+n_directions+n_directions*i)
            #print(n_directions, nbv+1, 1+nbv+n_directions*i)
            pl.subplot(nbv, n_directions, 1+j+n_directions*i)
            pl.imshow(pca_list[0][i][j], cmap='Blues',interpolation='nearest')
            pl.xticks(())
            pl.yticks(())
            if i==0:
                pl.title('{} {}'.format('DWE',j+1))            
    pl.tight_layout()
    #pl.savefig("imgs/{}_wpca_{}.png".format(expe,c))
# TO DO: better subplot
#%%    
def bilinear_interpolation(dataset_name, feat, unfeat, repo, nbt=3):

    tlist=np.linspace(0,1,nbt)
    _, _, test=get_data(dataset_name, repo)
    xtest1, _, _ = test
    n = xtest1.shape[-1]
    N = len(xtest1)
    
    tuple_index=np.random.permutation(N)[:4]
    embeddings = feat.predict(xtest1[tuple_index])
    
    interp_array = np.zeros((nbt, nbt, n, n))
    
    for i in range(nbt):
        for j in range(nbt):
            x = (tlist[i]*embeddings[0] + (1-tlist[0])*embeddings[1])
            y = (tlist[i]*embeddings[2] + (1-tlist[0])*embeddings[3])
            x_interp = unfeat.predict(((tlist[j]*x + (1-tlist[j])*y))[None])
            interp_array[i,j]=x_interp[0,0]
            
    pl.figure(1)
    for i in range(nbt):
        for j in range(nbt):
            nb=i*nbt +j+1
            pl.subplot(nbt*100+(nbt)*10 +nb)
            pl.imshow(interp_array[i,j], cmap='Blues',interpolation='nearest')
            
            
            
#%%
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset_name', type=str, default='cat', help='dataset name')
    parser.add_argument('--method_name', type=str, default='INTERPOLATION', help='number of pairwise emd')
    parser.add_argument('--repo', type=str, default=REPO, help='number of iterations')
    
    args = parser.parse_args()                                                                                                                                                                                                                             
    dataset_name=args.dataset_name
    method_name=args.method_name
    repo=args.repo
    compute(method_name=method_name, dataset_name=dataset_name, repo=REPO)
    
        
    
    

