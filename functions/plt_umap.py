#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for UMAP plots

Created on Thur April 27 2023

@author: muziyu
"""

import shutil
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps


def save(var, name):
    if os.path.exists(name+'.pkl'):
        os.remove(name+'.pkl')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(var, f)
        
        
def load(name):
    if not os.path.exists(name+'.pkl'):
        raise ValueError(name)
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)
    

def flo2str(n): #删除小数点后多余的0并转换为字符
    n = str(n)
    if '.' in n:
        n = n.rstrip('0')  # 删除小数点后多余的0
        if n.endswith('.'):
            n = n.rstrip('.')
    return n


def savepic(pth, name):
    if not os.path.exists(pth):
        os.makedirs(pth)
    pic = pth+'/'+name
    if os.path.exists(pic+'.png'):
        os.remove(pic+'.png')
    plt.savefig(pic, bbox_inches='tight') 


def res_umap(wpth, w, d, init, sim, lamb=10, dname='', s=6, cmap='Set3', 
            colorbar=True, plt_save=True, plt_show=False):
 
    if colorbar:
        figsize=(10,8)
    else:
        figsize=(8,8)
        
    oriClus = load(wpth+'/oriClus')
    file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
    res_pth = os.path.join(wpth, 'result', init, file, sim)
    umap_pth = os.path.join(res_pth, 'umap')
    stat_pth = os.path.join(res_pth, 'statistics')
    
    ari = round(load(stat_pth+'/ari_'+sim),3)
    jcrd = round(load(stat_pth+'/jcrd_'+sim),3)
    xumap = load(umap_pth+'/xumap')
    umap = pd.DataFrame(xumap, columns=['UMAP-1','UMAP-2'])
    umap['Cluster'] = oriClus
    fig, ax = plt.subplots()
    ax.set_facecolor('#393F4C')
    _= umap.plot.scatter(x='UMAP-1', y='UMAP-2', c='Cluster', ax=ax, 
                         colorbar=colorbar, figsize=figsize, colormap=cmap, s=s)
    _= plt.title('UMAP-'+file+dname)
    _= plt.xticks([])
    _= plt.yticks([])
    
    if plt_save:
        picname = 'a'+str(ari)+'j'+str(jcrd)
        picname = picname.replace(".", ",")
        savepic(umap_pth, picname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
    
def unit_umap(wpth, method='norm', dname='', s=6, cmap='Set3', 
            colorbar=True, plt_save=True, plt_show=False):
    '''
    pkl: sprod, norm, wedge, sparse
    ''' 
    if method == 'norm':
        pname = 'original'
    else:
        pname = method
    if colorbar:
        figsize=(10,8)
    else:
        figsize=(8,8)
        
    oriClus = load(wpth+'/oriClus')
    umap_pth = os.path.join(wpth, 'statistics', method, 'umap')
    stat_pth = os.path.join(wpth, 'statistics', method)
    ari = round(load(stat_pth+'/ari_'+method),3)
    jcrd = round(load(stat_pth+'/jcrd_'+method),3)
    xumap = load(umap_pth+'/xumap')
    umap = pd.DataFrame(xumap, columns=['UMAP-1','UMAP-2'])
    umap['Cluster'] = oriClus
    fig, ax = plt.subplots()
    ax.set_facecolor('#393F4C')
    _= umap.plot.scatter(x='UMAP-1', y='UMAP-2', c='Cluster', ax=ax, 
                         colorbar=colorbar, figsize=figsize, colormap=cmap, s=s)
    _= plt.title('UMAP-'+pname+dname)
    _= plt.xticks([])
    _= plt.yticks([])
    
    if plt_save:
        picname = 'a'+str(ari)+'j'+str(jcrd)
        picname = picname.replace(".", ",")
        savepic(umap_pth, picname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
        
def lamb_umap(wpth, w, d, lamb, dname='', s=6, cmap='Set3', 
            colorbar=True, plt_save=True, plt_show=False):
    init = 'wdgsvd'
    sim = 'SPR'
    if colorbar:
        figsize=(10,8)
    else:
        figsize=(8,8)
        
    file = 'w'+flo2str(w)+'-d'+flo2str(d)
    res_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb))    
    oriClus = load(wpth+'/oriClus')
    umap_pth = os.path.join(res_pth, 'umap')
    stat_pth = os.path.join(res_pth, 'statistics')
    
    ari = round(load(stat_pth+'/ari_'+sim),3)
    jcrd = round(load(stat_pth+'/jcrd_'+sim),3)
    xumap = load(umap_pth+'/xumap')
    umap = pd.DataFrame(xumap, columns=['UMAP-1','UMAP-2'])
    umap['Cluster'] = oriClus
    fig, ax = plt.subplots()
    ax.set_facecolor('#393F4C')
    _= umap.plot.scatter(x='UMAP-1', y='UMAP-2', c='Cluster', ax=ax, 
                         colorbar=colorbar, figsize=figsize, colormap=cmap, s=s)
    _= plt.title('UMAP-lamb'+str(lamb)+dname)
    _= plt.xticks([])
    _= plt.yticks([])
    
    if plt_save:
        picname = 'a'+str(ari)+'j'+str(jcrd)
        picname = picname.replace(".", ",")
        savepic(umap_pth, picname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)