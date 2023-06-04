#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for HEATMAP plots

Created on Thur April 27 2023

@author: muziyu
"""

import shutil
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        
        
def ori_heatmap(wpth, gene=0, gname=None, plt_save=True, plt_show=False, dname='', s=3, figsize=(5,3)):    
    spatial = load(wpth+'/spa')
    if os.path.exists(wpth+'/xnorm.pkl'):
        xori = load(wpth+'/xnorm')
    else:
        xori = load(wpth+'/xsparse')
    fig, ax = plt.subplots(figsize=figsize)
    _= ax.set_facecolor('gray')
    _= ax.scatter(spatial[:,0],spatial[:,1],c=xori[:,gene],cmap='Blues',s=s)
    if gname is None:
        gname = 'gene'+str(gene)
    _= plt.title(gname+'-original'+dname)
    _= plt.xticks([])
    _= plt.yticks([])
    pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap')
    if plt_save:
        savepic(pic_pth, 'original-g'+str(gene))
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
        
def geneSelect_heatmap(wpth, w, dim, init='wdgsvd', sim='SPR', gene=0, gname=None, lamb=10, s=3,
                figsize=(9,4), plt_save=True, plt_show=False, dname='', sparse=None):
    
    
    file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(dim)
    spatial = load(wpth+'/spa')
    if gname is None:
        gname = 'gene'+str(gene)
    if sparse is None:
        res_pth = os.path.join(wpth, 'result', init, file, sim)
        figtit = gname+dname
    else:
        res_pth = os.path.join(wpth, str(sparse), 'result', init, file, sim)
        
        figtit = gname+'-'+str(sparse)+'%'+dname
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit)
    if os.path.exists(wpth+'/xnorm.pkl'):
        xori = load(wpth+'/xnorm')
    else:
        xori = load(wpth+'/xsparse')
    _= ax[0].set_title('original')
    _= ax[0].set_facecolor('gray')
    _= ax[0].scatter(spatial[:,0],spatial[:,1],c=xori[:,gene],cmap='Blues',s=s)
    xres = load(res_pth+'/x_'+sim)
    _= ax[1].set_title(sim+'-'+init+'-'+file)
    _= ax[1].set_facecolor('gray')
    _= ax[1].scatter(spatial[:,0],spatial[:,1],c=xres[:,gene],cmap='Blues',s=s)
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
                                   
    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap', 'geneSelect')
        savepic(pic_pth, sim+'-'+init+'-'+file+'-g'+str(gene))
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
        
def dataSPR_heatmap(wpth, gene=0, gname=None, plt_save=True, plt_show=False, dname='', s=3, figsize=(9,4)):
    spatial = load(wpth+'/spa')    
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    if gname is None:
        gname = 'gene'+str(gene)
    fig.suptitle(gname+dname)
    if os.path.exists(wpth+'/xnorm.pkl'):
        xsparse = load(wpth+'/xnorm')
    else:
        xsparse = load(wpth+'/xsparse')
    _= ax[0].set_title('sparse')
    _= ax[0].set_facecolor('gray')
    _= ax[0].scatter(spatial[:,0],spatial[:,1],c=xsparse[:,gene],cmap='Blues',s=s)
    xsprod = load(wpth+'/xsprod')
    _= ax[1].set_title('sprod')
    _= ax[1].set_facecolor('gray')
    _= ax[1].scatter(spatial[:,0],spatial[:,1],c=xsprod[:,gene],cmap='Blues',s=s)
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
                                   
    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap')
        savepic(pic_pth, 'dataSPR-g'+str(gene))
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
 
                                   
def paraCompare_heatmap(wpth, paracheck='w', paraset='d', checklist=[0.2, 0.5], 
                        setvalue=12, gene=0, gname=None, plt_save=True, plt_show=False, 
                        dname='', s=3, unitfigsz=(3,2.5), lamb=10,
                        sim='SPR', initial=['rand','nmf','nndsvd','wdgsvd','tcdsvd']):
                                   
    row = len(initial) #生成图行数
    col = len(checklist) #生成图行数
    spatial = load(wpth+'/spa')
    if gname is None:
        gname = 'gene'+str(gene)
    figtit = gname+'-'+sim+'-'+paraset+flo2str(setvalue)+dname                   
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit)
    j = 0
    for check in checklist:
        if paracheck=='w' and paraset=='d':
            file = 'w'+flo2str(check)+'-l'+flo2str(lamb)+'-d'+flo2str(setvalue)
        elif paracheck=='d' and paraset=='w':
            file = 'w'+flo2str(setvalue)+'-l'+flo2str(lamb)+'-d'+flo2str(check)
        i = 0
        for init in initial:
            res_pth = os.path.join(wpth, 'result', init, file, sim)
            xplot = load(res_pth+'/x_'+sim)
            if i == 0:
                t=ax[0][j].set_title(paracheck+str(check))
            if j == 0:
                t=ax[i][0].set_ylabel(init)
            t=ax[i][j].set_facecolor('gray')
            t=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
            i = i + 1
        j = j + 1
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
                                   
    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap', paracheck+'Compare', sim, str(gene))
        picname = paraset+flo2str(setvalue)+'-l'+flo2str(lamb)
        picname = picname.replace(".", ",")
        savepic(pic_pth, picname+'-g'+str(gene))
    if plt_show:
        fig.show()
    else:
        plt.close(fig)

        
def methods_heatmap(wpth, w, d, glist, gnlist=None, sparse=False, pic_pth=None, lamb=10, pad=0.93,
                plt_save=True, plt_show=False, dname='', savename='mth', s=3, unitfigsz=(3,2.5)):
    #original,sprod,dentist,wedge
    init = 'wdgsvd'
    sim = 'SPR'
    if sparse:
        col0tit = 'sparse'
    else:
        col0tit = 'original'
                                   
    row = len(glist) #生成图行数
    col = 4
    
    if lamb == 10:
        file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
        res_pth = os.path.join(wpth, 'result', init, file, sim)
    else:
        file = 'w'+flo2str(w)+'-d'+flo2str(d)
        res_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb))
                
    spatial = load(wpth+'/spa')     
    figsize = (col*unitfigsz[0], row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    if pic_pth is None:
        fig.suptitle('GeneHeatmap-'+dname, y=pad)
    else:
        fig.suptitle(dname, y=pad)
    
    if os.path.exists(wpth+'/xnorm.pkl'):
        xori = load(wpth+'/xnorm')
    else:
        xori = load(wpth+'/xsparse')
    xsprod = load(wpth+'/xsprod')
    xdent = load(res_pth+'/x_'+sim)
    xwedge = load(wpth+'/xwedge')
    
    _=ax[0][0].set_title(col0tit)
    _=ax[0][1].set_title('sprod')
    _=ax[0][2].set_title('dentist')
    _=ax[0][3].set_title('wedge')
                
    for j in range(col):
        i = 0
        for gene in glist:
            if j == 0:
                if gnlist is None:
                    _=ax[i][0].set_ylabel('g'+str(gene))
                else:
                    _=ax[i][0].set_ylabel(gnlist[i])
                _=ax[i][0].scatter(spatial[:,0],spatial[:,1],c=xori[:,gene],cmap='Blues',s=s)
            elif j == 1:
                _=ax[i][1].scatter(spatial[:,0],spatial[:,1],c=xsprod[:,gene],cmap='Blues',s=s)
            elif j == 2:
                _=ax[i][2].scatter(spatial[:,0],spatial[:,1],c=xdent[:,gene],cmap='Blues',s=s)
            else:
                _=ax[i][3].scatter(spatial[:,0],spatial[:,1],c=xwedge[:,gene],cmap='Blues',s=s)
            _=ax[i][j].set_facecolor('gray')
            i = i + 1
        
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
                                   
    if plt_save:
        if pic_pth is None:
            pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap')
            savepic(pic_pth, 'methods')
        else:
            pic_pth = os.path.join(pic_pth, 'paper', 'methods')
            savepic(pic_pth, savename+dname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
        
def simCompare_heatmap(wpth, w, dim, pic_pth, gene=0, gname=None, similarity=['SNN','SPR'], lamb=10, 
                       plt_save=True, plt_show=False, dname='', savename='sim,th', s=3, unitfigsz=(3,2.5),
                       pad=0.82, initial=['rand','nmf','nndsvd','wdgsvd','tcdsvd'], figsize=(3,2.5)):

    file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(dim)
    spatial = load(wpth+'/spa')
    
    row = 1 #生成图行数
    col = len(similarity)
    if gname is None:
        gname = 'gene'+str(gene)
    figtit = dname+'-'+gname                
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit, y=pad)
    
    res_pth = os.path.join(wpth, '..')
    xplot = load(res_pth+'/xnorm')
    t=ax[0].set_title('original')
    t=ax[0].set_facecolor('gray')
    t=ax[0].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
    
    res_pth = os.path.join(wpth)
    xplot = load(res_pth+'/xsparse')
    t=ax[1].set_title('sparse')
    t=ax[1].set_facecolor('gray')
    t=ax[1].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        simori_pth = os.path.join(pic_pth, 'paper', 'simori')
        savepic(simori_pth, savename+dname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
    row = len(initial) #生成图行数
    col = len(similarity)
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    j = 0
    for sim in similarity:
        i = 0
        for init in initial:
            res_pth = os.path.join(wpth, 'result', init, file, sim)
            xplot = load(res_pth+'/x_'+sim)
            if i == 0:
                t=ax[0][j].set_title(sim)
            if j == 0:
                t=ax[i][0].set_ylabel(init)
            t=ax[i][j].set_facecolor('gray')
            t=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
            i = i + 1
        j = j + 1
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        simmth_pth = os.path.join(pic_pth, 'paper', 'simmth')
        savepic(simmth_pth, savename+dname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
def sim_heatmap(wpth, w, dim, pic_pth, gene=0, gname=None, similarity=['SNN','SPR'], lamb=10, pad=0.9,
                plt_save=True, plt_show=False, init='wdgsvd', dname='', savename='sim', s=3, unitfigsz=(3,2.5)):

    file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(dim)
    spatial = load(wpth+'/spa')
    
    row = 2 #生成图行数
    col = len(similarity)
    if gname is None:
        gname = 'gene'+str(gene)
    figtit = dname+'-'+gname                
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit, y=pad)
    i = 0
    res_pth = os.path.join(wpth, '..')
    xplot = load(res_pth+'/xnorm')
    t=ax[i][0].set_title('original')
    t=ax[i][0].set_facecolor('gray')
    t=ax[i][0].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
    
    res_pth = os.path.join(wpth)
    xplot = load(res_pth+'/xsparse')
    t=ax[i][1].set_title('sparse')
    t=ax[i][1].set_facecolor('gray')
    t=ax[i][1].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
    i = 1
    j = 0
    for sim in similarity:
        res_pth = os.path.join(wpth, 'result', init, file, sim)
        xplot = load(res_pth+'/x_'+sim)
        t=ax[i][j].set_title(sim)
        t=ax[i][j].set_facecolor('gray')
        t=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
        j = j + 1
        
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        pic_pth = os.path.join(pic_pth, 'paper', 'sim')
        savepic(pic_pth, savename+dname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
    
        
def multiOri_heatmap(wpth, glist, gnlist=None, plt_save=True, plt_show=False,
                     dname='', s=3, unitfigsz=(3,2.5), pic_pth=None, pad=0.94):
                                   
    row = len(glist) #生成图行数

    spatial = load(wpth+'/spa')
    figsize = (unitfigsz[0], row*unitfigsz[1])
    fig, ax = plt.subplots(row, 1, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(dname, y=pad)
     
    if os.path.exists(wpth+'/xnorm.pkl'):
        xori = load(wpth+'/xnorm')
    else:
        xori = load(wpth+'/xsparse')
    #_=ax[0].set_title('original')
                
    i = 0
    for gene in glist:
        if gnlist is None:
            _=ax[i].set_ylabel('g'+str(gene))
        else:
            _=ax[i].set_ylabel(gnlist[i])
        _=ax[i].scatter(spatial[:,0],spatial[:,1],c=xori[:,gene],cmap='Blues',s=s)
        _=ax[i].set_facecolor('gray')
        i = i + 1
        
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        if pic_pth is None:
            pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap')
            savepic(pic_pth, 'multiOri')
        else:
            pic_pth = os.path.join(pic_pth, 'paper', 'mthori')
            savepic(pic_pth, dname)
    if plt_show:
        fig.show()
    else:
        plt.close(fig)
        
        
def lambCompare_heatmap(pth_dict, pic_pth, gene=0, gname=None, lambdas=[5,10,18], plt_save=True, 
                        plt_show=False, dname='', s=3, unitfigsz=(3,2.5)):
    
    row = len(lambdas) #生成图行数
    col = len(pth_dict)
    init='wdgsvd'
    sim='SPR'
    if gname is None:
        gname = 'gene'+str(gene)
    figtit = gname+'-'+'lambdaCompare'+'-'+dname                
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit)
    
    j = 0
    for key, value in pth_dict.items():
        wpth = value['pth']
        spatial = load(wpth+'/spa')
        w = value['w']
        d = value['d']
        i = 0
        for lamb in lambdas:
            if lamb == 10:
                file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                res_pth = os.path.join(wpth, 'result', init, file, sim)
            else:
                file = 'w'+flo2str(w)+'-d'+flo2str(d)
                res_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb))
            xplot = load(res_pth+'/x_'+sim)
            if i == 0:
                t=ax[0][j].set_title(key)
            if j == 0:
                t=ax[i][0].set_ylabel('lamb'+str(lamb))
            t=ax[i][j].set_facecolor('gray')
            t=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
            i = i + 1
        j = j + 1
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        pic_pth = os.path.join(pic_pth, 'plots', 'lambda', 'gene_heatmap')
        savepic(pic_pth, dname+'-g'+str(gene))
    if plt_show:
        plt.show()
    else:
        plt.close()


def sinlambCompare_heatmap(wpth, w, d, glist, gnlist=None, lambdas=[5,10,18], plt_save=True, 
                        plt_show=False, dname='', s=3, unitfigsz=(3,2.5)):
    
    col = len(lambdas)
    row = len(glist)
    init='wdgsvd'
    sim='SPR'
    figtit = 'GeneHeatmap-lambdaCompare'+'-'+dname                
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit)

    spatial = load(wpth+'/spa')

    j = 0
    for lamb in lambdas:
        i = 0
        if lamb == 10:
            file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
            res_pth = os.path.join(wpth, 'result', init, file, sim)
        else:
            file = 'w'+flo2str(w)+'-d'+flo2str(d)
            res_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb))
            xplot = load(res_pth+'/x_'+sim)
        for gene in glist:
            if j == 0:
                if gnlist is None:
                    _=ax[i][0].set_ylabel('g'+str(gene))
                else:
                    _=ax[i][0].set_ylabel(gnlist[i]) 
            if i == 0:
                t=ax[0][j].set_title('lamb'+str(lamb))
            _=ax[i][j].set_facecolor('gray')
            _=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
            i = i + 1
        j = j + 1
        
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'lambda')
        savepic(pic_pth, 'gene_heatmap')
    if plt_show:
        plt.show()
    else:
        plt.close()
        

def PolyGene_heatmap(wpth, w, d, polygene, plt_save=True, lamb=10, 
                    plt_show=False, dname='', ori=False, s=3, unitfigsz=(3,2.5)):
    
    row = len(polygene) #生成图行数
    col = 3
    init='wdgsvd'
    sim='SPR'
    spatial = load(wpth+'/spa')
    if ori:
        xplot = load(wpth+'/xnorm')
        xtit = 'original'
    else:
        if lamb == 10:
            file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
            res_pth = os.path.join(wpth, 'result', init, file, sim)
        else:
            file = 'w'+flo2str(w)+'-d'+flo2str(d)
            res_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb))
        xplot = load(res_pth+'/x_'+sim)
        xtit='dentist'

    figtit = 'PolygeneHeatmap-'+xtit+dname                
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit)
    
    i = 0
    for diction in polygene:
        glist = diction['glist']
        gnlist = diction['gnlist']
        j = 0
        for gene in glist:
            t=ax[i][j].set_title(gnlist[j])
            t=ax[i][j].set_facecolor('gray')
            t=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
            j = j + 1
        i = i + 1
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap')
        savepic(pic_pth, 'polygene-'+xtit)
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def MarkerGene_heatmap(wpth, w, d, markergene, plt_save=True, lamb=10, 
                    plt_show=False, dname='', s=3, ori=False, unitfigsz=(3,2.5)):
    
    row = len(markergene) #生成图行数
    col = 4
    init='wdgsvd'
    sim='SPR'
    spatial = load(wpth+'/spa')
    if ori:
        xplot = load(wpth+'/xnorm')
        xtit = 'original'
    else:
        if lamb == 10:
            file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
            res_pth = os.path.join(wpth, 'result', init, file, sim)
        else:
            file = 'w'+flo2str(w)+'-d'+flo2str(d)
            res_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb))
        xplot = load(res_pth+'/x_'+sim)
        xtit='dentist'

    figtit = 'MarkergeneHeatmap-'+xtit+dname                
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtit)
    
    i = 0
    for key, value in markergene.items():
        glist = value['glist']
        gnlist = value['gnlist']
        j = 0
        for gene in glist:
            if j == 0:
                _=ax[i][0].set_ylabel(key)
            _=ax[i][j].set_title(gnlist[j])
            _=ax[i][j].set_facecolor('gray')
            _=ax[i][j].scatter(spatial[:,0],spatial[:,1],c=xplot[:,gene],cmap='Blues',s=s)
            j = j + 1
        i = i + 1
    _= plt.xticks([])
    _= plt.yticks([])
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'gene_heatmap')
        savepic(pic_pth, 'markergene-'+xtit)
    if plt_show:
        plt.show()
    else:
        plt.close()