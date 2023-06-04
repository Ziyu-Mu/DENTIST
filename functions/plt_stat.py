#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for STATISTICAL VALUE plots

Created on Thur April 27 2023

@author: muziyu
"""

import shutil
import pickle
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        

def wXsparsity_scatter(indicator, pth_dict, pic_pth, checklist=[0.2,0.5,0.7],
                        dname='', s=100, lamb=10, plt_show=False, plt_save=True, 
                        initial=['rand','nmf','nndsvd','wdgsvd','tcdsvd']):
    # indicator = ['cmd_gene', 'cmd_cell']
    sim = 'SPR'
    paracheck='w'
    color2 = ['#FFA500','#1047A9']
    color3 = ['#FFE500','#FF8E00','#1921B1']
    color4 = ['#FFBB00','#FF8C00','#0969A2','#1924B1']
    clength = len(checklist)
    marker2 = ['^', 'o']
    marker3 = ['^', 'o', '*']
    mlength = len(pth_dict)
    if mlength == 2:
        marker = marker2
        if clength == 2:
            color = color2
            xdeviate = [-0.6, -0.2, 0.2, 0.6]
        elif clength == 3:
            color = color3
            xdeviate = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
        elif clength == 4:
            color = color4
            xdeviate = [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
        else:
            raise ValueError('can only compare limited number of parameters')
    elif mlength == 3:
        marker = marker3
        if clength == 2:
            color = color2
            xdeviate = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
        elif clength == 3:
            color = color3
            xdeviate = [-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6]
        else:
            raise ValueError('can only compare limited number of parameters')
    else:
        raise ValueError('can only compare limited number of parameters')
            
    map_xaxis = {}
    map_color = {}
    map_marker = {}
    map_xaxis_m = {}
    i = 0
    for init in initial:
        map_xaxis[init] = 2 * i + 1
        i = i + 1
    xrlim =  2 * i
    i = 0
    for check in checklist:
        map_color[check] = color[i]
        i = i + 1
    i = 0
    for data in pth_dict.keys():
        map_marker[data] = marker[i]
        pd_v = xdeviate[clength*i: clength*(i+1)]
        para_deviate = {}
        j = 0
        for check in checklist:
            para_deviate[check] = pd_v[j]
            j = j + 1
        map_xaxis_m[data] = para_deviate
        i = i + 1

    df = pd.DataFrame()
    indi_v = []
    init_v = []
    check_v = []
    data_v = []
    
    for data, value in pth_dict.items():
        wpth = value['pth']
        dim = value['d']
        for check in checklist:
            file = 'w'+flo2str(check)+'-l'+flo2str(lamb)+'-d'+flo2str(dim)
            for init in initial:
                res_pth = os.path.join(wpth, 'result', init, file)
                stat_pth = os.path.join(res_pth, sim, 'statistics')
                indi_v.append(load(stat_pth+'/'+indicator+'_'+sim))
                init_v.append(init)
                check_v.append(check)
                data_v.append(data)
    df[indicator] = indi_v
    df['initial'] = init_v
    df[paracheck] = check_v
    df['data'] = data_v
    df['data-para'] = df[['data',paracheck]].values.tolist()
    
    df_plot = df.copy()
    x1 = np.array(list(map(lambda x: map_xaxis[x], df_plot['initial'])))
    x2 = np.array(list(map(lambda x: map_xaxis_m[x[0]][x[1]], df_plot['data-para'])))
    df_plot['xaxis'] = x1 + x2
    df_plot['maker'] = list(map(lambda x: map_marker[x], df_plot['data']))
    df_plot['color'] = list(map(lambda x: map_color[x], df_plot[paracheck]))
    
    patch = []
    for value in map_marker.values():
        p = plt.scatter([20], [indi_v[0]], color='black', marker=value, s=100)
        patch.append(p)
    lgd_m = plt.legend(patch, pth_dict.keys(), loc=3, bbox_to_anchor=(1., 0.6))
    for data in pth_dict.keys():
        legend = []
        label = []
        for check in checklist:
            label.append(data+'-'+paracheck+str(check))
            df_sub = df_plot.loc[df_plot['data']==data]
            df_sub = df_sub.loc[df_sub[paracheck]==check]
            plt.scatter(df_sub['xaxis'], df_sub[indicator], 
                                   color=map_color[check], marker=map_marker[data], s=s)
    patch = []
    for key, value in map_color.items():
        p = mpatches.Patch(color=value, label=paracheck+str(key))
        patch.append(p)
    lgd_c = plt.legend(handles=patch, loc=3, bbox_to_anchor=(1.,0.3))
    _= plt.gca().add_artist(lgd_m)
    
    _=plt.grid(False)
    for vline in range(1, len(initial)):
        plt.axvline(2*vline,ls='--',c="gray")#添加垂直直线
    _= plt.xlim([0,xrlim])
    
    tick = ['rand','nmf','nnd','wdg','tcd']
    _= plt.xticks(list(map_xaxis.values()), tick)
    _= plt.title(indicator+'-'+dname)
    
    if plt_save:
        pic_path = os.path.join(pic_pth, 'paper', 'choose_w', dname)
        savepic(pic_path, indicator)
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def statCombine(pth_dict, cmdhigh, pcclim, pic_pth, lamb=10, dname='',
                plt_show=False, plt_save=True, unitfigsz=(4,3), width=0.8, pad=0.89):
    indis = ['cmd','pcc']
    relation = ['cell','gene']
    row = len(indis) #生成图行数
    col = len(relation) #生成图行数
    
    init='wdgsvd'
    sim='SPR'
    figsize = (col*unitfigsz[0],row*unitfigsz[1])
    fig, ax = plt.subplots(row, col, figsize=figsize, sharex=True, sharey='row')
    fig.suptitle(dname, y=pad)
    j = 0
    for rel in relation:
        i = 0
        for intd in indis:
            indicator = intd+'_'+rel
            if i == 0:
                t=ax[0][j].set_title(rel)
                t=ax[0][j].set_ylim([0,cmdhigh])
            else:
                t=ax[1][j].set_ylim(pcclim)
            if i == 0:
                indi_v = []
                mthd_v = []
                data_v = []
                order = ['norm', 'sprod', 'dentist', 'wedge']
                for key, value in pth_dict.items():
                    wpth = value['pth']
                    w = value['w']
                    d = value['d']
                    for method in order:
                        if method == 'norm':
                            if os.path.exists(wpth+'/xnorm.pkl'):
                                xtit = 'original'
                                stat_pth = os.path.join(wpth, 'statistics', 'norm')
                                affix = 'norm'
                            else:
                                xtit = 'sparse'
                                stat_pth = os.path.join(wpth, 'statistics', 'sparse')
                                affix = 'sparse'
                        elif method == 'dentist':
                            file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                            stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
                            affix = sim
                        else:
                            stat_pth = os.path.join(wpth, 'statistics', method)
                            affix = method
    
                        indivalue = load(stat_pth+'/'+indicator+'_'+affix)
                        indi_v.append(indivalue)
                        mthd_v.append(method)
                        data_v.append(key)
    
                dfc = pd.DataFrame()
                dfc[indicator] = indi_v
                dfc['method'] = mthd_v
                dfc['data'] = data_v
                dfc.loc[dfc['method']=='norm','method']=xtit
                order[0] = xtit
    
                bp = sns.barplot(x='data', y=indicator, data=dfc, hue_order=order,ax=ax[i][j],
                                 hue='method', palette='Set2', width=width)
                if j == 0:
                    _= ax[i][0].legend('',frameon=False)
                else:
                    _= ax[i][1].legend(loc=3, bbox_to_anchor=(1., 0.2))
                _= ax[i][j].set_xlabel('')
                #_= ax[i][j].xticks(rotation=20)
                #bp.figure.set_size_inches(figsize[0],figsize[1])
            else:
                df_list = []
                order = ['norm', 'sprod', 'dentist', 'wedge']
                for key, value in pth_dict.items():
                    wpth = value['pth']
                    w = value['w']
                    d = value['d']
                    for method in order:
                        if method == 'norm':
                            if os.path.exists(wpth+'/xnorm.pkl'):
                                xtit = 'original'
                                stat_pth = os.path.join(wpth, 'statistics', 'norm')
                                affix = 'norm'
                            else:
                                xtit = 'sparse'
                                stat_pth = os.path.join(wpth, 'statistics', 'sparse')
                                affix = 'sparse'
                        elif method == 'dentist':
                            file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                            stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
                            affix = sim
                        else:
                            stat_pth = os.path.join(wpth, 'statistics', method)
                            affix = method
    
                        indivalue = load(stat_pth+'/'+indicator+'_'+affix)
                        df_sub = pd.DataFrame()
                        df_sub[indicator] = indivalue
                        df_sub['method'] = method
                        df_sub['data'] = key
                        df_list.append(df_sub)
            
                dfp = pd.concat(df_list)
                dfp.reset_index(drop=True, inplace=True)
                dfp.loc[dfp['method']=='norm','method']=xtit
                order[0] = xtit
    
                bp = sns.boxplot(x='data', y=indicator, data=dfp, hue='method', hue_order=order, 
                            palette='Set2', linewidth=1, showfliers=False, ax=ax[i][j])
                if j == 0:
                    _= ax[i][0].legend('',frameon=False)
                else:
                    _= ax[i][1].legend(loc=3, bbox_to_anchor=(1., 0.2))
                _= ax[i][j].set_xlabel('')
                #_= ax[i][j].set_xticks(rotation=20)
                #bp.figure.set_size_inches(figsize[0],figsize[1])
            if j == 0:
                t=ax[0][0].set_ylabel('CMD')
                t=ax[1][0].set_ylabel('PCC')
            else:
                t=ax[i][j].set_ylabel('')
            i = i + 1
        j = j + 1
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    
    if plt_save:
        pic_pth = os.path.join(pic_pth, 'paper', 'statcombine')
        savepic(pic_pth, dname)
    if plt_show:
        plt.show()
    else:
        plt.close()
    
def initPARAsim_scatter(wpth, indicator, paracheck='w', paraset='d', checklist=[0.2,0.5,0.7], 
                    setvalue=7, dname='', s=200, lamb=10, plt_show=False, plt_save=True, 
                     similarity=['SNN','SPR'], initial=['rand','nmf','fac','nndsvd','wegsvd','tcdsvd']):
    # indicator = ['cmd_gene', 'cmd_cell']

    color2 = ['#FFA500','#1047A9']
    color3 = ['#FFE500','#FF8E00','#1921B1']
    color4 = ['#FFBB00','#FF8C00','#0969A2','#1924B1']
    clength = len(checklist)
    marker2 = ['^', 'o']
    marker3 = ['^', 'o', '*']
    mlength = len(similarity)
    if mlength == 2:
        marker = marker2
        if clength == 2:
            color = color2
            xdeviate = [-0.6, -0.2, 0.2, 0.6]
        elif clength == 3:
            color = color3
            xdeviate = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
        elif clength == 4:
            color = color4
            xdeviate = [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
        else:
            raise ValueError('can only compare limited number of parameters')
    elif mlength == 3:
        marker = marker3
        if clength == 2:
            color = color2
            xdeviate = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
        elif clength == 3:
            color = color3
            xdeviate = [-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6]
        else:
            raise ValueError('can only compare limited number of parameters')
    else:
        raise ValueError('can only compare limited number of parameters')
            
    map_xaxis = {}
    map_color = {}
    map_marker = {}
    map_xaxis_m = {}
    i = 0
    for init in initial:
        map_xaxis[init] = 2 * i + 1
        i = i + 1
    xrlim =  2 * i
    i = 0
    for check in checklist:
        map_color[check] = color[i]
        i = i + 1
    i = 0
    for sim in similarity:
        map_marker[sim] = marker[i]
        pd_v = xdeviate[clength*i: clength*(i+1)]
        para_deviate = {}
        j = 0
        for check in checklist:
            para_deviate[check] = pd_v[j]
            j = j + 1
        map_xaxis_m[sim] = para_deviate
        i = i + 1

    df = pd.DataFrame()
    indi_v = []
    init_v = []
    check_v = []
    simi_v = []
    for check in checklist:
        if paracheck=='w' and paraset=='d':
            file = 'w'+flo2str(check)+'-l'+flo2str(lamb)+'-d'+flo2str(setvalue)
        elif paracheck=='d' and paraset=='w':
            file = 'w'+flo2str(setvalue)+'-l'+flo2str(lamb)+'-d'+flo2str(check)
        i = 0
        for init in initial:
            res_pth = os.path.join(wpth, 'result', init, file)
            for sim in similarity:
                stat_pth = os.path.join(res_pth, sim, 'statistics')
                indi_v.append(load(stat_pth+'/'+indicator+'_'+sim))
                init_v.append(init)
                check_v.append(check)
                simi_v.append(sim)
    df[indicator] = indi_v
    df['initial'] = init_v
    df[paracheck] = check_v
    df['similarity'] = simi_v
    df['sim-para'] = df[['similarity',paracheck]].values.tolist()
    
    df_plot = df.copy()
    x1 = np.array(list(map(lambda x: map_xaxis[x], df_plot['initial'])))
    x2 = np.array(list(map(lambda x: map_xaxis_m[x[0]][x[1]], df_plot['sim-para'])))
    df_plot['xaxis'] = x1 + x2
    df_plot['maker'] = list(map(lambda x: map_marker[x], df_plot['similarity']))
    df_plot['color'] = list(map(lambda x: map_color[x], df_plot[paracheck]))
    
    patch = []
    for value in map_marker.values():
        p = plt.scatter([20], [indi_v[0]], color='black', marker=value, s=150)
        patch.append(p)
    lgd_m = plt.legend(patch, similarity, loc=3, bbox_to_anchor=(1., 0.6))
    for sim in similarity:
        legend = []
        label = []
        for check in checklist:
            label.append(sim+'-'+paracheck+str(check))
            df_sub = df_plot.loc[df_plot['similarity']==sim]
            df_sub = df_sub.loc[df_sub[paracheck]==check]
            plt.scatter(df_sub['xaxis'], df_sub[indicator], 
                                   color=map_color[check], marker=map_marker[sim], s=s)
    patch = []
    for key, value in map_color.items():
        p = mpatches.Patch(color=value, label=paracheck+str(key))
        patch.append(p)
    lgd_c = plt.legend(handles=patch, loc=3, bbox_to_anchor=(1.,0.3))
    _= plt.gca().add_artist(lgd_m)
    
    _=plt.grid(False)
    for vline in range(1, len(initial)):
        plt.axvline(2*vline,ls='--',c="gray")#添加垂直直线
    _= plt.xlim([0,xrlim])
    
    low = min(indi_v)
    high = max(indi_v)
    if low < -0.5:
        ylow = -1.05
    elif low < 0:
        ylow = -0.55
    elif low < 0.5:
        ylow = -0.05
    else:
        ylow = 0.45
        
    if high > 0.5:
        yhigh = 1.05
    elif high > 0:
        yhigh = 0.55
    elif high > -0.5:
        yhigh = 0.05
    else:
        yhigh = -0.45
    _= plt.ylim([ylow, yhigh])
    
    _= plt.xticks(list(map_xaxis.values()), initial, rotation=30)
    pictit = paraset+flo2str(setvalue)+'-l'+flo2str(lamb)
    _= plt.title(indicator+'-'+pictit+dname)
    
    if plt_save:
        pic_path = os.path.join(wpth, 'plots', 'statistics', indicator, paracheck+'Compare')
        picname = pictit.replace(".", ",")
        savepic(pic_path, picname)
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def paraCompare_boxplot(wpth, indicator, paracheck='w', paraset='d', checklist=[0.2,0.5,0.7], 
                    setvalue=7, dname='', lamb=10, plt_show=False,  plt_save=True, 
                     sim='SPR', initial=['rand','nmf','nndsvd','wdgsvd','tcdsvd']):
    # indicator = ['pcc_cell', 'pcc_gene']

    df_list = []
    for check in checklist:
        if paracheck=='w' and paraset=='d':
            file = 'w'+flo2str(check)+'-l'+flo2str(lamb)+'-d'+flo2str(setvalue)
        elif paracheck=='d' and paraset=='w':
            file = 'w'+flo2str(setvalue)+'-l'+flo2str(lamb)+'-d'+flo2str(check)
        for init in initial:
            df_sub = pd.DataFrame()
            stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
            pcc = load(stat_pth+'/'+indicator+'_'+sim)
            df_sub = pd.DataFrame()
            df_sub[indicator] = pcc
            df_sub['initial'] = init
            df_sub[paracheck] = paracheck+str(check)
            df_list.append(df_sub)
    df = pd.concat(df_list)
    df.reset_index(drop=True, inplace=True)
    order = list(map(lambda x:paracheck+str(x), checklist))
    bp = sns.boxplot(x=paracheck, y=indicator, data=df, order=order, hue='initial', 
                hue_order=initial, palette='Set2', linewidth=1, showfliers=False)
    _= plt.legend(loc=3, bbox_to_anchor=(1., 0.2))
    _= plt.ylabel('')
    _= plt.xlabel('')

    if bp.get_ylim()[0] < -0.5:
        ylow = -1
    elif bp.get_ylim()[0] < 0:
        ylow = -0.5
    elif bp.get_ylim()[0] < 0.5:
        ylow = 0
    else:
        ylow = 0.5
        
    if bp.get_ylim()[1] > 0.5:
        yhigh = 1
    elif bp.get_ylim()[1] > 0:
        yhigh = 0.5
    elif bp.get_ylim()[1] > -0.5:
        yhigh = 0
    else:
        yhigh = -0.5
    _= plt.ylim([ylow, yhigh])
         
    pictit = sim+'-'+paraset+flo2str(setvalue)+'-l'+flo2str(lamb)
    _= plt.title(indicator+'-'+pictit+dname)

    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'statistics', indicator, paracheck+'Compare', sim)
        picname = pictit.replace(".", ",")
        savepic(pic_pth, picname)
    if plt_show:
        plt.show()
    else:
        plt.close()

        
def time_barplot(wpth, dname='', figsize=(5,2.5), width=0.8, plt_show=False, plt_save=True):
    # indicator = ['pcc_cell', 'pcc_gene']
    runTime = load(wpth+'/time')
    time_v = []
    init_v = []
    simi_v = []
    for key, value in runTime.items():
        for subkey, subvalue in value.items():
            time_v.append(subvalue)
            init_v.append(subkey)
            simi_v.append(key)
            
    df = pd.DataFrame()
    df['time'] = time_v
    df['initial'] = init_v
    df['similarity'] = simi_v
    
    similarity = list(runTime.keys())
    initial = list(value.keys())
    bp = sns.barplot(x='initial', y='time', data=df, order=initial, hue='similarity', 
                hue_order=similarity, palette='Set2', width=0.8)
    _= plt.legend(loc=3, bbox_to_anchor=(1., 0.2))
    _= plt.ylabel('')
    _= plt.xlabel('')
    _= plt.title('average runTime'+dname)
    bp.figure.set_size_inches(figsize[0],figsize[1])

    if plt_save:
        pic_pth = os.path.join(wpth, 'plots', 'statistics')
        savepic(pic_pth, 'runTime')
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def lambCompare_boxplot(pth_dict, indicator, pic_pth, lambdas=[5,10,18],
                    dname='', plt_show=False,  plt_save=True):
    # indicator = ['pcc_cell', 'pcc_gene']
    init='wdgsvd'
    sim='SPR'
    df_list = []
    for key, value in pth_dict.items():
        wpth = value['pth']
        w = value['w']
        d = value['d']
        for lamb in lambdas:
            if lamb == 10:
                file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
            else:
                file = 'w'+flo2str(w)+'-d'+flo2str(d)
                stat_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb), 'statistics')
            
            df_sub = pd.DataFrame()
            pcc = load(stat_pth+'/'+indicator+'_'+sim)
            df_sub = pd.DataFrame()
            df_sub[indicator] = pcc
            df_sub['lambda'] = lamb
            df_sub['data'] = key
            df_list.append(df_sub)
    df = pd.concat(df_list)
    df.reset_index(drop=True, inplace=True)
    bp = sns.boxplot(x='data', y=indicator, data=df, hue='lambda', 
                palette='Set2', linewidth=1, showfliers=False)
    _= plt.legend(loc=3, bbox_to_anchor=(1., 0.2))
    _= plt.ylabel('')
    _= plt.xlabel('')

    if bp.get_ylim()[0] < -0.5:
        ylow = -1.05
    elif bp.get_ylim()[0] < 0:
        ylow = -0.55
    elif bp.get_ylim()[0] < 0.5:
        ylow = -0.05
    else:
        ylow = 0.45
        
    if bp.get_ylim()[1] > 0.5:
        yhigh = 1.05
    elif bp.get_ylim()[1] > 0:
        yhigh = 0.55
    elif bp.get_ylim()[1] > -0.5:
        yhigh = 0.05
    else:
        yhigh = -0.45
        
    _= plt.ylim([ylow, yhigh])
    _= plt.title(indicator+'-'+dname)
    if plt_save:
        pic_pth = os.path.join(pic_pth, 'plots', 'lambda', 'statistics')
        savepic(pic_pth, indicator)
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def lambCompare_barplot(pth_dict, indicator, pic_pth, lambdas=[5,10,18], width=0.8,
                        figsize=(3,2.8), dname='', plt_show=False,  plt_save=True):
    # indicator = ['pcc_cell', 'pcc_gene']
    init='wdgsvd'
    sim='SPR'
    indi_v = []
    lamb_v = []
    data_v = []
    for key, value in pth_dict.items():
        wpth = value['pth']
        w = value['w']
        d = value['d']
        for lamb in lambdas:
            if lamb == 10:
                file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
            else:
                file = 'w'+flo2str(w)+'-d'+flo2str(d)
                stat_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb), 'statistics')
            
            df_sub = pd.DataFrame()
            cmd = load(stat_pth+'/'+indicator+'_'+sim)
            indi_v.append(cmd)
            lamb_v.append(lamb)
            data_v.append(key)
    
    df = pd.DataFrame()
    df[indicator] = indi_v
    df['lambda'] = lamb_v
    df['data'] = data_v
    
    bp = sns.barplot(x='data', y=indicator, data=df, hue='lambda', palette='Set2', width=width)
    _= plt.legend(loc=3, bbox_to_anchor=(1., 0.2))
    _= plt.ylabel('')
    _= plt.xlabel('')
    _= plt.title(indicator+'-'+dname)
    bp.figure.set_size_inches(figsize[0],figsize[1])

    if plt_save:
        pic_pth = os.path.join(pic_pth, 'plots', 'lambda', 'statistics')
        savepic(pic_pth, indicator)
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def methodCompare_barplot(pth_dict, indicator, pic_pth, width=0.8, dname='', 
                plt_save=True, lamb=10, plt_show=False, figsize=(3,2.8)):
   
    # indicator = ['pcc_cell', 'pcc_gene']
    init = 'wdgsvd'
    sim = 'SPR'

    indi_v = []
    mthd_v = []
    data_v = []
    order = ['norm', 'sprod', 'dentist', 'wedge']
    for key, value in pth_dict.items():
        wpth = value['pth']
        w = value['w']
        d = value['d']
        for method in order:
            if method == 'norm':
                if os.path.exists(wpth+'/xnorm.pkl'):
                    xtit = 'original'
                    stat_pth = os.path.join(wpth, 'statistics', 'norm')
                    affix = 'norm'
                else:
                    xtit = 'sparse'
                    stat_pth = os.path.join(wpth, 'statistics', 'sparse')
                    affix = 'sparse'
            elif method == 'dentist':
                if lamb == 10:
                    file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                    stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
                else:
                    file = 'w'+flo2str(w)+'-d'+flo2str(d)
                    stat_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb), 'statistics')
                affix = sim
            else:
                stat_pth = os.path.join(wpth, 'statistics', method)
                affix = method
    
            indivalue = load(stat_pth+'/'+indicator+'_'+affix)
            indi_v.append(indivalue)
            mthd_v.append(method)
            data_v.append(key)
    
    df = pd.DataFrame()
    df[indicator] = indi_v
    df['method'] = mthd_v
    df['data'] = data_v
    df.loc[df['method']=='norm','method']=xtit
    order[0] = xtit
    
    bp = sns.barplot(x='data', y=indicator, data=df, hue_order=order, hue='method', palette='Set2', width=width)
    _= plt.legend(loc=3, bbox_to_anchor=(1., 0.2))
    _= plt.ylabel('')
    _= plt.xlabel('')
    #_= plt.xticks(rotation=20)
    _= plt.title(indicator+dname)
    bp.figure.set_size_inches(figsize[0],figsize[1])

    if plt_save:
        pic_pth = os.path.join(pic_pth, 'paper', 'statistics')
        savepic(pic_pth, indicator)
    if plt_show:
        plt.show()
    else:
        plt.close()
        
        
def methodCompare_boxplot(pth_dict, indicator, pic_pth, width=0.8, dname='', 
                plt_save=True, lamb=10, plt_show=False, figsize=(3,2.8)):
   
    # indicator = ['pcc_cell', 'pcc_gene']
    init = 'wdgsvd'
    sim = 'SPR'

    df_list = []
    order = ['norm', 'sprod', 'dentist', 'wedge']
    for key, value in pth_dict.items():
        wpth = value['pth']
        w = value['w']
        d = value['d']
        for method in order:
            if method == 'norm':
                if os.path.exists(wpth+'/xnorm.pkl'):
                    xtit = 'original'
                    stat_pth = os.path.join(wpth, 'statistics', 'norm')
                    affix = 'norm'
                else:
                    xtit = 'sparse'
                    stat_pth = os.path.join(wpth, 'statistics', 'sparse')
                    affix = 'sparse'
            elif method == 'dentist':
                if lamb == 10:
                    file = 'w'+flo2str(w)+'-l'+flo2str(lamb)+'-d'+flo2str(d)
                    stat_pth = os.path.join(wpth, 'result', init, file, sim, 'statistics')
                else:
                    file = 'w'+flo2str(w)+'-d'+flo2str(d)
                    stat_pth = os.path.join(wpth, 'result', file, 'l'+flo2str(lamb), 'statistics')
                affix = sim
            else:
                stat_pth = os.path.join(wpth, 'statistics', method)
                affix = method
    
            indivalue = load(stat_pth+'/'+indicator+'_'+affix)
            df_sub = pd.DataFrame()
            df_sub[indicator] = indivalue
            df_sub['method'] = method
            df_sub['data'] = key
            df_list.append(df_sub)
            
    df = pd.concat(df_list)
    df.reset_index(drop=True, inplace=True)
    df.loc[df['method']=='norm','method']=xtit
    order[0] = xtit
    
    bp = sns.boxplot(x='data', y=indicator, data=df, hue='method', hue_order=order, 
                palette='Set2', linewidth=1, showfliers=False)
    _= plt.legend(loc=3, bbox_to_anchor=(1., 0.2))
    _= plt.ylabel('')
    _= plt.xlabel('')
    #_= plt.xticks(rotation=20)
    _= plt.title(indicator+dname)
    bp.figure.set_size_inches(figsize[0],figsize[1])

    if plt_save:
        pic_pth = os.path.join(pic_pth, 'paper', 'statistics')
        savepic(pic_pth, indicator)
    if plt_show:
        plt.show()
    else:
        plt.close()