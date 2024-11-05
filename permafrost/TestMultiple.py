#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:18:45 2023

@author: duval
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as st


data = pd.read_csv("borehole_Samoylov_byday.csv") #pd.read_csv("borehole_grenoble_AdM_NW.csv") 


data.isna().sum()
data = data.dropna()



depth = data.drop(['Date.Depth'], axis = 1)
day = data['Date.Depth']
year = np.zeros_like(day)
j = 0
for i in day.index:
    year[j] = day[i][0:4]
    j+=1


x = depth.columns

import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, depth.shape[1]))

#figure = plt.figure()
#axes = figure.add_subplot(111)
for index in range(depth.shape[1]):
    plt.plot(np.arange(day.shape[0]), 
             depth.iloc[:,index], 
             c = colors[index],
             label = x[index],
                 linewidth = 4)
    #axes.xaxis.set_ticks(day)
    plt.title('Tracé de températures à différentes profondeurs')
    plt.xlabel('Date')
    plt.xticks(rotation = '90')
    plt.ylabel('Température')
    plt.legend(prop = {'size': 7})
    
    
    
Ind = depth>0
Y = np.unique(year)
AggY = np.zeros((Y.shape[0], depth.shape[1]))
j=0
for i in Y:
    AggY[j,:] = np.array(Ind.iloc[year == i,]).mean(axis=0)
    j+=1

AggY
plt.plot(Y, AggY.mean(axis=1), '*')


#Les huit années les plus chaudes ont toutes été enregistrées depuis 2015, 
#les années 2016, 2019 et 2020 arrivant en tête du classement. 
#L'année 2016 a été marquée par un épisode El Niño d'une intensité exceptionnelle, 
#qui a contribué à des températures records à l'échelle mondiale



# Test température par profondeur entre les années


# Faire une boucle qui regarde le signe de la statistique de test et si la p-valeur rejette égalité


p = 23# profondeur

Res = np.zeros((Y.shape[0],Y.shape[0]))
ii, jj =0, 0
for i in Y:
    for j  in Y:
        Z = depth[year == i][x[p]]
        X = depth[year == j][x[p]]
        fvalue, pvalue = st.ttest_ind(X,Z)
        if pvalue > 0.05/16**2:# On fait une correction de Bonferroni, 
            # On peut aussi faire  Benjamini- Hochberg
            Res[ii,jj] = 0
        else:
            if fvalue > 0:
                Res[ii,jj] = 1
            else:
                Res[ii,jj] = -1
        jj+=1
    ii+=1
    jj = 0
            
plt.matshow(Res)
plt.colorbar()

    
