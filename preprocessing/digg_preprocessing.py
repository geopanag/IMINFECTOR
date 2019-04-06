# -*- coding: utf-8 -*-
"""
@author: george

Clean Digg and create cascades
"""

import os
import pandas as pd
import networkx as nx
import numpy as np

os.chdir("Path/To/Init_Data")

friends = pd.read_csv("digg_friends.csv",header=None)

#--------- Remove self friendships
friends = friends[friends[2]!=friends[3]]

#--------- Repeat the reciprocal edges and append them
reciprocal = friends[friends[0]==1]
friends = friends.drop(0,1)
reciprocal = reciprocal.drop(0,1)

#---- Create the reciprocal edge for each pair
tmp = reciprocal[2].copy()
reciprocal[2] = reciprocal[3]
reciprocal[3] = tmp


#--------- Find the edges that already exist in the dataset as reciprocal, and remove them, 
#--------- to avoid overwriting the currect time of the reciprocal edges that already exist
to_remove = reciprocal.reset_index().merge(friends,left_on=[2,3],right_on=[2,3]).set_index('index').index
reciprocal = reciprocal.drop(to_remove)

friends = friends.append(reciprocal)
friends[friends.duplicated([2,3],keep=False)] #-- this should be empty

#----------- Store the weighted follow network
friends.columns = ["time","a","b"]
friends = friends[["a","b","time"]]
friends.to_csv("../digg_network.txt",index=False,sep=" ",header=False)


#----------- Derive and store the train and test cascades
votes = pd.read_csv("digg_votes1.csv",header=None)
votes.columns = ["time","user","post"]
votes = votes.sort_values(by=["time"])

#---- Find the threshold after which the cascades are test cascades (final 20% of cascades)
start_times = votes.groupby("post")["time"].min() #--- take into consideration only the starting time of each cascade
start_times = start_times.sort_values()
no_test_cascades = round(20*len(start_times)/100)
threshold = min(start_times.tail(no_test_cascades))
sum(start_times<threshold )/start_times.shape[0]


f_train = open("digg_train_cascades.txt","w")
f_test = open("digg_test_cascades.txt","w")

#--------- For each cascade
for i in votes["post"].unique():
    print(i)
    sub = votes[votes["post"]==i]
    s = ""

    #---- id:time, id:time etc...
    for post in sub.sort_values(by=['time']).iterrows():
        s = s+str(post[1]["user"])+" "+str(post[1]["time"])+";"#":"+str(post[1]["time"])+","
    s = s[:-1]

    #---- Check if it has started before or after the threshold
    if(min(sub["time"])<threshold):
        f_train.write(s+"\n")
    else:
        f_test.write(s+"\n")

f_train.close()
f_test.close()



