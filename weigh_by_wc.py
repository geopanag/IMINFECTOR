"""
Weigh all networks based on weighted cascade, and derive the attribute file required for IMM
"""

import pandas as pd
import numpy as np
import os
import json
import time

os.chdir("Path/To/Data")
log= open("time_log.txt","a")

for dat in ["weibo","digg","mag_cs"]:
    start = time.time()
    print(dat)
    #--- Read graph
    attribute = open(dat+"/wc_"+dat.replace("mag_","")+"_attribute.txt","w")
    graph = pd.read_csv(dat+"/"+dat.replace("mag_","")+"_network.txt",sep=" ")
    if(graph.shape[1]>2):
        graph = graph.drop(graph.columns[2],1)
    graph.columns = ["node1","node2"]

    #--- if it is undirected, turn it into directed
    if "mag" in dat:
        tmp = graph.copy()
        graph = pd.DataFrame(np.concatenate([graph.values, tmp[["node2","node1"]].values]),columns=graph.columns)
        del tmp

    #--- Compute influence weight
    outdegree = graph.groupby("node1").agg('count').reset_index()
    outdegree.columns = ["node1","outdegree"]
    
    outdegree["outdegree"] = 1/outdegree["outdegree"]
    outdegree["outdegree"] = outdegree["outdegree"].apply(lambda x:float('%s' % float('%.6f' % x)))
    
    #--- Assign it
    graph = graph.merge(outdegree, on="node1")
    
    al = list(set(graph["node1"].unique()).union(set(graph["node2"].unique()))) 
    
    dic = {al[i]:i for i in range(0,len(al))}
    graph['node1'] = graph['node1'].map(dic)
    graph['node2'] = graph['node2'].map(dic)
    
    f= open(dat+"/"+dat.replace("mag_","")+"_node_dic.json","w")
    #f.write(json.dumps(dic))
    json.dump(dic,f)
    f.close()
    
    #--- Store
    graph = graph[["node2","node1","outdegree"]]
    graph.to_csv(dat+"/wc_"+dat.replace("mag_","")+"_network.txt",header=False,index=False, sep=" ")
    log.write("Time for wc "+dat+" network:"+str(time.time()-start)+"\n")
    
    attribute.write("n="+str(len(al)+1)+"\n")
    attribute.write("m="+str(graph.shape[0])+"\n")
    attribute.close()

log.close()
