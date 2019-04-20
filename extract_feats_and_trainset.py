# -*- coding: utf-8 -*-
"""
@author: georg

Extract summary features about the participation of nodes in the cascades
Compute kcore and avg cascade length
Etract the train set for INFECTOR
"""

import os
import igraph as ig
import time
import pandas as pd
import numpy as np
from datetime import datetime


def sort_papers(papers):
    """
    # Sort MAG diffusion cascade, which is a list of papers and their authors, in the order the paper'sdate
    """
    x =list(map(int,list(map(lambda x:x.split()[-1],papers))))
    return [papers[i].strip() for i in np.argsort(x)]
            
            
def remove_duplicates(cascade_nodes,cascade_times):
    """
    # Some tweets have more then one retweets from the same person
    # Keep only the first retweet of that person
    """
    duplicates = set([x for x in cascade_nodes if cascade_nodes.count(x)>1])
    for d in duplicates:
        to_remove = [v for v,b in enumerate(cascade_nodes) if b==d][1:]
        cascade_nodes= [b for v,b in enumerate(cascade_nodes) if v not in to_remove]
        cascade_times= [b for v,b in enumerate(cascade_times) if v not in to_remove]

    return cascade_nodes, cascade_times

    
def store_samples(fn,cascade_nodes,cascade_times,op_time,sampling_perc=120):
    """
    # Store the samples  for the train set as described in the node-context pair creation process for INFECTOR
    """
    #---- Inverse sampling based on copying time
    no_samples = round(len(cascade_nodes)*sampling_perc/100)
    times = [op_time/(abs((cascade_times[i]-op_time))+1) for i in range(0,len(cascade_nodes))]
    s_times = sum(times)
    if s_times==0:
        samples = []	
    else:
        probs = [float(i)/s_times for i in times]
        samples = np.random.choice(a=cascade_nodes, size=int(no_samples), p=probs) 
    
    casc_len = str(len(cascade_nodes))
    
    to_train_on = open(fn+"/train_set.txt","w")
    #----- Store train set
    if(fn=="mag"):
        for op_id in initiators:    
            for i in samples:
                #---- Write inital node, copying node,length of cascade
                to_train_on.write(str(op_id)+","+i+","+casc_len+"\n")                	
    else:                
        for i in samples:
            #---- Write initial node, copying node, copying time, length of cascade
            to_train_on.write(str(op_id)+","+i+","+casc_len+"\n")


            
if __name__ == '__main__':
    start = time.time()
    os.chdir("Path/To/Data")
    
    sampling_perc = 120
    log= open("time_log.txt","a")
    
    for fn in ["weibo","digg","mag"]:
        print(fn)
        print("Reading the network")
        g = ig.Graph.Read_Ncol(fn+"/"+fn+"_network.txt")
        # in mag it is undirected
        if fn =="mag":
            g.to_undirected()
            
        f = open(fn+"/train_cascades.txt")  
        
        #----- Initialize features
        idx = 0
        deleted_nodes = []
        g.vs["Cascades_started"] = 0
        g.vs["Cumsize_cascades_started"] = 0
        g.vs["Cascades_participated"] = 0
        log.write(" net:"+fn+"\n")
        start_t = int(f.next())
        
        times = []
        idx=0
    
        start = time.time()    
        #---------------------- Iterate through cascades to crate the train set
        for line in f:
            if(fn=="mag"):
                parts = line.split(";")
                initiators = parts[0].replace(",","").split(" ")
                op_time = int(initiators[-1])+start_t
                initiators = initiators[:-1]
                papers = parts[1].replace("\n","").split(":")
                papers = sort_papers(papers)
                papers = [list(map(lambda x: x.replace(",",""),i)) for i in list(map(lambda x:x.split(" "),papers))]
                
                #---- Extract the authors from the paper list
                flatten = []
                for i in papers:
                    flatten = flatten+i[:-1]
                u,i = np.unique(flatten,return_index=True)
                cascade_nodes = list(u[np.argsort(i)])
                
                #--- Update metrics of initiators
                for op_id in initiators:
                    try:
                        g.vs.find(name=op_id)["Cascades_started"]+=1
                        g.vs.find(name=op_id)["Cumsize_cascades_started"]+=len(papers)
                    except:
                        continue
                    
                cascade_times = []
                cascade_nodes = []
                for p in papers:
                    tim = int(p[-1])+start_t            
                    for j in p[:-1]:
                        if j!="" and j!=" " and j not in cascade_nodes:
                            try:
                                g.vs.find(name=j)["Cascades_participated"]+=1
                            except:
                                continue
                            cascade_nodes.append(j)
                            cascade_times.append(tim)
                            
            else:
                cascade = line.replace("\n","").split(";")
                cascade_nodes = map(lambda x:  x.split(" ")[0],cascade[1:])
                if(fn=="weibo"):
                    cascade_times = map(lambda x:  datetime.strptime(x.replace("\r","").split(" ")[1], '%Y-%m-%d-%H:%M:%S'),cascade[1:])
                else:
                    cascade_times = map(lambda x:  x.replace("\r","").split(" ")[1],cascade[1:])
                
                #---- Remove retweets by the same person in one cascade
                cascade_nodes, cascade_times = remove_duplicates(cascade_nodes,cascade_times)
                
                #---------- Dictionary nodes -> cascades
                op_id = cascade_nodes[0]
                op_time = cascade_times[0]

                #---------- Update metrics
                try:
                    g.vs.find(name=op_id)["Cascades_started"]+=1
                    g.vs.find(op_id)["Cumsize_cascades_started"]+=len(cascade_nodes)
                except: 
                    deleted_nodes.append(op_id)
                    continue
                
                if(len(cascade_nodes)<2):
                    continue
    
            store_samples(fn,cascade_nodes,cascade_times,op_time,sampling_perc=120)
                        
            idx+=1
            if(idx%1000==0):
                print("-------------------",idx)
            
        print("Number of nodes not found in the graph: ",len(deleted_nodes))
        f.close()
        
        log.write("Training time:"+str(time.time()-start)+"\n")
        
        start = time.time()
        kcores = g.shell_index()
        log.write("K-core time:"+str(time.time()-start)+"\n")
        
        #------ Store node charateristics
        pd.DataFrame({"Node":g.vs["name"],
                      "Kcores":kcores,
                      "Participated":g.vs["Cascades_participated"],
        			    "Avg_Cascade_Size": g.vs["Cumsize_cascades_started"]/g.vs["Cascades_started"]}).to_csv(fn+"/node_features.csv",index=False)

