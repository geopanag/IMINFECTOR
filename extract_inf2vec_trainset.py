# -*- coding: utf-8 -*-
"""
Extract the propagation network of each cascade and the train set for Inf2vec
"""

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

    
def store_samples(fn,cascade_nodes,cascade_times,initiators,op_id, op_time,to_train_on,sampling_perc=120):
    """
    # Store the samples  for the train set as described in the node-context pair creation process for INFECTOR
    """
    #---- Inverse sampling based on copying time
    #cascade_nodes=cascade_nodes[1:]
    #cascade_times=cascade_times[1:]
    no_samples = round(len(cascade_nodes)*sampling_perc/100)
    if(fn=="weibo"):
      times = [1*1.0/(abs((cascade_times[i]-op_time).total_seconds())+1) for i in range(0,len(cascade_nodes))]
    else:  
      times = [1*1.0/(abs((cascade_times[i]-op_time))+1) for i in range(0,len(cascade_nodes))]
    s_times = sum(times)
    
    if s_times==0:
        samples = []	
    else:
        probs = [float(i)/s_times for i in times]
        samples = np.random.choice(a=cascade_nodes, size=int(no_samples), p=probs) 
    
    casc_len = str(len(cascade_nodes))
    
    #----- Store train set
    if(fn=="mag"):
        for op_id in initiators:    
            for i in samples:
                #---- Write inital node, copying node,length of cascade
                to_train_on.write(str(op_id)+","+i+","+casc_len+"\n")                	
    else:                
        for i in samples:
            #if(op_id!=i):# though this can t be 
                #---- Write initial node, copying node, copying time, length of cascade
            to_train_on.write(str(op_id)+","+i+","+casc_len+"\n")


def run_rwr(prop_net,restart = 0.5,path_size = 10):
    """
    # Run a Random Walk with restart to retreive a set of nodes for each node
    """
    train_set = {}
    for v in prop_net.vs:
        steps = 0
        rwr = []
        #---- RWR on v
        current = v
        while steps<path_size:
            steps+=1
            #-- Jump randomly in one of the neighbors
            neighs = prop_net.neighbors(current)
            if(len(neighs)==0):
                continue
            
            current = prop_net.vs[np.random.choice(neighs)]
            rwr.append(current["name"])
            if np.random.choice(2,p=[1-restart,restart]):
                current = v
            if(len(rwr)==5):
                break
        #----- Random sample
        train_set[v] = rwr+list(np.random.choice(prop_net.vs["name"],45))
            
    return train_set

            
def run(fn,sampling_perc,log):    
    print("Reading the network")
    g = ig.Graph.Read_Ncol(fn+"/"+fn+"_network.txt")
    
    #------ in mag it is undirected
    #if fn =="mag":
    #    g.to_undirected()
    start = time.time()  
    f = open(fn+"/train_cascades.txt","r")  
    
    inf2vec_set = open(fn+"/inf2vec_set.txt","a")
    #----- Initialize features
    idx = 0
    deleted_nodes = []
    g.vs["Cascades_participated"] = 0
    
    if(fn=="mag"):
        start_t = int(next(f))

    idx=0

    start = time.time()    
    #---------------------- Iterate through cascades to create the train set
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
            
            cascade_times = []
            cascade_nodes = []
            for p in papers:
                tim = int(p[-1])+start_t            
                for j in p[:-1]:
                    if j!="" and j!=" " and j not in cascade_nodes:
                        try:
							#----- to check if the node is in the network
                            g.vs.find(name=j)["Cascades_participated"]+=1
                        except:
                            continue
                        cascade_nodes.append(j)
                        cascade_times.append(tim)
                        
            #---- Define the propagation network
            prop_net = ig.Graph(directed=True)
            tmpt = initiators+cascade_nodes
            prop_net.add_vertices(tmpt)     
            #--- Draw edges between initiators and the rest
            for i in initiators:
                for j in cascade_nodes:
                    try:
                        #------ i is co authors with j
                        edge = g.get_eid(i,j)    
                        #------ i influences j
                        prop_net.add_edge(i,j)
                    except:
                        continue
            
            #--- Draw edges between cascade nodes
            for p in range(0,len(papers)):
                # if it takes more than 15 mins, go away)
                for i in papers[p]:
                   #---- Draw edges to the authors from the subsequent papers
                   for p2 in range(p+1,len(papers)):
                       for j in papers[p2]:
                           try:
                             #------ i is followed by j
                               edge = g.get_eid(i,j)    
                               #------ i influences j
                               prop_net.add_edge(i,j)
                           except:
                               continue
                                     
        else:
            initiators = []
            cascade = line.replace("\n","").split(";")
           
            if(fn=="weibo"):
                cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade[1:]))
                cascade_times = list(map(lambda x:  datetime.strptime(x.replace("\r","").split(" ")[1], '%Y-%m-%d-%H:%M:%S'),cascade[1:]))
            else:
                cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade))
                cascade_times = list(map(lambda x:  int(x.replace("\r","").split(" ")[1]),cascade))
            
            #---- Remove retweets by the same person in one cascade
            cascade_nodes, cascade_times = remove_duplicates(cascade_nodes,cascade_times)
            
            #---------- Dictionary nodes -> cascades
            op_id = cascade_nodes[0]
            op_time = cascade_times[0]

            if(len(cascade_nodes)<2):
                continue

            #---- Derive propagation network
            prop_net = ig.Graph(directed=True)

            prop_net.add_vertices(cascade_nodes)
            
			#---- Data-based weighing
            for i in range(len(cascade_nodes)): 
                # if it takes more than 15 mins, go away)
                #--- Add to the propagation graph all i's edges that point to j
                for j in range(i+1,len(cascade_nodes)):
                    try:
                        #------ i is followed by j
                        edge = g.get_eid(cascade_nodes[j],cascade_nodes[i])    
                    except:
                        continue
						
        pairs = run_rwr(prop_net)
        for node,rwr in pairs.iteritems():
            for r in rwr:
                inf2vec_set.write(node["name"]+","+str(r)+"\n")   
             
        idx+=1
        if(idx%1000==0):
            print("-------------------",idx)
    
    idx=0
    inf2vec_set.close()
    print("Number of nodes not found in the graph: ",len(deleted_nodes))
    f.close()
    
    log.write("inf2vec preprocess time "+fn+" "+str(time.time()-start)+"\n")
  