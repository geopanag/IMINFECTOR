# -*- coding: utf-8 -*-
"""
@author: georg
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import sparse
import random
import numpy as np
import json


def embedding_matrix(embedding_file,embed_dim,var):
    print(embed_dim)

    size = embed_dim[1]
    emb = np.zeros((size,50), dtype=np.float)
    nodes = []
    f = open(embedding_file,"r")
    i=0

    for l in f:
        if "[" in l:
            combined = ""
        if "]" in l:
            combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
            parts = combined.split(":")
            nodes.append(int(parts[0]))
            x =  np.asarray([float(p.strip()) for p in parts[1].split(" ") if p!=""],dtype=np.float)
            emb[i] = x#np.asarray([float(p.strip()) for p in parts[1].split(" ") if p!=""],dtype=np.float)
            i+=1
        combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
    return nodes, emb
  
def fix_file(file_fix):
    #f = codecs.open(file_fix, "r",encoding='utf-8', errors='ignore')
    f = open(file_fix,encoding="utf8")
    f_new = open("new_"+file_fix,"w")
    for l in f:
        if("]" not in l):
            f_new.write(l.replace("\n",""))
        else:
            #print(l)
            f_new.write(l)
    f_new.close()
    f.close()


def softmax_(x):
    return np.exp(x)/np.sum(np.exp(x))
    
def load_initiator(file_Sn,file_Tn,P,perc, embed_dim):
    """
    Load the embeddings of inf2vec+
    """
    nodes_emb, S = embedding_matrix(file_Sn,embed_dim,"S")
    _, T = embedding_matrix(file_Tn,embed_dim,"T")
    #nodes_emb = pd.DataFrame(nodes_emb,columns=["nodes"])
    
    #feats = pd.read_csv(fn+"/node_features_"+str(perc)+".csv")
    feats = pd.read_csv(fn+"/node_features.csv")
    P = P*embed_dim[0]/100
    #print(P)
    #feats["cascs"] = feats["Cascades_participated"]+feats["Cascades_started"]
    chosen = feats.nlargest(P,"Cascades_started")["Node"].values #Cascades_participated
    chosen_ = []
    for c in chosen:
        try:
            chosen_.append(np.where(nodes_emb==c)[0][0])
        except:
            continue
    #chosen_ = [np.where(nodes_emb==c)[0][0] for c in chosen]
    S = S[chosen_]   
    ILM = np.dot(S,T.T)    
    ILM = np.apply_along_axis(softmax_, 1, ILM)    
    return ILM, chosen_,nodes_emb
    



def compute_influence_spread(influence_set):
    """
    Given the set of influenced nodes in each simulation, compute the total influenced spread
    """
    return np.sum([len(i) for i in influence_set])
    

    
def compute_influence_set(influenced_set):
    """
    The length of the cumulative influence set from all simulations
    """
    return len(set.union(*influenced_set))
    

    
def marginal_gain(ILM,candidate,influenced,no_simulations,edge_samples):
    """
    For eah simulation, sample edges of the seed based on their probability and compute the 
    joint influence set with the ones sampled in the respective simulations up from the rest of the seeds
    """    
    for i in range(no_simulations):   
        idx = np.random.choice(range(ILM.shape[1]),edge_samples,p=ILM[candidate,:],replace=False)
        influenced[i][idx] = 1
        #influenced[idx,i] = 1
    return influenced
    
    
    

"""
Main
"""
os.chdir("path/to/data")

log = open("../time_log.txt","a")    
for fn in ["digg","weibo","mag"]:#,
    #----------------- Target node dictionary
    f = open(fn+"/"+fn+"_node_dic.json","r")
    dic_out = json.load(f)
    target_size = len(dic_out)

    top_p = 1
    perc = 80
    
    start = time.time()
    print(perc)
    f = open(fn+"/"+fn+"_sizes_"+str(perc)+".txt","r")
    target_size = int(next(f))
    input_size = int(next(f))
    f.close()

    embed_dim = [input_size,target_size]
    #----- Iterate using the nodeswith the best feature

    file_Sn = fn+"/embeddings/source_"+str(perc)+"_embeddings.txt"
    file_Tn = fn+"/embeddings/target_"+str(perc)+"_embeddings.txt"

    ILM, chosen, nodes_emb =  load_initiator(file_Sn,file_Tn,top_p,perc,embed_dim)
    ILM = np.apply_along_axis(softmax_, 1, ILM)   
    #--------------------------------------- Run 
    seed_set = []
    if(fn=="weibo"):
        size = 1000
    elif(fn=="mag"):
        size = 10000
    else:
        size = 50
        
    # sample 1000 edges from each source node for each simulation
    edge_samples = 1000
    no_simulations = 50

    Q = [] 
    S = []
    #S = [0,521,545,586,791,827,922]

    nid = 0
    mg = 1
    iteration = 2

    #tmp =open("../tmp.txt","w")
    #----------------- Since we draw 50000 edges from each their total spread will always be the same
    spr = no_simulations*edge_samples
    for u in range(ILM.shape[0]):
        temp_l = []
        #value = marginal_gain(ILM,u,seed_set_spread,no_simulations)
        temp_l.append(u)		
        #spr = compute_influence_spread(influenced)
        temp_l.append(spr)
        #tmp.write(str(u)+" " +str(spr)+"\n")
        temp_l.append(0) #iteration
        Q.append(temp_l)

    #Q = sorted (Q, key=lambda x:x[1],reverse=True)
    #nodes_emb[chosen[0]]
    print("done first iteration")

    #----- Celf
    #------------------- First computation of the marginal gain for all nodes
    seed_set_influenced = 0
    infl_spread = 0

    influenced_set = [np.zeros(embed_dim[1]) for i in range(no_simulations)]

    fw =open(fn+"/seeds_final/celfie_seeds_fin.txt","w+")

    idx=0

    while len(S) < size :
        try:
            u = Q[0]
        except:
            break
        if (u[iteration] == len(S)):
            print(nodes_emb[chosen[u[nid]]])
            t = time.time()
            influenced_set = marginal_gain(ILM,u[nid],influenced_set[:],no_simulations,edge_samples)
            print(time.time()-t)
            infl_spread = np.sum(influenced_set)

            #----- Store the new seed
            try:
                fw.write(str(nodes_emb[chosen[u[nid]]])+"\n")
            
                S.append(u[nid])

                #----- Delete uid
                Q = [l for l in Q if l[0] != u[nid]]
            except:
                break
        else:
            #----- Update this node
            #------- Keep only the number of nodes influenceed to rank the candidate seed        
            influenced = marginal_gain(ILM,u[nid],influenced_set[:],no_simulations,edge_samples)
            #value = marginal_gain_ind(ILM,u[nid],seed_set_spread,no_simulations)#max(enumerate([len(final_cascade.union(set(casc))) for casc in cascades]), key=operator.itemgetter(1))
            u[mg] = np.sum(influenced)-infl_spread
            if(u[mg]<0):
                print("Something is wrong")
            u[iteration] = len(S)
            Q = sorted(Q, key=lambda x:x[1],reverse=True)
        idx+=1
        #if(idx%100==0):
        #    print("Still here...")

    fw.close()
    log.write(fn+" celfie : "+str(time.time()-start)+"\n")
    print(time.time()-start)
log.close()

      
                 
