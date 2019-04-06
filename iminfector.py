# -*- coding: utf-8 -*-
"""
@author: george

IMINFECT
"""
import pandas as pd
import numpy as np
import os
import json
import time

#----- Algorithm
def infl_set(ILM,candidate,size,uninfected):
    return np.argpartition(ILM[candidate,uninfected],-size)[-size:]

def infl_spread(ILM,candidate,size,uninfected):
    return sum(np.partition(ILM[candidate,uninfected], -size)[-size:])
	
def embedding_matrix(embedding_file,embed_dim):
    """
    Derive the matrix embeddings vector from the file
    """
    nodes = []
    f = open(embedding_file,"r")
    emb = np.zeros((embed_dim[0],embed_dim[1]), dtype=np.float)
    i=0
    for l in f:
        if "[" in l:
            combined = ""
        if "]" in l:
            combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
            parts = combined.split(":")
            nodes.append(int(parts[0]))
            emb[i] = np.asarray([float(p.strip()) for p in parts[1].split(" ") if p!=""],dtype=np.float)
            i+=1
        combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
    return nodes, emb


"""
Main
"""
embeddings_size=50
st = time.time()    
os.chdir("Path/To/Data")

for fn in ["digg","weibo","mag_cs"]:
    f = open(fn+"/train_set.txt","r")
    initiators = []
    mi = np.inf
    ma = 0
    for l in f:
        parts  = l.split(",")
        initiators.append(parts[0])

    initiators = np.unique((initiators))
    dic_in = {initiators[i]:i for i in range(0,len(initiators))}
    f.close()        
    del initiators

    input_size = len(dic_in)
    del dic_in

    #----------------- Target node dictionary
    f = open(fn+"/"+fn.replace("mag_","")+"_node_dic.json","r")
    dic_out = json.load(f)
    target_size = len(dic_out) 
    del dic_out	

    file_Sn = fn+"/embeddings/mtl_n_source_embeddings_p.txt"
    file_Tn = fn+"/embeddings/mtl_n_target_embeddings_p.txt"

    nodes_idx, T = embedding_matrix(file_Tn, [target_size,embeddings_size])
    init_idx, S = embedding_matrix(file_Sn, [input_size,embeddings_size])
    
    print("ready")
    # Compute bins
    perc = int(10*S.shape[0]/100)
    norm = np.apply_along_axis(lambda x: sum(x**2),1,S)
    chosen = np.argsort(-norm)[0:perc]
    norm = norm[chosen]
    bins = target_size*norm/sum(norm)
    bins = np.rint(bins)
    S = S[chosen] 
    
    ILM = np.dot(np.around(S,4),np.around(T.T,4))  
    del nodes_idx, T, S, norm    
    ILM = np.apply_along_axis(lambda x:x-abs(max(x)), 1, ILM) 
    ILM = np.around(ILM,3)
    #ILM.sort(axis=1)
    ILM = abs(ILM)
    #np.save(fn+"/ILM", ILM )
    #np.savetxt(fn+"/ILM.csv", ILM.T,fmt='%.3f',delimiter=",")
    #init_idx = np.load(fn+"/init_idx.npy")
    #chosen = np.load(fn+"/chosen.npy")
    #bins = list(np.load(fn+"/bins.npy"))
    bins = [int(i) for i in list(bins)]
    ILM = np.load(fn+"/ILM.npy")
            
    if(fn=="Digg"):
        size=100
    elif(fn=="weibo"):
        size=1000
    else:
        size=10000
    
    Q = []
    S = []   
    nid = 0
    mg = 1
    iteration = 2
    infed = np.zeros(ILM.shape[1])
    total = set([i for i in range(ILM.shape[1])])
    uninfected = list(total-set(np.where(infed)[0]))
    
    #----- Initialization
    for u in range(ILM.shape[0]):
        temp_l = []
        #value = marginal_gain(ILM,u,seed_set_spread,no_simulations)
        temp_l.append(u)
        spr = infl_spread(ILM,u,bins[u],uninfected)    
        temp_l.append(spr)
        #tmp.write(str(u)+" " +str(spr)+"\n")
        temp_l.append(0) #iteration
        Q.append(temp_l)
    # Do not sort
   
    ftp = open(fn+"/seeds/final_tmp_seeds.txt","w")  
    idx = 0
    while len(S) < size :
        u = Q[0]
        new_s = u[nid]
        if (u[iteration] == len(S)):
            influenced = infl_set(ILM,new_s,bins[new_s],uninfected)   
            infed[influenced]  = 1         
            uninfected = list(total-set(np.where(infed)[0]))
            
            #----- Store the new seed
            ftp.write(str(init_idx[chosen[new_s]])+"\n")
            S.append(new_s)
            if(len(S)%50==0):
                print(len(S))
            #----- Delete uid
            Q = [l for l in Q if l[0] != new_s]
        else:
            #------- Keep only the number of nodes influenceed to rank the candidate seed        
            spr = infl_spread(ILM,new_s,bins[new_s],uninfected)        
            #influenced = marginal_gain(ILM,u[nid],influenced_set.copy(),no_simulations,edge_samples)
            u[mg] = spr
            if(u[mg]<0):
                print("Something is wrong")
            u[iteration] = len(S)
            Q = sorted(Q, key=lambda x:x[1],reverse=True)
        
    ftp.close()
        
    ft = open(fn+"/seeds/final_seeds.txt","w")   
    for s in S:
        ft.write(str(init_idx[chosen[s]])+" ")
    ft.close()

print(time.time()-st)
