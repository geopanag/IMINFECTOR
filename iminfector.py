# -*- coding: utf-8 -*-
"""
@author: george
"""
import pandas as pd
import numpy as np
import os
import json
import time

def softmax_(x):
        return np.exp(x)/np.sum(np.exp(x))

class IMINFECTOR:
    def __init__(self, fn, embedding_size,seed_set_size):
        self.fn=fn
        self.embedding_size = embedding_size
        self.num_samples = num_samples
        self.file_Sn = fn+"/embeddings/infector_source.txt"
        self.file_Tn = fn+"/embeddings/infector_target.txt"
        if(fn=="Digg"):
            self.size=100
        elif(fn=="weibo"):
            self.size=1000
        else:
            self.size=10000
            
    def infl_set(candidate,size,uninfected):
        return np.argpartition(self.D[candidate,uninfected],-size)[-size:]
    
    def infl_spread(candidate,size,uninfected):
        return sum(np.partition(self.D[candidate,uninfected], -size)[-size:])
    	
    def embedding_matrix(var):
        """
        Derive the matrix embeddings vector from the file
        """
        if var=="T":
            embedding_file= self.file_Tn
            embed_dim = [self.target_size,self.embedding_size]
        else:
            embedding_file= self.file_Sn
            embed_dim = [self.input_size,self.embedding_size]

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
        
    def create_dicts(self):
        """
        # Min max normalization of cascade length and source-target dictionaries
        """
        initiators = []
        self.mi = np.inf
        self.ma = 0
        for l in f:
            parts  = l.split(",")
            initiators.append(parts[0])
        initiators = np.unique((initiators))
        dic_in = {initiators[i]:i for i in range(0,len(initiators))}
        f.close()        
        
        self.input_size = len(dic_in)
        
        #----------------- Target node dictionary
        f = open(self.fn+"/"+self.fn+"_incr_dic.json","r")
        dic_out = json.load(f)
        self.target_size = len(dic_out)   

    def compute_D(self,S,T,nodes_idx,init_idx):
        """
        # Derive matrix D and vector E
        """
        perc = int(10*S.shape[0]/100)
        norm = np.apply_along_axis(lambda x: sum(x**2),1,S)
        self.chosen = np.argsort(-norm)[0:perc]
        norm = norm[chosen]
        E = self.target_size*norm/sum(norm)
        self.E = np.rint(E)
        self.E = [int(i) for i in list(self.E)]
        np.save(fn+"/E", self.E)
        S = S[chosen] 
        
        self.D = np.dot(np.around(S,4),np.around(T.T,4))  
    
    def process_D(self):
        """
        # Derive the diffusion probabilities. Had to be separated with compute_D, beause of memory
        """
        self.D = np.apply_along_axis(lambda x:x-abs(max(x)), 1, self.D) 
        self.D = np.apply_along_axis(softmax, 1, self.D) 
        self.D = np.around(self.D,3)
        self.D = abs(self.D)
        np.save(fn+"/D")
     
    def run_method():
        """
        # IMINFECTOR algorithm
        """
        Q = []
        self.S = []   
        nid = 0
        mg = 1
        iteration = 2
        infed = np.zeros(self.D.shape[1])
        total = set([i for i in range(self.D.shape[1])])
        uninfected = list(total-set(np.where(infed)[0]))
        
        #----- Initialization
        for u in range(self.D.shape[0]):
            temp_l = []
            temp_l.append(u)
            spr = self.infl_spread(self.D,u,bins[u],uninfected)    
            temp_l.append(spr)
            temp_l.append(0)
            Q.append(temp_l)
    		
        # Do not sort
        ftp = open(fn+"/seeds/final_tmp_seeds.txt","w")  
        idx = 0
        while len(self.S) < size :
            u = Q[0]
            new_s = u[nid]
            if (u[iteration] == len(self.S)):
                influenced = self.infl_set(self.D,new_s,bins[new_s],uninfected)   
                infed[influenced]  = 1         
                uninfected = list(total-set(np.where(infed)[0]))
                
                #----- Store the new seed
                ftp.write(str(init_idx[chosen[new_s]])+"\n")
                self.S.append(new_s)
                if(len(self.S)%50==0):
                    print(len(self.S))
                #----- Delete uid
                Q = [l for l in Q if l[0] != new_s]
    			
            else:
                #------- Keep only the number of nodes influenceed to rank the candidate seed        
                spr = self.infl_spread(self.D,new_s,bins[new_s],uninfected)        
                u[mg] = spr
                if(u[mg]<0):
                    print("Something is wrong")
                u[iteration] = len(self.S)
                Q = sorted(Q, key=lambda x:x[1],reverse=True)
            
        ftp.close()
            
        
def run(fn,embedding_size ,log):
    f = open(fn+"/train_set.txt","r")
    start = time.time()
    infector = IMINFECTOR(fn,learning_rate,n_epochs,embedding_size,num_neg_samples)

    iminfector.create_dicts()
    
    nodes_idx, T = infector.embedding_matrix("T")
    init_idx, S = infector.embedding_matrix("S")
    
    iminfector.compute_D(S,T,nodes_idx,init_idx)
    del T,S,nodes_idx
    iminfector.process_D()
    iminfector.run_method()
    
        

