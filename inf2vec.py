# -*- coding: utf-8 -*-
"""
Train on inf2vec and derive the network
"""

# -*- coding: utf-8 -*-
import os
import time
import math  
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import json
import igraph as ig

def embedding_matrix(file_,voc_size,embed_size):
    """
    Derive the matrix embeddings vector from the file
    """
    embedding_file= file_
    f = open(file_,"r")
    list_ = {}
    for l in f:
        if "[" in l:
            combined = ""
        if "]" in l:
            combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
            parts = combined.split(":")
            #nodes.append(int(parts[0]))
            list_[parts[0].replace(" ","")] =np.asarray([float(p.strip()) for p in parts[1].split(" ") if p!=""],dtype=np.float) 
        combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
    return list_

num_samples = 10

def run(fn,log,learning_rate,n_epochs,embedding_size,num_samples):    
    print(fn)
    start = time.time()
    file_Sn = fn+"/embeddings/inf2vec_source_embeddings.txt"
    file_Tn = fn+"/embeddings/inf2vec_target_embeddings.txt"
    
    #----------------- Node name - node incr id dictionary
    f = open(fn+"/"+fn+"_node_dic.json","r")
    dic = json.load(f)
    target_size = len(dic)
    vocabulary_size = len(dic)
    #f = open(fn+"/"+fn+"_sizes.txt","r")
    #target_size = int(next(f).strip())
    #source_size = int(next(f).strip())
    #f.close()
    #------- Same graph as original
    #----------------- Define the model with biases
    graph1 = tf.Graph()
    with graph1.as_default():
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)   
        
        u = tf.placeholder(tf.int32, shape=[batch_size,1])
        v = tf.placeholder(tf.int32, shape=[batch_size,1]) 
        
        #------------ Source (hidden layer embeddings)
        S = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="S")
        
        u2 =  tf.squeeze(u)
        Su = tf.nn.embedding_lookup(S,u2)
        
        #------------ Target (hidden and output weights)
        T = tf.Variable(tf.truncated_normal( [target_size,embedding_size], stddev = 1.0 / math.sqrt(embedding_size)), name="T")
        
        #---- Noise contrastive loss function
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss1 = tf.reduce_mean(tf.nn.nce_loss(weights= T,
                 biases=nce_biases,
                 labels=v,
                 inputs=Su,
                 num_sampled=num_samples,
                 num_classes=vocabulary_size))
        
        #------------- Second task
        train_step = optimizer.minimize(loss1)   

        #---- To retreive the embedding-node pairs after training
        n_in = tf.placeholder(tf.int32,shape=[1])
        Sn = tf.nn.embedding_lookup(S,n_in)
        
        n_out = tf.placeholder(tf.int32,shape=[1])
        Tn = tf.nn.embedding_lookup(T,n_out)

    
    #--- training
  	losses = []  
  	sess = tf.InteractiveSession(graph = graph1)
  	with tf.Session(graph = graph1) as sess:
    		sess.run(tf.initialize_all_variables())    
        
    		#summary_writer = tf.summary.FileWriter(result_dir, sess.graph)    
    		#13334927 node-context pairs
    		for epoch in range(n_epochs):
    				#--------- Train 
    				inputs = np.zeros(batch_size, dtype=np.int32)#np.ndarray(shape=(batch_size,1), dtype=np.int32)
    				labels = np.zeros(batch_size, dtype=np.int32)#np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    				f = open(fn+"/inf2vec_set.txt","r")     
    				idx2 = 0
    				idx = 0  
    				#---------- Only one epoch for now
    				for line in f:
    					#---- input node, output node 
    					sample = line.replace("\r","").replace("\n","").split(",")              
    					try:
    							original = dic[sample[0]]
    							label = dic[sample[1]]
    					except:
    							continue
    					inputs[idx] = original
    					labels[idx] = label
    					idx+=1  
                        
    					if(idx%batch_size==0):
    							idx2+=1
    							#---------- Run one training batch
    							inputs = inputs.reshape((batch_size,1))
    							labels = labels.reshape((batch_size,1))
                           
    							sess.run([train_step], feed_dict = {u: inputs, v: labels}) 
    							#summary_writer.add_summary(summary, idx2*idx)
                              
    							if idx2%1000 == 0:
    								jl = loss1.eval( feed_dict = {u: inputs, v: labels}) 
    								#l2 = loss2.eval( feed_dict = {u: inputs, v: labels, w : negative_samples, t:tim, c:casc})  
    								print('Joint Loss at step %s: %s' % (idx2*idx, jl))
                                      
    							inputs = np.zeros(batch_size, dtype=np.int)
    							labels = np.zeros(batch_size, dtype=np.int)
    							idx=0
    				f.close()
    
    		fsn = open(file_Sn,"w")
    		ftn = open(file_Tn,"w")
    		#---------- Get the source embedding of each node
    		Sn_list =  {}
    		Tn_list =  {}
    		for node in dic.keys():
    			emb_Sn = sess.run([Sn],feed_dict = {n_in:np.asarray([dic[node]])})
    			Sn_list[node] = emb_Sn
    			fsn.write(node+":"+",".join([str(s) for s in list(emb_Sn)])+"\n")
    		#Sn_list = np.concatenate( Sn_list, axis=0 )
    
    		for node in dic.keys():
    			emb_Tn = sess.run([Tn],feed_dict = {n_out:np.asarray([dic[node]])})
    			Tn_list[node] = emb_Tn  
    			ftn.write(node+":"+",".join([str(s) for s in list(emb_Tn)])+"\n")
    	  #Tn_list = np.concatenate( Tn_list, axis=0 )
    		fsn.close()
    		ftn.close()
  	log.write("inf2vec training time "+fn+" "+str(time.time()-start)+"\n")
   
    f = open(fn+"/"+fn+"_sizes.txt","r")
    target_size = int(next(f).strip())
    f.close()
	
	#----- create the network based on teh embeddings
    Sn_list = embedding_matrix(file_Sn,target_size,50)
    Tn_list = embedding_matrix(file_Tn,target_size,50)
    
    # weigh the edges based on the embedings
    g = ig.Graph.Read_Ncol(fn+"/"+fn+"_network.txt")
    db_w = open(fn+"/"+fn+"_inf2vec.inf","w")
    # compute the weight for each edge
    for edge in g.es:
        #--------------- Node 2 has influence on Node 1
        #idx1 = np.where(==nodes_idx1)
        #idx2 = np.where(g.vs[edge.tuple[0]]["name"]==nodes_idx1)
        S_emb = Sn_list[g.vs[edge.tuple[1]]["name"]]
        T_emb = Tn_list[g.vs[edge.tuple[0]]["name"]]
        weight = np.dot(S_emb,T_emb)
        db_w.write(str(g.vs[edge.tuple[1]]["name"])+" "+str(g.vs[edge.tuple[0]]["name"])+" "+str(weight)+"\n") 
        
    #--- normalize the weight based on the degree of each node
    d = pd.read_csv(fn+"/"+fn+"_inf2vec_weights.csv",header=None,sep =" ")
    d.columns = ["Node","node2","inf"]
    gr = d.groupby("Node").agg({"inf":"sum"}).reset_index()
    d = d.merge(gr,on="Node")
    d["inf"] = d["inf_x"]/d["inf_y"]
    del d["inf_x"]
    del d["inf_y"]
    d.to_csv(fn+"/"+fn+"_inf2vec.inf",index=False,header=False,sep=" ")
    log.write("inf2vec create weighted network "+fn+" "+str(time.time()-start)+"\n")
 
log.close()
    
