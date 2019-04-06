# -*- coding: utf-8 -*-
"""
@author: george

INFECT
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
     
os.chdir("Path/To/Data")

#----------------- Parameters
learning_rate = 0.1
n_epochs = 5
batch_size = 128
embedding_size1 =embedding_size2= 50
num_samples = 10


fn = "digg"

file_Sn = fn+"/embeddings/infect_source_embeddings_p.txt"
file_Tn = fn+"/embeddings/infect_target_embeddings_p.txt"
file_Cn = fn+"/embeddings/infect_c_embeddings_p.txt"
result_dir = 'summaries/'
    
log = open("time_log.txt","a")

#----------------- Min max normalization of cascade length and initator dictionary
f = open(fn+"/train_set.txt","r")
initiators = []
mi = np.inf
ma = 0
for l in f:
    parts  = l.split(",")
    initiators.append(parts[0])
    t = int(parts[2])
    if(t<mi):
        mi = t
    if(t>ma):
        ma = t
rang= ma-mi


initiators = np.unique((initiators))
dic_in = {initiators[i]:i for i in range(0,len(initiators))}
f.close()        
del initiators
vocabulary_size = len(dic_in)

#----------------- Target node dictionary
f = open(fn+"/"+fn.lower().replace("mag_","")+"_node_dic.json","r")
dic_out = json.load(f)
target_size = len(dic_out)   
      

#------- Same graph as original
start = time.time()

#----------------- Define the model with biases
graph1 = tf.Graph()
with graph1.as_default():
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)   
    
    #---- Batch size depends on the cascade
    u = tf.placeholder(tf.int32, shape=[None,1])
    v = tf.placeholder(tf.int32, shape=[None,1]) 
    
    #------------ Source (hidden layer embeddings)
    S = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size1], -1.0, 1.0), name="S")
    u2 =  tf.squeeze(u)
    Su = tf.nn.embedding_lookup(S,u2)
    #------------- First task
    #------------ Target (hidden and output weights)
    T = tf.Variable(tf.truncated_normal( [target_size, embedding_size1], stddev = 1.0 / math.sqrt(embedding_size1)), name="T")
    #T = tf.glorot_uniform_initializer( [target_size,embedding_size])
    
    #---- Noise contrastive loss function
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss1 = tf.reduce_mean(
        tf.nn.nce_loss(weights= T,
             biases=nce_biases,
             labels=v,
             inputs=Su,
             num_sampled=num_samples,
             num_classes=vocabulary_size))
    
    #------------- Second task
    #---- Cascade length
    c = tf.placeholder(tf.float32) 
    
    #------------ Cascade length weights (output layer of cascade length prediction)
    #C = tf.Variable(tf.truncated_normal([embedding_size2,1], stddev = 1.0 / math.sqrt(embedding_size2)), name="C")    
    #C = tf.Variable(tf.ones([embedding_size2]), name="C")    
    C = tf.constant(np.repeat(1,50).reshape((50,1)),tf.float32, name="C")
    #------------ Bias for cascade length
    b_c = tf.Variable(tf.zeros((1,1)),name="b_c")
    
    #------------ Loss2
    #o2 = tf.sigmoid(tf.matmul(tf.expand_dims(Su,0),C)+b_c)
    #tmp=tf.tensordot(tf.expand_dims(Su,1),tf.expand_dims(C,1), 1)
    #tmp=tf.matmul(tf.expand_dims(Su,1),tf.expand_dims(C,1), 1)
    
    tmp= tf.tensordot(Su,C,1)
    o2 = tf.sigmoid(tmp+b_c)
    #o2 = tf.sigmoid(tf.reduce_sum(tmp, axis=0)+b_c)
    #loss2 = tf.square(tf.subtract(c,o2))
    loss2 = tf.square(o2-c)
    #joint_loss = loss1+loss2
    
    train_step1 = optimizer.minimize(loss1)    
    train_step2 = optimizer.minimize(loss2)    
#    train_step = optimizer.minimize(joint_loss)    

    #---- To retreive the embedding-node pairs after training
    n_in = tf.placeholder(tf.int32,shape=[1])
    Sn = tf.nn.embedding_lookup(S,n_in)	

    n_out = tf.placeholder(tf.int32,shape=[1])
    Tn = tf.nn.embedding_lookup(T,n_out)
	
    #----------- Summaries
    #tf.summary.scalar('joint_loss', joint_loss)
   # tf.summary.scalar('loss1', loss1)
   # tf.summary.scalar('loss2', loss2)
   # summary_all = tf.summary.merge_all()    
 
    
    
"""
Main
"""

n_epochs = 5 
l1s = []
losses = []  

#sess = tf.InteractiveSession(graph=graph1)
#sess.run(tf.global_variables_initializer())       

with  tf.Session(graph = graph1) as sess:
    sess.run(tf.initialize_all_variables())    
    saver = tf.train.Saver() 
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)    
    for epoch in range(n_epochs):
        #--------- Train 
        f = open(fn+"/train_set.txt","r")
        idx2 = 0
        idx = 0  
        init= -1
        inputs = []#np.zeros(0, dtype=np.int32)#np.ndarray(shape=(batch_size,1), dtype=np.int32)
        labels = []#np.zeros(0, dtype=np.int32)#np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        
        #---------- Only one epoch for now
        for line in f:
        #---- input node, output node, copying_time, cascade_length, 10 negative samples 
            sample = line.replace("\r","").replace("\n","").split(",")              
            try:
                original = dic_in[sample[0]]
                label = dic_out[sample[1]]
            except:
                continue
            #---- check if we are at the same cascade
            if(init==original or init<0):
                init = original
                inputs.append(original) #=  np.vstack((inputs,original))
                labels.append(label) #labels =  np.vstack((labels,label))
                casc = (float(sample[2])-mi)/rang
            #---- New cascade, train on the previous one
            else:
                #---------- Run one training batch
                #--- Train cascade
                if len(inputs)<2:
                    inputs.append(inputs[0])
                    labels.append(labels[0])
                inputs = np.asarray(inputs).reshape((len(inputs),1))
                labels = np.asarray(labels).reshape((len(labels),1))
                sess.run([train_step1], feed_dict = {u: inputs, v: labels, c: [[0]]}) 
                #--- Train length
                #t  = Su.eval(feed_dict = {u: inputs[0].reshape(1,1), v: labels, c: [[casc]]})
                sess.run([train_step2], feed_dict = {u: inputs[0].reshape(1,1), v: labels, c: [[casc]]}) 
                idx2+=1
				
                if idx2%1000 == 0:
                    #jl = joint_loss.eval(feed_dict={u: inputs, v: labels, c:casc}) 
                    l1 = loss1.eval(feed_dict = {u: inputs, v: labels, c: [[0]]}) 
                    l2 = loss2.eval(feed_dict = {u: inputs[0].reshape(1,1), v: labels, c: [[casc]]}) 
                    print('Loss 2 at step %s: %s' % (idx2, l2))
                    print('Loss 1 at step %s: %s' % (idx2, l1))
                   
                inputs = []#np.zeros(0, dtype=np.int32)#np.ndarray(shape=(batch_size,1), dtype=np.int32)
                labels = []#np.zeros(0, dtype=np.int32)#np.ndarray(shape=(batch_size, 1), dtype=np.int32)
                inputs.append(original)
                labels.append(label)
                casc = (float(sample[2])-mi)/rang
                init = original
                #idx=0             
        f.close()
    
    fsn = open(file_Sn,"w")
    ftn = open(file_Tn,"w")
    fcn = open(file_Cn,"w")
    
    #---------- Get the source embedding of each node
    for node in dic_in.keys():
        emb_Sn = sess.run([Sn],feed_dict = {n_in:np.asarray([dic_in[node]])})
        fsn.write(node+":"+",".join([str(s) for s in list(emb_Sn)])+"\n")
    fsn.close()	
    for node in dic_out.keys():
        emb_Tn = sess.run([Tn],feed_dict = {n_out:np.asarray([dic_out[node]])})
        ftn.write(node+":"+",".join([str(s) for s in list(emb_Tn)])+"\n")
    ftn.close()
    
    emb_c =  sess.run([C])
    emb_c = emb_c[0].tolist()
    fcn.write(",".join([str(s) for s in emb_c]))
    fcn.close()

    
        
log.write("\n Time taken for the "+fn+" infect:"+str(time.time()-start)+"\n")
log.close()
with open(fn+'_losses.pickle', 'wb') as handle:
    pickle.dump(losses,handle)
