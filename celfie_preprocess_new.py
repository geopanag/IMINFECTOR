import os 
import json
from datetime import datetime
import pandas as pd
import igraph as ig
import numpy as np
import time
import tensorflow as tf
import math

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

    
def store_samples(fn,cascade_nodes,cascade_times,initiators,op_time,train_set,sampling_perc=120):
    """
    # Store the samples  for the train set as described in the node-context pair creation process for INFECTOR
    """
    #---- Inverse sampling based on copying time
    op_id = cascade_nodes[0]
    casc_len = len(cascade_nodes)-len(initiators)
    #print(casc_len)
    no_samples = round(casc_len*sampling_perc/100)
    #print(no_samples)
    times = [op_time/(abs((cascade_times[i]-op_time))+1) for i in range(len(initiators),len(cascade_nodes))]
    s_times = sum(times)
    
    if s_times==0:
        samples = []	
    else:
        probs = [float(i)/s_times for i in times]
        samples = np.random.choice(a=cascade_nodes[len(initiators):], size=int(no_samples), p=probs) 
    
    casc_len = str(casc_len)
    #to_train_on = open(fn+"/train_set.txt","w")
    #----- Store train set
    if(fn=="mag"):
        for op_id in initiators:    
            for i in samples:
                #---- Write inital node, copying node,length of cascade
                train_set.write(str(op_id)+","+i+","+casc_len+"\n")                	
    else:                
        for i in samples:
            if(op_id!=i):
                #---- Write initial node, copying node, copying time, length of cascade
                train_set.write(str(op_id)+","+i+","+casc_len+"\n")


"""
Main
"""
os.chdir("path/to/data")
sampling_perc = 120
embedding_size = 50
negative_size = 10
learning_rate = 0.01
n_epochs = 5
batch_size = 128
num_samples = 10

start_t = 5364662400
log = open("time_log.txt","a")

for fn in ["digg","weibo","mag"]:
	f_tr = open(fn+"/train_cascades.txt","r")
	f_te = open(fn+"/test_cascades.txt","r")
	total = []
	for l in f_tr:
		total.append(l)
	for l in f_te:
		total.append(l)
	f_tr.close()
	f_te.close()
	
	li = [80]
	for perc in li:     
		start = time.time()
		#----- Split different percentages
		thres = len(total)*perc/100
		#f_tr = open(fn+"/train_cascades"+str(perc)+".txt","w")
		#for i in total[0:thres]:
		#    f_tr.write(i)
		#f_tr.close()
		print(fn)
		print(perc)
		#f_te = open(fn+"/test_cascades"+str(perc)+".txt","w")
		#for i in total[thres:]:
		#	f_te.write(i)
		#f_te.close()
		train_set = open(fn+"/celfie_train_set_"+str(perc)+".txt","w")
	
		print(thres)
		for line in total[0:thres]:
			if(fn=="mag"):
				parts = line.split(";")
				initiators = parts[0].replace(",","").split(" ")
				op_time = int(initiators[-1])+start_t
				initiators = initiators[:-1]
				if(len(parts)<2):
					continue
				papers = parts[1].replace("\n","").split(":")
				papers = sort_papers(papers)
				papers = [list(map(lambda x: x.replace(",",""),i)) for i in list(map(lambda x:x.split(" "),papers))]
          
				#--- Update metrics of initiators
				cascade_times = []
				for op_id in initiators:
					cascade_times.append(op_time)
		    
				cascade_nodes = initiators[:]
				for p in papers:
					tim = int(p[-1])+start_t            
					for j in p[:-1]:
						if j!="" and j!=" " and j not in cascade_nodes:
							cascade_nodes.append(j)
							cascade_times.append(tim)    
			else:
				cascade = line.replace("\n","").split(";")#.replace("[","").replace("]","")
				#print(cascade)
				if(fn=="weibo"):
					cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade[1:]))
					cascade_times = list(map(lambda x:  int(( (datetime.strptime(x.replace("\r","").split(" ")[1], '%Y-%m-%d-%H:%M:%S')-datetime.strptime("2011-10-28", "%Y-%m-%d")).total_seconds())),cascade[1:]))
				else:
					cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade))
					cascade_times = list(map(lambda x:  int(x.replace("\r","").split(" ")[1]),cascade))
				#---------- Dictionary nodes -> cascades
				op_id = cascade_nodes[0]
				op_time = cascade_times[0]

				if(len(cascade_nodes)<2):
					continue
				initiators = [op_id]
			store_samples(fn,cascade_nodes,cascade_times,initiators,op_time,train_set)
		train_set.close()
		log.write("Total time for celfie preprocess: "+fn+"="+str(time.time()-start)+"\n")

		f = open(fn+"/celfie_train_set_"+str(perc)+".txt","r")
		initiators = []
		for l in f:
			parts  = l.split(",")
			initiators.append(parts[0])
			t = int(parts[2])
        
		#----------------- Source node dictionary
		initiators = np.unique((initiators))
        
		dic_in = {initiators[i]:i for i in range(0,len(initiators))}
		f.close()     

		input_size = len(dic_in)
		print(input_size)
		#----------------- Target node dictionary
		
		f = open(fn+"/"+fn+"_node_dic.json","r")
		dic_out = json.load(f)
		target_size = len(dic_out)
		
		f = open(fn+"/"+fn+"_sizes_"+str(perc)+".txt","w")
		f.write(str(target_size)+"\n")
		f.write(str(input_size))
		f.close()
		
		file_Sn = fn+"/embeddings/source_"+str(perc)+"_embeddings.txt"
		file_Tn = fn+"/embeddings/target_"+str(perc)+"_embeddings.txt"
	
		start = time.time()
		#----------------- Define the model with biases
		graph1 = tf.Graph()
		with graph1.as_default():
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
       
			#------------ User u, all nodes v in u's context 
			#u = tf.placeholder(tf.int32, shape=[batch_size,1])
			#v = tf.placeholder(tf.int32, shape=[batch_size,1]) 
			u = tf.placeholder(tf.int32, shape=[batch_size,1],name="u")
			v = tf.placeholder(tf.int32, shape=[batch_size,1],name="v")
         
			#------------ Source and Target (hidden and output weights)
			S = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0), name="S")
			T = tf.Variable(tf.truncated_normal( [target_size,embedding_size], stddev = 1.0 / math.sqrt(embedding_size)), name="T")
     
			u2 =  tf.squeeze(u)
			#Su = tf.matmul(u,S)
			Su = tf.nn.embedding_lookup(S,u2)
        
			#---- To retreive the embedding-node pairs
			n = tf.placeholder(tf.int32,shape=[1])
			Tn = tf.nn.embedding_lookup(T,n)
			Sn = tf.nn.embedding_lookup(S,n)
    
			#---- Noise contrastive loss function
			nce_biases = tf.Variable(tf.zeros([input_size]))
    
			loss = tf.reduce_mean(
				tf.nn.nce_loss(weights= T,
					 biases=nce_biases,
					 labels=v,
					 inputs=Su,
					 num_sampled=num_samples,
					 num_classes=target_size))
        
			train_step = optimizer.minimize(loss)

		#-------------------------- Run the model
		to_train_on = fn+"/celfie_train_set_"+str(perc)+".txt"
		losses = []  
		#sess = tf.InteractiveSession(graph = graph1)
		with tf.Session(graph = graph1) as sess:
			sess.run(tf.initialize_all_variables())
			#test_writer = tf.summary.FileWriter(test_result_dir, sess.graph)
        
			#--------- Train 
			inputs = np.zeros((batch_size,1), dtype=np.int)#np.ndarray(shape=(batch_size,1), dtype=np.int32)
			labels = np.zeros((batch_size,1), dtype=np.int)#np.ndarray(shape=(batch_size, 1), dtype=np.int32)
			#negative_samples = np.zeros((batch_size,10), dtype=np.int)#np.ndarray(shape=(batch_size, 10), dtype=np.int32)
         
			#13334927 node-context pairs
			for epoch in range(n_epochs):
				f = open(to_train_on,"r")
				next(f)
				idx2 = 0
				idx = 0   
				#---------- Only one epoch for now
				for line in f:
					sample = line.replace("\r","").replace("\n","").split(",")            
					#original= int(sample[0])
					#label = int(sample[1])
					try:
						original = dic_in[sample[0]]
						label = dic_out[sample[1]]
					except:
						continue
					#if(original not in nodes):
					#   continue
					inputs[idx] = original
					labels[idx] = label
					idx+=1
                
					if(idx%batch_size==0):
						idx2+=1
						#---------- Run one training batch
						_ = sess.run([train_step], feed_dict = {u: inputs, v: labels}) 
						
						if idx2%1000000 == 0:
							# output the training accuracy every 100 iterations
							l = loss.eval( feed_dict = {u: inputs, v: labels})  
							losses.append(l)
							print('Loss at step %s: %s' % (idx2*idx, l))
		                
						inputs = np.zeros((batch_size,1), dtype=np.int)
						labels = np.zeros((batch_size,1), dtype=np.int)
						#tim = np.zeros(batch_size, dtype=np.float32)
						idx=0
                
				f.close()
            
			#emb_S,emb_T = sess.run([S,T])
			fsn = open(file_Sn,"w")
			ftn = open(file_Tn,"w")
            
			#---------- Get the embeddings of each node and store it
			for node in dic_in.keys():
				emb_Sn = sess.run([Sn],feed_dict = {n:np.asarray([dic_in[node]])})
				fsn.write(str(node)+":"+",".join([str(s) for s in list(emb_Sn[0])])+"\n")
			for node in dic_out.keys():
				emb_Tn = sess.run([Tn],feed_dict = {n:np.asarray([dic_out[node]])})
				ftn.write(str(node)+":"+",".join([str(s) for s in list(emb_Tn[0])])+"\n")
			fsn.close()
			ftn.close()
        
		log.write("Total time for celfie training: "+fn+" "+str(perc)+"="+str(time.time()-start)+"\n")


log.close()  

