# -*- coding: utf-8 -*-
"""
Extract databased weighted network for simpath and imm
"""

import time
import igraph as ig
import pandas as pd
import numpy as np
from datetime import datetime
import json


def sort_papers(papers):
    """
    # Sort MAG diffusion cascade, which is a list of papers and their authors, in the order the paper'sdate
    """
    x =list(map(int,list(map(lambda x:x.split()[-1],papers))))
    return [papers[i].strip() for i in np.argsort(x)]
  
  
def run(fn,log):    
	print("Reading the network")
	start = time.time()  

	g = ig.Graph.Read_Ncol(fn+"/"+fn+"_network.txt")    
	#------ in mag it is undirected
	if fn =="mag":
	    g.to_undirected()
	f = open(fn+"/train_cascades.txt","r")    
	#----- initialize features
	deleted_nodes = []
	g.vs["Cascades_started"] = 0
	g.vs["Cascades_participated"] = 0
	g.es["Inf"] = 0
	if(fn=="mag"):
		start_t = int(next(f))
	idx=0  
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
                        
			#--- Draw edges between initiators and the rest
			for i in initiators:
				for j in cascade_nodes:
					try:
						#------ i is co author with j
						#------ i influences j
						edge = g.get_eid(i,j)    
						#prop_net.add_edge(i,j)
						#------ add to the graph edge attributes
						g.es[edge]["Inf"]+=1
						#g.es[edge]["Dt"]+= (cascade_times[j]-cascade_times[i]).total_seconds()
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
								#prop_net.add_edge(i,j)
								#------ add to the graph edge attributes
								g.es[edge]["Inf"]+=1
							except:
								continue                        
		else:
			initiators = []
			cascade = line.replace("\n","").split(";")
			cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade[1:]))
			
			#---------- Dictionary nodes -> cascades
			op_id = cascade_nodes[0]
			#op_time = cascade_times[0]

			#---------- Update metrics
			try:
				g.vs.find(name=op_id)["Cascades_started"]+=1
				#g.vs.find(op_id)["Cumsize_cascades_started"]+=len(cascade_nodes)
			except: 
				deleted_nodes.append(op_id)
				continue
            
			if(len(cascade_nodes)<2):
				continue
            
			for i in cascade_nodes[1:]:
				try:
					g.vs.find(name=i)["Cascades_participated"]+=1
				except:
					continue
                
			#---- Data-based weighing
			for i in range(len(cascade_nodes)): 
				# if it takes more than 15 mins, go away)
				#--- Add to the propagation graph all i's edges that point to j
				for j in range(i+1,len(cascade_nodes)):
					try:
						#------ i is followed by j
						edge = g.get_eid(cascade_nodes[j],cascade_nodes[i])    
						#------ i influences j
						#prop_net.add_edge(cascade_nodes[i], cascade_nodes[j])
						#------ add to the graph edge attributes
						g.es[edge]["Inf"]+=1
					except:
						continue
                   
		idx+=1
		if(idx%1000==0):
			print("-------------------",idx)
    
	idx=0    
	print("Number of nodes not found in the graph: ",len(deleted_nodes))
	f.close()

	#db weigthed network for simpath
	print("writing db")
	db_w = open(fn+"/"+fn+"_db_weights.	csv","w")
	#db_w2 = open(fn+"/"+fn+"_db_weights_tmp2.csv","w")
	for edge in g.es:
		idx+=1
		if(edge["Inf"]!=0):
			#--------- devide the weight by the activity of the influencer
			activity = g.vs.find(name=g.vs[edge.tuple[1]]["name"])
			activity = activity["Cascades_participated"]+activity["Cascades_started"]
			#--------- Node 2 has influence on Node 1
			#if(w>0):
			db_w.write(str(g.vs[edge.tuple[1]]["name"])+","+str(g.vs[edge.tuple[0]]["name"])+","+str(edge["Inf"])+","+str(activity)+"\n")
      
			#--------- Weight for DB
			if(idx%100000==0):
				print("-------------------",idx)	
	db_w.close()
	#db_w2.close()  

	d = pd.read_csv(fn+"/"+fn+"_db_weights.csv",header=None)
	d.columns = ["node1","node2","inf","act"]
	#d_alt = d[["node1","node2","inf"]].copy()
	d["inf"]= d["inf"]* 1.0 /d["act"]
	del d["act"]
	
	#---- normalize for the sum of output edges
	gr = d.groupby("node1").agg({"inf":"sum"}).reset_index()
	d = d.merge(gr,on="node1")
	d["inf"] = d["inf_x"]*1.0/d["inf_y"]
	del d["inf_x"]
	del d["inf_y"]
	d.loc[d["inf"].isna(),"inf"] = 0
 
	d.to_csv(fn+"/"+fn+"_db.inf",index=False,header=False,sep=" ")
	nodes = list(set(d["node1"].unique()).union(set(d["node2"].unique()))) 
    
	dic = {int(nodes[i]):i for i in range(0,len(nodes))}
	d['node1'] = d['node1'].map(dic)
	d['node2'] = d['node2'].map(dic)
    
	#--- Store the ids to translate the seeds of IMM
	f= open(fn+"/"+fn+"_db_incr_dic.json","w")
	json.dump(dic,f)
	f.close()
	
	d.to_csv(fn+"/"+fn+"_db.inf",index=False,header=False,sep=" ")
    
	attribute = open(fn+"/"+fn+"_db_attribute.txt","w")
	attribute.write("n="+str(len(nodes)+1)+"\n")
	attribute.write("m="+str(d.shape[0])+"\n")
	attribute.close()
    
	log.write("extract db "+fn+" :"+str(time.time()-start)+"\n")
        