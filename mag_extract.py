"""
@author: georg

Extract summary features about the participation of nodes in the cascades
Extract node centralities from the follower network and the cascades
Extract the train node cascades without duplicates
"""


import pandas as pd
import os
import numpy as np
import igraph as ig
import time 
import json
os.chdir("Path/To/Data")

start = time.time()
sampling_perc = 120
log= open("../time_log.txt","a")
for fn in  ["cs"]:
    print(fn)
    to_train_on = open("mag_"+fn+"/train_set.txt","w")
    f = open("mag_"+fn+"/train_cascades.csv","r")  
	g= ig.Graph.Read_Ncol("mag_"+fn+"/"+fn+"_network.txt")
    g.to_undirected()
    start = time.time()
    kcores = g.shell_index()
    log.write("K-core time:"+str(time.time()-start)+"\n")

    g.vs["Cascades_started"] = 0
    g.vs["Cumsize_cascades_started"] = 0
    g.vs["Cascades_participated"] = 0
    log.write(" net:"+fn+"\n")
    start_t = int(f.next())
    
    node_cascades ={} 
    #--- Break train and test
    times = []
    idx=0
    for l in f:
        idx+=1
        if(idx%10000==0):
            print("------------------"+str(idx))

        parts = l.split(";")
        initiators = parts[0].replace(",","").split(" ")
        op_time = int(initiators[-1])+start_t
        initiators = initiators[:-1]
        papers = parts[1].replace("\n","").split(":")
        #papers = sort_papers(papers)
        papers = [list(map(lambda x: x.replace(",",""),i)) for i in list(map(lambda x:x.split(" "),papers))]
        
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
                
        no_samples = round(len(cascade_nodes)*sampling_perc/100)
        times = [op_time/(abs((cascade_times[i]-op_time))+1) for i in range(0,len(cascade_nodes))]
        s_times = sum(times)
        
        if s_times==0:
	    samples = []	
	    else:
           probs = [float(i)/s_times for i in times]
            samples = np.random.choice(a=cascade_nodes, size=int(no_samples), p=probs) 
        
        casc_len = str(len(cascade_nodes))
        for op_id in initiators:    
            if op_id in node_cascades:
                node_cascades[op_id].append(cascade_nodes)
            else:
                node_cascades[op_id] =  []
                node_cascades[op_id].append(cascade_nodes)
            for i in samples:
                #---- Write inital node, copying node, copying time, length of cascade
                to_train_on.write(str(op_id)+","+i+","+casc_len+"\n")

    log.write("Training time fast "+fn+":"+str(time.time()-start)+"\n") # cs 3644.97332572937  # 2045556428
    
	pd.DataFrame({"Node":g.vs["name"],	
              "Kcores":kcores,
			  "avg_cascades_size": g.vs["Cumsize_cascades_started"]/g.vs["Cascades_started"]}).to_csv("node_features.csv",index=False) 
    f = open("mag_"+fn+"/node_cascades_"+fn+".json","w")
    json.dump(node_cascades,f)
    f.close()
    to_train_on.close()
    
log.close()
