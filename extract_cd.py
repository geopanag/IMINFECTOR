# -*- coding: utf-8 -*-
"""
Extract the graphFile, userInflFile, trainingActionsFile and actionsFile for credit distribution model
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
            
def run(fn,log):    
    print("Reading the network")
    g = ig.Graph.Read_Ncol(fn+"/"+fn+"_network.txt")
    
    #----- Initialize features
    idx = 0
    deleted_nodes = []
    g.vs["Cascades_started"] = 0
    g.vs["Cumsize_cascades_started"] = 0
    g.vs["Cascades_participated"] = 0
    g.es["Inf"] = 0
    g.es["Dt"] = 0
    start = time.time()    
    f = open(fn+"/train_cascades.txt","r")  
    if(fn=="mag"):
        start_t = int(next(f))

    idx=0

    #------ actions and trainingactions file
    actionfile = open(fn+"/cd/actionFile.txt","w")
    train_actions = open(fn+"/cd/trainingActionsFile.txt","w")
    
    #---------------------- Iterate through cascades to create the train set
    not_found = 0
    for line in f:
        idx+=1
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
                actionfile.write(op_id+" "+str(idx)+" "+str(op_time)+"\n")   
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
                        #--- store action file
                        actionfile.write(j+" "+str(idx)+" "+str(tim)+"\n")        

            train_actions.write(str(idx)+"\n")
            
            #--- Draw edges between initiators and the rest
            for i in initiators:
                for j in range(0,len(cascade_nodes)):
                    try:
                        #------ i is co author with cascade_nodes[j]
                        edge = g.get_eid(cascade_nodes[j],i)    
                        #------ i influences j
                        #------ add to the graph edge attributes
                        g.es[edge]["Inf"]+=1
                        g.es[edge]["Dt"]+= (cascade_times[j]-op_time)
                    except:
                        continue
            
            #--- Draw edges between cascade nodes
            idx_of_node = 0;
            for p in range(0,len(papers)):
                for i in range(0,len(papers[p])):
                   #---- Draw edges to the authors from the subsequent papers
                   for p2 in range(p+1,len(papers)):
                       for j in range(0,len(papers[p2])):
                           try:
                               #------ i is followed by j
                               edge = g.get_eid(papers[p2][j],papers[p][i])    
                               #------ i influences j
                               #------ add to the graph edge attributes
                               g.es[edge]["Inf"]+=1
                               #np.where(cascade_nodes==papers[p][j])
                               idx_of_node2 = idx_of_node+(len(papers[p])-i)+j
                               g.es[edge]["Dt"]+= (cascade_times[idx_of_node2]-cascade_times[idx_of_node])
                           except:
                               continue
                   idx_of_node+=1                  
        else:
            initiators = []
            cascade = line.replace("\n","").split(";")
            cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade[1:]))
            if(fn=="weibo"):
                cascade_times = list(map(lambda x:  int(( (datetime.strptime(x.replace("\r","").split(" ")[1], '%Y-%m-%d-%H:%M:%S')-datetime.strptime("2011-10-28", "%Y-%m-%d")).total_seconds())),cascade[1:]))
            else:
                cascade_times = list(map(lambda x:  int(x.replace("\r","").split(" ")[1]),cascade[1:]))
            
            #---- Remove retweets by the same person in one cascade
            cascade_nodes, cascade_times = remove_duplicates(cascade_nodes,cascade_times)
            
            #---------- Dictionary nodes -> cascades
            op_id = cascade_nodes[0]
            #--- store action files
            for i in range(0,len(cascade_nodes)):
                actionfile.write(cascade_nodes[i]+" "+str(idx)+" "+str(cascade_times[i])+"\n")
            
            train_actions.write(str(idx)+"\n")
            
            #---------- Update metrics
            try:
                g.vs.find(name=op_id)["Cascades_started"]+=1
                g.vs.find(op_id)["Cumsize_cascades_started"]+=len(cascade_nodes)
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
            for i in range(0,len(cascade_nodes)): 
                # if it takes more than 15 mins, go away)
                #--- Add to the propagation graph all i's edges that point to j
                for j in range(i+1,len(cascade_nodes)):
                    try:
                        #------ i is followed by j
                        edge = g.get_eid(cascade_nodes[j],cascade_nodes[i]) 
                        
                        #------ add to the graph edge attributes
                        g.es[edge]["Inf"]+=1
                        g.es[edge]["Dt"]+= (cascade_times[j]-cascade_times[i])
                    except:
                        not_found+=1
                        continue
        if(idx%1000==0):
            print("-------------------",idx)
    actionfile.close()
    train_actions.close()
    print("Number of nodes not found in the graph: ",len(deleted_nodes))
    print("Number of edges not found in the graph: ",not_found)
    f.close()
    
    total_cascades = [sum(x) for x in zip(g.vs["Cascades_started"], g.vs["Cascades_participated"])]
    
    #------ user influence file
    userInf = pd.DataFrame({"Node":g.vs["name"],
                  "Activity":total_cascades,
                  "Cascades_participated":g.vs["Cascades_participated"]})
    
    
    userInf = userInf[userInf["Activity"]!=0]
    userInf.to_csv(fn+"/cd/userInflFile.txt",index=False,header=False,sep=" ")

    #------ graph file
    db_w = open(fn+"/cd/net.csv","w")
    # db weigthed network for simpath
    for edge in g.es:
        idx+=1
        if(edge["Inf"]!=0):
            #--------------- Node 2 has influence on Node 1
            weight = int(edge["Dt"]/edge["Inf"])
            db_w.write(str(g.vs[edge.tuple[1]]["name"])+" "+str(g.vs[edge.tuple[0]]["name"])+" "+str(weight)+"\n")
        else:
            db_w.write(str(g.vs[edge.tuple[1]]["name"])+" "+str(g.vs[edge.tuple[0]]["name"])+" 0"+"\n")
            if(idx%1000000==0):
                print("-------------------",idx)
    db_w.close()   
    
    dat = pd.read_csv(fn+"/cd/net.csv",header=None,sep=" ")
    dat.columns = ["n1","n2","w"]
    dat['id'] = dat.apply(lambda x: '-'.join([str(j) for j in sorted([x['n1'],x['n2']])]),axis=1)
    tmp = dat.drop(["n1","n2"],axis=1)
    #215086-336224
    idx = dat.duplicated(["id"],keep=False)
    recip = dat[~idx]
    #---- merge the reciprocal edges with the different weights
    dat = dat[idx].drop_duplicates(['id'])
    tmp = tmp[idx].drop_duplicates(['id'],keep="last")
    d = dat.merge(tmp,on="id")
    recip = recip.rename(columns = {"w":"w_x"})
    recip["w_y"] = 0
    dat = pd.concat([recip,d])
    del dat["id"]
    dat["ts"] = 0
    dat.to_csv(fn+"/cd/graphFile.txt",index=False,header=False,sep=" ")
    log.write("extract cd "+fn+" :"+str(time.time()-start)+"\n")
  
    
    