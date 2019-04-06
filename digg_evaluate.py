"""
@author: georg
Evaluate a set of seed sets
"""

import os
import pandas as pd
from igraph import *
import time
import numpy as np
import operator
import glob


def evaluate_distinct(seed_set_cascades,seed_set):
    """
    Measure the number of distinct nodes in the test cascades started of the seed set
    """
    combined = set()    
    for i in seed_set_cascades.keys():
        for j in seed_set_cascades[i]:    
            combined = combined.union(j)
    return len(combined)


"""
Main
"""
os.chdir("Path/To/Data")
fn="digg"
fa = open(fn+"/precision.txt","a")
for seed_set_file in glob.glob("seeds/*"):
    print(seed_set_file)
    f = open(seed_set_file,"r")
    l = f.read().replace("\n"," ")
    seed_set_all = [x for x in l.split(" ") if x!=''] # if x!='\n'
    f.close()
    
    #------- Estimate the spread of that seed set in the test cascades
    spreading_of_set = {}
    for seed_set_size in range(10,110,10):
        spreading_of_set[seed_set_size] = 0
        seeds = seed_set_all[0:seed_set_size]
        start = time.time()
         
        #------- List of cascades for each seed
        seed_cascades =  {}
        for s in seeds:
            seed_cascades[str(s)] = []
       
        #------- Fill the seed_cascades
        seed_set = set()
        with open("test_cascades.csv") as f:
            for line in f:
                cascade = line.split(";")
                op_id = cascade[0].split(" ")[0]
                cascade = set(map(lambda x: x.split(" ")[0],cascade[1:]))
                if op_id in seed_cascades:
                    seed_cascades[op_id].append(cascade)
                    seed_set.add(op_id)
           
        #------- Fill the seed_cascades 
        seed_set_cascades = { seed: seed_cascades[seed] for seed in seed_set if len(seed_cascades[seed])>0 }
        print("Seeds found :",len(seed_set_cascades))
        fa.write(str(len(seed_set_cascades))+"\n")
        
        #--- Do greedy
        spreading_of_set[seed_set_size] = evaluate_distinct(seed_set_cascades,list(seed_set))
                
    pd.DataFrame({"Feature":spreading_of_set.keys(), "Cascade Size":spreading_of_set.values()}).to_csv(seed_set_file.replace(".txt","_spreading.txt").replace("seeds/","spreading/"),index=False)
fa.close()  






