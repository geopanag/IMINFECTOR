"""
@author: george

Evaluate seed sets based on DNI and precision
"""
import pandas as pd
import os
import numpy as np
import glob


def DNI(seed_set_cascades):
    """
    Measure the number of distinct nodes in the test cascades started of the seed set
    """
    combined = set()    
    for i in seed_set_cascades.keys():
        for j in seed_set_cascades[i]:    
            combined = combined.union(j)
    return len(combined)

        
def run(fn,log):
    fa = open(fn+"/precision.txt","a")
    for seed_set_file in glob.glob(fn+"/seeds/*"):#:
        print(seed_set_file)
        #--- Compute precision
        print("------------------")
        fa.write(seed_set_file+"\n")
        f = open(seed_set_file,"r")
        l = f.read().replace("\n"," ")
        seed_set_all = [x for x in l.split(" ") if x!='']
        f.close()

        #------- Estimate the spread of that seed set in the test cascades
        spreading_of_set = {}
        if(fn=="mag"):
            step=1000, ma=11000
        elif(fn=="digg"):
            step=10,ma = 110
        else:
            step=100,ma = 1100
            
        for seed_set_size in range(step,ma,step):     
            seeds = seed_set_all[0:seed_set_size]

            #------- List of cascades for each seed
            seed_cascades =  {}
            for s in seeds:
                seed_cascades[str(s)] = []

            #------- Fill the seed_cascades
            seed_set = set()
            with open(fn+"/test_cascades.csv") as f:
                start_t = int(next(f))
                for line in f:
                    cascade= line.split(";")
                    op_ids = cascade[0].replace(",","").split(" ")
                    op_ids = op_ids[:-1]
                    #set(map(lambda x: x.split(" ")[0],cascade[2:]))
                    cascade = set(np.unique([i for i in cascade[1].replace(",","").split(" ") if "\n" not in i and ":" not in i]))
                    for op_id in op_ids:
                        if op_id in seed_cascades:
                            seed_cascades[op_id].append(cascade)
                            seed_set.add(op_id)

            #------- Fill the seed_cascades 
            seed_set_cascades = { seed: seed_cascades[seed] for seed in seed_set if len(seed_cascades[seed])>0 }
            print("Seeds found :",len(seed_set_cascades))
            fa.write(str(len(seed_set_cascades))+"\n")

            spreading_of_set[seed_set_size] = evaluate_distinct(seed_set_cascades)
        pd.DataFrame({"Feature":list(spreading_of_set.keys()), "Cascade Size":list(spreading_of_set.values())}).to_csv(seed_set_file.replace("seeds","spreading").replace("seeds/","spreading/"),index=False)
    fa.close()
