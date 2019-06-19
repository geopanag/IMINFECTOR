# -*- coding: utf-8 -*-
"""
@author: georg
"""

import time
import os
import extract_feats_and_trainset
import preprocess_for_imm
import rank_nodes
import infector
import iminfector
import evaluation


if __name__ == '__main__':
    start = time.time()
    os.chdir("Path/To/Data")
    
    #--- Parameters
    sampling_perc = 120
    learning_rate = 0.1
    n_epochs = 5
    embedding_size = 50
    num_neg_samples = 10
    
    log= open("time_log.txt","a")
    
    for fn in ["weibo","digg","mag"]:
        extract_feats_and_trainset.run(fn,sampling_perc,log)
        preprocess_for_imm.run(fn,log)
        rank_nodes.run(fn) 
        infector.run(fn,learning_rate,n_epochs,embedding_size,num_neg_samples,log)
		iminfector.run(fn,embedding_size,log)
        evaluation.run(fn,log)
    log.close()
