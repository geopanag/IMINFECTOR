# -*- coding: utf-8 -*-
"""
@author: georg
"""

import time
import os
import sys
"""
import extract_feats_and_trainset
import preprocess_for_imm
import rank_nodes
import infector
import iminfector
import evaluation
"""

if __name__ == '__main__':
	start = time.time()
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	os.chdir(os.path.join(dname,"..","Data"))
	
	print(sys.argv[1])
	#--- Parameters
	if(len(sys.argv[1])>0):
		sampling_perc = int(sys.argv[1])
	else:
		sampling_perc = 120	
		
	if(len(sys.argv[2])>0):
		learning_rate = float(sys.argv[2])
	else:
		learning_rate = 0.1

	if(len(sys.argv[3])>0):
		n_epochs = int(sys.argv[3])
	else:
		n_epochs = 5

	if(len(sys.argv[4])>0):
		embedding_size = int(sys.argv[4])
	else:
		embedding_size = 50

	if(len(sys.argv[5])>0):
		num_neg_samples = int(sys.argv[5])
	else:
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