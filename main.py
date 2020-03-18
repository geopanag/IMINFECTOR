# -*- coding: utf-8 -*-
import time
import os
import sys
import argparse
import extract_feats_and_trainset
import preprocess_for_imm
import rank_nodes
import infector
import iminfector
import evaluation


parser=argparse.ArgumentParser()
parser.add_argument('--sampling_perc', help='')
parser.add_argument('--learning_rate', help='')
parser.add_argument('--n_epochs', help='')
parser.add_argument('--embedding_size', help='')
parser.add_argument('--num_neg_samples', help='')



if __name__ == '__main__':
	start = time.time()
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	os.chdir(os.path.join(dname,"..","Data"))
	args=parser.parse_args()
	
	#--- Parameters
	if(len(args.sampling_perc)>0):#sys.argv[1])>0):
		sampling_perc = int(args.sampling_perc)
	else:
		sampling_perc = 120	
		
	if(len(args.learning_rate)>0):
		learning_rate = float(args.learning_rate)
	else:
		learning_rate = 0.1

	if(len(args.n_epochs)>0):
		n_epochs = int(args.n_epochs)
	else:
		n_epochs = 5

	if(len(args.embedding_size)>0):
		embedding_size = int(args.embedding_size)
	else:
		embedding_size = 50

	if(len(args.num_neg_samples)>0):
		num_neg_samples = int(args.num_neg_samples)
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
