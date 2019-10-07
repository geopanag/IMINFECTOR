import os
from weibo_preprocessing import weibo_preprocessing
from digg_preprocessing import digg_preprocessing

def split_train_and_test(cascades_file):
    """
	Data from https://aminer.org/influencelocality 
    # Keeps the ids of the users that are actively retweeting
    # Train time:(2011.10.29 -2012.9.28) and test time (2012.9.28 -2012.10.29)
    """
    
    f = open(cascades_file)
    ids = set()
    train_cascades = []
    test_cascades = []
    counter = 0
    
    for line in f: 
        
        date = line.split(" ")[1].split("-")
        original_user_id = line.split(" ")[2]
        
        retweets = f.next().replace(" \n","").split(" ")
        #----- keep only the cascades and the nodes that are active in train (2011.10.29 -2012.9.28) and test (2012.9.28 -2012.10.29)
           
        retweet_ids = ""
        
        #------- last month at test
        if int(date[0])==2012 and ((int(date[1])>=9 and int(date[2])>=28)  or (int(date[1])==10  and int(date[2])<=29)): 
            ids.add(original_user_id)           
           
            cascade = ""
            for i in range(0,len(retweets)-1,2):
                ids.add(retweets[i])
                retweet_ids = retweet_ids+" "+retweets[i]
                cascade = cascade+";"+retweets[i]+" "+retweets[i+1]
               
           #------- For each cascade keep also the original user and the relative day of recording (1-32)
            date = str(int(date[2])+3)
            op = line.split(" ")
            op = op[2]+" "+op[1]
            test_cascades.append(date+";" +op+cascade)
    
       #------ The rest are used for training
        elif (int(date[0])==2012 and int(date[1])<9 and int(date[2])<28) or (int(date[0])==2011 and int(date[1])>=10 and int(date[2])>=29):
             
            ids.add(original_user_id)          
            cascade = ""          
            for i in range(0,len(retweets)-1,2):
                ids.add(retweets[i])
                retweet_ids = retweet_ids+" "+retweets[i]
                cascade = cascade+";"+retweets[i]+" "+retweets[i+1]
            if(int(date[1])==9):
                date = str(int(date[2])-27)
            else:
                date = str(int(date[2])+3)
            op = line.split(" ")
            op = op[2]+" "+op[1]
            train_cascades.append(date+";" +op+cascade)
           
        counter+=1    
        if (counter % 10000==0):
            print("------------"+str(counter))
			

"""
Main
"""
if __name__ == '__main__':
	## Create folder structure
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	os.chdir(os.path.join(dname,"..","Data"))
	for dataset in ["Digg","Weibo","MAG"]:
		for folder in ["Init_Data","Embeddings", "Seeds","Spreading"]:
			os.makedirs(os.path.join(dataset,folder))
		
	ans = weibo_preprocessing(os.path.join("Weibo","Init_Data"))
	print(ans)
	ans = digg_preprocessing(os.path.join("..","Digg","Init_Data"))
	print(ans)
	
