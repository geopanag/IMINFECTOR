# -*- coding: utf-8 -*-
"""
To derive the MAG network and diffusion cascades we employed the tables Paper, Paper References, Author, PaperAuthorAffiliation, Fields of Study, Paper Fields of Study from MAG 
https://docs.microsoft.com/en-us/academic-services/graph/reference-data-schema

Extract network and diffusion cascades from CS of MAG
"""


import pandas as pd
import os
import numpy as np
import networkx as nx

def clean_authors_by_name():
    #---- Clean authors
    auth = pd.read_csv("authors.txt")
    auth = auth.ix[auth.columns[0,2]]
    # the unique display names are 83209107 and the normalized are 60m
    idx = auth.iloc[:,1].apply(lambda x:x[0]<"a" or x[0]>"z")
    auth=auth[~idx]
    auth.to_csv("authors.txt",index=False)
    
def prepare_fields():
    #---- Keep the CS papers that have high confidence
    f = pd.read_csv("fields.txt",sep="\t",header=None)
    cs_fields = f.loc[f[4].str.contains('computer')==True,0].values

    f1 = open("paper_fields.txt","r")
    f2 = open("paper_fields_filtered.txt","w")
    i = 0
    for l in f1:
        i+=1
        if(i%1000000==0):
            print(i)
        parts = l.split("\t")
        #-- check if the confidence is enough
        try:
            ty = int(parts[1])
            conf= float(parts[2])
        except:
            next
        if(conf>0.5):
            if(ty in cs_fields):
                f2.write("cs"+","+parts[0]+"\n")
    f1.close()

def extract_network():
    pap_auth = pd.read_csv("paper_author.txt")
    pap_auth = pap_auth.drop(pap_auth.columns[0],axis=1)
    pap_auth.to_csv("paper_author.txt",index=False)
    
    fields = pd.read_csv("paper_fields_filtered.txt")
    fields.columns = ["PapID","field"]
     #--- Optional: Keep only the authors whose names do not have many possible IDs (name ambiguity)
    to_remove = pd.read_csv("ambig_authors_papers.txt")
    
    to_remove.columns = ["AuthID","PapID","label"]
    to_remove = to_remove.loc[to_remove["label"]==0,"AuthID"].unique()
    fields = pd.read_csv("paper_fields_filtered.txt")
    fields.columns = ["PapID","field"]
    for f in fields.field.unique():
        print(f)
        net_name="../network_"+f+".csv"
        tmp_f = fields.loc[fields.field==f,"PapID"]
        #---- First filtering (keep papers in field f)
        tmp_f = pap_auth.merge(tmp_f.to_frame("PapID"),on="PapID")
        #---- Second filtering (remove ambiguous authors with 1 paper)
        #---- Create the edge list
        tmp_f = tmp_f.merge(tmp_f,on="PapID")
        tmp_f.loc[tmp_f.AuthID_x<tmp_f.AuthID_y,["AuthID_x","AuthID_y"]].to_csv(net_name,index=False)
        
def extract_cascades():
    #----- Extract cascades
    pap = pd.read_csv("papers_cs.txt",sep=";", encoding = "ISO-8859-1") 
    pap = pap.drop(pap.columns[[1,2,4]],axis=1)
    pap["PapID"] = pap["PapID"].astype(int)
    pap['Date'] = pd.to_datetime(pap["Date"]).values.astype(np.int64) // 10 ** 9
    
    ref = pd.read_csv("references.txt",header=None)
    ref.columns = ["PapID","RefID"]
    
    auth = pd.read_csv("author_papers.txt")
    auth["AuthID"] = auth["AuthID"].map(str)
    auth = auth.groupby("PapID").agg(lambda x:"%s" % ', '.join(x)).reset_index()
    
    pap_fields = pd.read_csv("paper_fields.txt")
    pap_fields.columns=["PapID","fields"]
    
    pap_fields = pap_fields.loc[pap_fields["PapID"]!="paper",:]
    pap_fields["PapID"] = pap_fields["PapID"].astype(int)
    
    print("done preprocessing, starting the extraction...")
    
    f="cs"
    f_name="cascades_"+f+".csv"
    print("at "+f_name)
    tmp = pap_fields.loc[pap_fields["fields"]==f,:]
    del tmp["fields"]
    tmp = tmp.merge(ref,on="PapID").merge(pap,on="PapID")
    tmp= tmp.sort_values(by=["RefID", "Date"]).merge(auth,on="PapID")
    
    print("constructing the final dataset")   
    tmp["Auth"] = tmp["AuthID"].map(str)+" "+tmp["Date"].map(str)
    tmp.drop(['AuthID', 'Date','PapID'], axis=1, inplace=True)
    
    tmp = tmp.merge(auth,left_on="RefID",right_on = "PapID")
    del tmp["PapID"]
    tmp = tmp.groupby("RefID").agg({"AuthID":"first","Auth":lambda x:"%s" % ': '.join(x)}).reset_index()
    print("adding date of paper")
    
    tmp = tmp.merge(pap,left_on="RefID",right_on="PapID")
    tmp["AuthID"] = tmp["AuthID"].map(str)+" "+tmp["Date"].map(str)
    tmp.drop(['PapID', 'RefID','Date'], axis=1, inplace=True)
    tmp = tmp[["AuthID","Auth"]]
    tmp.to_csv(f_name,sep=";",index=False,header=False)

    f_n = open("reduced_cascades_"+fn+".csv","w") 
    f = open("cascades_"+fn+".csv","r") 
    for l in f:
        parts = l.replace("\n","").split(";")
        if(len(parts[0].split(":"))>=30): #10 for cs, 20 for mag
            f_n.write(parts[1]+";"+parts[0]+"\n")
    f.close()
    f_n.close()
    
    f = open("reduced_cascades_"+fn+".csv","r")  
    #--- Break train and test
    times = []
    for l in f:
        times.append(l.split(";")[0].split(" ")[-1])
    times = list(map(int,times))
    times.sort()  
    start = abs(times[0])

    times=[i+start for i in times]

    print(start) 
    break_point = round(len(times)*80/100)
    train_times =  times[0:break_point]
    test_times =  times[break_point:len(times)]

    break_point = test_times[0]
    print(break_point) #

    f = open("../reduced_cascades_.csv","r")  
    f_test = open("../test_cascades.csv","w")  
    f_train = open("../train_cascades.csv","w")  
    f_test.write(str(start)+"\n")
    f_train.write(str(start)+"\n")
    #--- Break train and test
    times = []
    for l in f:
        time = start + int(l.split(";")[0].split(" ")[-1])
        if(time<break_point):
            f_train.write(l)
        else:
            f_test.write(l)

    f_test.close()
    f_train.close()

    
if __name__ == '__main__':
    os.chdir("Path/To/Init_data")
    
    clean_authors_by_name()
    prepare_fields()
    extract_network()
    extract_cascades()

#------ Reduce cascades and split them


