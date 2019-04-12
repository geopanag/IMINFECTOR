"""
r: georg

Take the top 0.5%  nodes ranked based on each feature and store it to be used for netrate. 
Take the top 100 based on each feature and store it as seed set.
"""

import os            
import pandas as pd

fn ="digg"
os.chdir("Path/To/"+fn+"Data")

dat = pd.read_csv("node_features.csv")

#------ Take the top 0.5%~=3000 seeds from each measure and store it in each column of top
if(fn =="digg"):
	perc = 100
elif(fn=="weibo"):
	perc = 1000
else:
	perc = 10000
	
top = pd.DataFrame(columns=dat.columns)
for col in dat.columns:
    if(col=="Node"):
        continue
    top[col] = dat.nlargest(perc,col)["Node"].values

top = top.drop(["Node"], axis=1)  

#------ Store as seed sets the top 100 of each feature
for c in top.columns:
    f = open("seeds/"+c.lower()+"_seeds.txt","w")
    f.write(" ".join([str(x) for x in list(top.loc[0:100,c].values)]))
    f.close()
    
