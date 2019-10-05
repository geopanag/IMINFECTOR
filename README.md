# Influence Maximization via Representation Learning

Code and instructions to reproduce the analysis in the [paper](https://arxiv.org/abs/1904.08804)

``` bash
mkdir Code
mkdir Data
mkdir Figures
cd Code
git clone https://github.com/GiorgosPanagopoulos/Influence-Maximization-via-Representation-Learning
cd Code
pip install -r requirements.txt
```



## Data
Data/-> subfolders Digg, Weibo, MAG each with subfolders-> Init_Data,Embeddings, Seeds,Spreading<br />

Download [Digg](https://www.isi.edu/~lerman/downloads/digg2009.html) 
into Digg->Init_Data, run digg_preprocessing.py <br />

For [Weibo](https://aminer.org/influencelocality) 
into Weibo->Init_Data and run weibo_preprocessing.py <br />

We used the official [MAG](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/), but there is an [open version](https://aminer.org/open-academic-graph). 
Add it to MAG->Init_data and run mag_preprocessing.py<br />

