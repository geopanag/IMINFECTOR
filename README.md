

Code and instructions to reproduce the analysis in [Influence Maximization via Representation Learning](https://arxiv.org/abs/1904.08804)

## Folder Structure
Code/ The current folder <br />
Data/-> subfolders Digg, Weibo, MAG each with subfolders-> Init_Data,Embeddings, Seeds,Spreading<br />
Figures/ <br />

## Data
Download [Digg](https://www.isi.edu/~lerman/downloads/digg2009.html) into Digg->Init_Data,  run digg_preprocessing.py <br />
For [Weibo](https://aminer.org/influencelocality) into Weibo->Init_Data and run weibo_preprocessing.py <br />
We used the official [MAG](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/), but there is an [open version](https://aminer.org/open-academic-graph). Add it to MAG->Init_data and run mag_preprocessing.py<br />

