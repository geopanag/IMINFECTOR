# Influence Maximization via Representation Learning
Code and instructions to reproduce the analysis in the [paper](https://arxiv.org/abs/1904.08804)

``` bash
mkdir Code Data Figures
cd Code
git clone https://github.com/GiorgosPanagopoulos/Influence-Maximization-via-Representation-Learning
pip install -r requirements.txt
```

## Infector
![infector architecture](https://github.com/GiorgosPanagopoulos/Influence-Maximization-via-Representation-Learning/figures/infector-scheme.pdf)

## Data
All datasets need certain preprocessing before the experiments. 

``` bash
python preprocessing
```
The script creates the required folder structure for every dataset (Digg, Weibo, MAG)->Init_Data,Embeddings, Seeds, Spreading.
It then downloads the [Digg](https://www.isi.edu/~lerman/downloads/digg2009.html) and 
[Weibo](https://aminer.org/influencelocality) datasets, and preprocesses them for curation and derivation of the network and the diffusion cascades.<br />
To derive the MAG network and diffusion cascades, we employed the tables Paper, Paper References, Author, PaperAuthorAffiliation, Fields of Study, Paper Fields of Study from the official [MAG](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/). 
There is also an [open version](https://aminer.org/open-academic-graph). 
Add these datasets to "MAG/Init_data" and run mag_preprocessing.py<br />


## Run
Run with default parameters for *sampling percentage*, *learning rate*, *number of epochs*, *embeddings size* and *number of negative samples*.

``` bash
python main --sampling_perc=120 learning_rate=0.1 --n_epochs=5 --embedding_size=50 --num_neg_samples=10
```

## Plots
Manually change the three paths to the directories of the datasets in plot_precision.R and plot_spreading.R and run them.

## Reference
If you use this work, please cite:
```
@article{panagopoulos2019influence,
  title={Influence Maximization via Representation Learning},
  author={Panagopoulos, George and Vazirgiannis, Michalis and Malliaros, Fragkiskos D},
  journal={arXiv preprint arXiv:1904.08804},
  year={2019}
}
```

