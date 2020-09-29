# Multi-task Learning for Influence Estimation and Maximization
Code and instructions to reproduce the analysis in the [paper](https://arxiv.org/abs/1904.08804).

You can find online videos that describe [IMINFECTOR](https://www.youtube.com/watch?v=x28jgYW6I3M&t=322s) and its previous [variant](https://www.youtube.com/watch?v=LoKQUcq2KTM&list=LLpiK7loHj0_zMHjndIT_dYA&index=9&t=28s).


``` bash
mkdir Code Data Figures
cd Code
git clone https://github.com/GiorgosPanagopoulos/Influence-Maximization-via-Representation-Learning
```

## Infector
![infector architecture](/figures/INFECTOR.png) 


## Requirements
To run this code you will need the following python packages: 
* [numpy](https://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)
* [tensorflow-gpu](https://www.tensorflow.org/)
* [igraph](https://igraph.org/python/)
* [pyunpack](https://pypi.org/project/pyunpack/)
* [patool](https://pypi.org/project/patool/)

which can be installed using the requirements.txt:

``` bash
pip install -r requirements.txt
```

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
The main function will derive and evaluate the seed sets of the two metrics and IMINFECTOR as well as the input for the baseline methods. <br /> 
However, some the baselines need to be run separately from their original codes, found in these locations: <br /> 
[Credit Distribution and Simpath](https://www.cs.ubc.ca/~goyal/code-release.php) <br /> 
[IMM](https://sourceforge.net/p/im-imm/wiki/Home/)

Run with default parameters for *sampling percentage*, *learning rate*, *number of epochs*, *embeddings size* and *number of negative samples*.

``` bash
python main --sampling_perc=120 learning_rate=0.1 --n_epochs=5 --embedding_size=50 --num_neg_samples=10
```

## Plots
Manually change the three paths to the directories of the datasets in plot_precision.R and plot_spreading.R and run them.

## Reference
If you use this work, please cite:
```
@inproceedings{panagopoulos2020influence,
  title={Influence Maximization Using Influence and Susceptibility Embeddings},
  author={Panagopoulos, George and Malliaros, Fragkiskos D and Vazirgianis, Michalis},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={14},
  pages={511--521},
  year={2020}
}
```
```
@article{panagopoulos2019influence,
  title={Multi-task Learning for Influence Estimation and Maximization},
  author={Panagopoulos, George and Vazirgiannis, Michalis and Malliaros, Fragkiskos D},
  journal={arXiv preprint arXiv:1904.08804},
  year={2019}
}
```

