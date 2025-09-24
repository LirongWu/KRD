#  Knowledge-inspired Reliable Distillation (KRD)

This is a PyTorch implementation of Knowledge-inspired Reliable Distillation (KRD), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv)

* Two evaluation settings: transductive and inductive

* Various teacher GNN architectures (GCN, SAGE, GAT) and student MLPs

* Training paradigm for teacher GNNs and student MLPs

* Visualization and evaluation metrics 

  

## Main Requirements

* numpy==1.21.6
* scipy==1.7.3
* torch==1.6.0
* dgl == 0.6.1
* sklearn == 1.0.2



## Description

* train_and_eval.py  
  * train_teacher() -- Pre-train the teacher GNNs
  * train_student() -- Train the student MLPs with the pre-trained teacher GNNs
* models.py  
  
  * MLP() -- student MLPs
  * GCN() -- GCN Classifier, working as teacher GNNs
  * GAT() -- GAT Classifier, working as teacher GNNs
  * GraphSAGE() -- GraphSAGE Classifier, working as teacher GNNs
  * Com_KD_Prob() -- Calculate knowledge reliability, model sampling probability, and updating power in a momentum manner
* dataloader.py  

  * load_data() -- Load Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv datasets
* utils.py  
  * set_seed() -- Set radom seeds for reproducible results
  * graph_split() -- Split the data for the inductive setting




## Running the code

1. Install the required dependency packages

3. To get the results on a specific *dataset* with specific *GNN* as the teacher under a specific *setting*, please run with proper hyperparameters:

  ```
python train.py --dataset data_name --teacher gnn_name --exp_setting setting_name
  ```

where (1) *data_name* is one of the seven datasets: Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy  ogbn-arxiv; (2) *gnn_name* is one of the three GNN architectures: GCN, SAGE, and GAT; (3) *setting_name* is one of the two evaluation settings: 0 (Transductive) and 1 (Inductive). Take the model in the transductive setting  (with GCN as the teacher) on the *Citeseer* dataset as an example: 

```
python train.py --dataset citeseer --teacher GCN --exp_setting 0
```



## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{wu2023quantifying,
  title={Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs},
  author={Wu, Lirong and Lin, Haitao and Huang, Yufei and and Li, Stan Z},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
