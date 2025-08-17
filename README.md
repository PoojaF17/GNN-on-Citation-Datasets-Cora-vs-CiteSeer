# GNN-on-Citation-Datasets-Cora-vs-CiteSeer
Comparative Study of Graph Neural Networks on Citation Datasets (Cora vs CiteSeer)

**Project Overview**
This project investigates the performance of popular Graph Neural Networks (GNNs) on two well-known citation graph datasets: Cora and CiteSeer.The aim is to evaluate how different GNN architectures generalize across small-scale citation networks in the Computer Science domain.

We compare three widely used models:
GCN (Graph Convolutional Network)
GAT (Graph Attention Network)
GraphSAGE (Graph Sample & Aggregate)

**Datasets**
-Cora
Contains 2,708 scientific publications classified into 7 categories.
Each paper is represented by a 1433-dimensional bag-of-words feature vector.
Citation links form the graph edges.

-CiteSeer
Contains 3,327 scientific publications classified into 6 categories.
Papers are described by 3703-dimensional sparse bag-of-words vectors.
Edges represent citation relationships.

Both datasets are standard benchmark datasets in the graph ML community, making them ideal for model comparison.

**Methodology**

1.Dataset Selection & Preprocessing
  Used Planetoid benchmark datasets (Cora and CiteSeer) from PyTorch Geometric.
  Preprocessed graph data (node features, edges, labels) with standard train/val/test masks.

2.Model Implementation
  Implemented three Graph Neural Network (GNN) architectures:
      GCN (Graph Convolutional Network)
      GAT (Graph Attention Network)
      GraphSAGE (Graph Sample & Aggregate)

3.Training Setup
  Trained each model for 200 epochs with Adam optimizer (lr = 0.01, weight decay = 5e-4).
  Applied cross-entropy loss on training nodes and dropout (0.5) for regularization.

4.Evaluation
  Computed Accuracy, F1-Macro, and F1-Weighted on test splits.
  Visualized results with Confusion Matrices and UMAP embeddings for node separability.

5.Comparative Analysis
  Compared models across both datasets to identify performance differences.
  Drew insights on the role of dataset complexity vs. model architecture sophistication.
  
**Results**
Dataset	Model	Accuracy	F1-macro	F1-weighted
0	Cora	GCN	0.808	0.802698	0.808869
1	Cora	GAT	0.793	0.789161	0.795424
2	Cora	GraphSAGE	0.801	0.791158	0.801679
3	CiteSeer	GCN	0.687	0.654745	0.688686
4	CiteSeer	GAT	0.690	0.660551	0.698728
5	CiteSeer	GraphSAGE	0.683	0.657122	0.692107

**Key Takeaways**
  -Cora outperforms CiteSeer across all models, highlighting that dataset structure and feature quality strongly influence GNN performance.
  -On Cora, all three models perform similarly (~80% accuracy), showing that even simpler architectures like GCN remain competitive.
  -On CiteSeer, performance drops to ~68–69%, suggesting higher class overlap and less informative features make learning harder.
  -GCN achieved the best balance on both datasets (Cora: 80.8%, CiteSeer: 68.7%), while GAT and GraphSAGE did not provide clear improvements, contrary to expectations from larger-scale benchmarks.
  -These results emphasize the importance of dataset characteristics over model complexity in smaller citation networks.
