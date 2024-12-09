# The Census-Stub Graph Invariant Descriptor

Full paper: https://arxiv.org/abs/2412.04582 (IEEE TVCG doi.org/10.1109/TVCG.2024.3513275)

Supplemental materials: https://osf.io/nmzra/

### Project Overview

An 'invariant descriptor' captures meaningful structural features of networks, useful where traditional visualizations, like node-link views, face challenges like the 'hairball phenomenon' (inscrutable overlap of points and lines). Designing invariant descriptors involves balancing abstraction and information retention, as richer data summaries demand more storage and computational resources. Building on prior work, chiefly the BMatrix—a matrix descriptor visualized as the invariant 'network portrait' heatmap—we introduce BFS-Census, a new algorithm computing our Census data structures: Census-Node, Census-Edge, and Census-Stub. Our experiments show Census-Stub, which focuses on 'stubs' (half-edges), has orders of magnitude greater discerning power (ability to tell non-isomorphic graphs apart) than any other descriptor in this study, without a difficult trade-off: the substantial increase in resolution does not come at a commensurate cost in storage space or computation power. We also present new visualizations—our Hop-Census polylines and Census-Census trajectories—and evaluate them using real-world graphs, including a sensitivity analysis that shows graph topology change maps to visual Census change.

The full codebase (https://osf.io/bgqe7) has three main components: the `EVAL_GraphAtlasCollider` folder, the `EVAL_EclecticNetworkBenchmark` folder, and the `FIG` Python scripts. We also provide the `Supplemental` companion PDF of the main paper. This project was written entirely in Python (v3.12) with additional functions from the NetworkX (v3.2.1), NumPy (v1.26.2), and Matplotlib (v3.8.2) open-source libraries.

Note that this GitHub repo only contains code, all PNG images and files above 100MB have been removed[^1].

## EVAL_GraphAtlasCollider

Graph Atlas Collider is our testing tool that quantifies the discerning power of invariant descriptors computed from the Graph Atlas benchmark dataset. The folder `Graph_Atlas`  stores the complete enumeration of tiny graphs from 3 to 10 nodes, across eight Graph6 files partitioned by order corpus (number of nodes).

The Python script `TinyGraphCollider_collisions.py` takes in the data from Graph Atlas and filters out all the disconnected graph entries. From the fully connected entries, TGC computes nine invariant descriptor data structures from each entry, which are stored as JSON dictionaries in the respective invariant descriptor folder within `TGC_collisions`. These JSON files have string keys that are lossless compressions of the descriptors themselves. The Census invariant descriptors are Census-Node (CN), Census-Edge (CE), Census-Stub (CS), BMatrix-Node (BN), BMatrix-Edge (BE), and BMatrix-Stub (BS).

The Python script `TinyGraphCollider_bytes.py` takes in the data from Graph Atlas and filters out all the disconnected graph entries, and computes for each Graph Atlas entry its collection of invariant descriptors lossless strings and the size in bytes for each. Then, each byte's occurrence frequency is recorded in CSV files within the `TGC_bytes` folder, for each invariant descriptor across the entire Graph Atlas benchmark. This compression analysis includes Graph6 and Edgelist data structures, which are not invariant descriptors, but are included as baseline given these are common file formats for graph storage. Graph Atlas is already stored in Graph6 format.

For collision set membership, the Python script `EVAL_CollisionSetMembership.py` takes in the JSON results from `TGC_collisions` and tags each Graph Atlas entry with a binary tag vector of 3 slots (for CN, CE, and CS, in that order) and exports a CSV file with one Graph Atlas entry per row. From this data, `partition_DATA.py` splits Graph Atlas entries into one of eight intersecting collision sets.

## EVAL_EclecticNetworkBenchmark

We include an eclectic collection of 81 networks to explore how the Heatmap Matrix, Hop-Census, and Census-Census plots visually encode large networks with various topological characteristics. The file `DATA_provenance.csv` contains the data source information for each entry in this collection.

In contrast to Graph Atlas Collider, which uses Graph Atlas as an input dataset, the eclectic collection's larger generated and real-world networks require `BFS_CENSUS_multithreaded.py` to run parallel BFS traversals.

The `DATA` folder has 81 subfolders, each dedicated to a network in the collection. Each subfolder has an edgelist `topology.csv` file that stores the connectivity information, a spatial placement layout `embedding.csv` file that determines the horizontal and vertical location of each node in the graph, and a `statistics.csv` file with common network metrics such as the number of triangles and clustering coefficient. Each subfolder also has PNG images for the 10 visual encodings: 1 node-link view, 3 Hop-Census plots (Census-Node, Census-Edge, Census-Stub), 3 Heatmap Matrix plots (BMatrix-Node, BMatrix-Edge, and BMatrix-Stub), and 3 Census-Census plots (CN-CE, CE-CS, and CN-CS).

The `FIG_VisualEncodingPipeline.py` is the main visual encoding script that computes and renders these normalized static images, without axes or labels. The script `FIG_SelectedResults.py` is a variant of the main pipeline that retains axes numbers and labels. The helper script `EXPORT_9x9.py` collects and converts the standalone PNG outputs into a single 9-by-9 grid collage image.

## FIGURE Python Scripts

Any Python file with the prefix `FIG` renders images that are featured in the main paper and in Supplemental materials.

In the main paper: Fig 2 comes from `FIG_Intro.py`; Fig 4 from `FIG_CensusDataStructure.py`; Fig 5 from `FIG_ColliderResults.py` and `FIG_Storage.py`; Fig 6 from `FIG_ColliderSet.py`; Fig 8 from `FIG_Echo.py`; Fig 9 from `FIG_VisualEncodingPipeline.py`; Fig 10 from `FIG_TriplePlot.py`; Fig 11 from `FIG_ParameterBA_m.py`, `FIG_ParameterER_p.py`, and `FIG_ParameterWS_p.py`; and Fig 12 from `FIG_DiameterLoop.py`.

In Supplemental: Fig S2 comes from `FIG_CumulativeBMatrix.py`; Fig S9 from `FIG_SuppOverviewPlot.py`; Figs S10 to S19 from `FIG_SelectedResults.py`; Figs S20 to S29 from `FIG_VisualEncodingPipeline.py`; and Figs S30 to S39 from `FIG_LocalGlobal.py`.

## Supplemental

This PDF document (https://osf.io/ex5nw) contains five sections: the algorithmic details of the previous network portrait work including illustrated explainers of the BFS-BMatrix and GraphPrism approaches (Supp 1); implementation details of our BFS-Census algorithm (Supp 2); further discussion of our quantitative Graph Atlas Collider tool, including an illustrated example of a 20-node collision noted in previous work (Supp 3); further details of our qualitative analysis through a visual encoding pipeline (Supp 4)—including many additional high-resolution images: full-page views for each of the networks featured in this paper, and full-page collages for each plot type across all 81 networks; and an additional sensitivity analysis results for 81 generated graphs that share topological similarity despite local differences in edge wiring, yet Census plots consistently capture their global similarity (Supp 5).

## Citation

If this research project has been helpful in your work, we kindly ask to cite using the following BibTeX entry:

```BibTeX
@article{oddo2024,
  title={The Census-Stub Graph Invariant Descriptor},
  author={Oddo, Matt I.B. and Kobourov, Stephen and Munzner, Tamara},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  month={December},
  year={2024}
}
```

[^1]: Specifically, the files above 100MB are `EVAL_GraphAtlasCollider/Graph_Atlas/order_10.g6`, `EVAL_GraphAtlasCollider/CollisionSetMembership/CollisionSet_CSV/DATA_10.csv`, `EVAL_GraphAtlasCollider/CollisionSetMembership/CollisionSet_CSV/DATA_10/CN_CE_xx.csv`, and `EVAL_GraphAtlasCollider/CollisionSetMembership/CollisionSet_CSV/DATA_10/CN_xx_xx.csv`.
