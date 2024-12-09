# The Census-Stub Graph Invariant Descriptor

Full paper: https://arxiv.org/abs/2412.04582

Supplemental materials: https://osf.io/nmzra/

An 'invariant descriptor' captures meaningful structural features of networks, useful where traditional visualizations, like node-link views, face challenges like the 'hairball phenomenon' (inscrutable overlap of points and lines). Designing invariant descriptors involves balancing abstraction and information retention, as richer data summaries demand more storage and computational resources. Building on prior work, chiefly the BMatrix—a matrix descriptor visualized as the invariant 'network portrait' heatmap—we introduce BFS-Census, a new algorithm computing our Census data structures: Census-Node, Census-Edge, and Census-Stub. Our experiments show Census-Stub, which focuses on 'stubs' (half-edges), has orders of magnitude greater discerning power (ability to tell non-isomorphic graphs apart) than any other descriptor in this study, without a difficult trade-off: the substantial increase in resolution does not come at a commensurate cost in storage space or computation power. We also present new visualizations—our Hop-Census polylines and Census-Census trajectories—and evaluate them using real-world graphs, including a sensitivity analysis that shows graph topology change maps to visual Census change.

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
