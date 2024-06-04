# BEAG
 
---

- Implementation of [Breadth-First Exploration on Adaptive Grid for Reinforcement Learning](https://openreview.net/forum?id=59MYoLghyk) (ICML 2024) in PyTorch.
- Our code is based on official implementation of [DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning](https://github.com/jayLEE0301/dhrl_official).

## Installation 
 ---
- create conda environment
```
conda create -n beag python=3.7
conda activate beag
```
- if permission denied,
```
chmod +x ./scripts/*.sh
```

## Experiments
---
- To reproduce our experiments, please run the script provided below
- ./scripts/{ENV}.sh {GPU} {SEED}
```
example
./scripts/Reacher.sh 0 1
./scripts/AntMaze.sh 2 3
```

## Citation
---
```bibtex
@inproceedings{yoon2024beag,
  title={Breadth-First Exploration on Adaptive Grid for Reinforcement Learning},
  author={Yoon, Youngsik and Lee, Gangbok and Ahn, Sungsoo and Ok, Jungseul},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=59MYoLghyk}
```