# Quadratic Objective Perturbation Mechanism

Code accompanying [[1]](#1), which introduces Quadratic Objective Perturbation (QOP), a curvature-based objective perturbation mechanism for differential privacy. QOP is compared with Linear Objective Perturbation (LOP), introduced in [[2]](#2), and Clipped Linear Objective Perturbation (LOP-Clip), introduced in [[3]](#3).

## How to Run

To run the code, install the dependencies and run the notebooks.

```bash
python3 -m pip install -r requirements.txt
```

## What it does

The repository contains two main notebooks:

1. `F1_UtilityBound.ipynb`: plots the optimal empirical loss upper bounds derived in the paper.
2. `F2_Comparison.ipynb`: compares the empirical performance of QOP, LOP and LOP-Clip.

## References

<a id="1">[1]</a> Cortild, D., & Cartis, C. (2026). Quadratic Objective Perturbation: Curvature-Based Differential Privacy. arXiv preprint arXiv:2605.XXXXX.

<a id="2">[2]</a> Chaudhuri, K., Monteleoni, C., & Sarwate, A. D. (2011). Differentially Private Empirical Risk Minimization. Journal of Machine Learning Research, 12, 1069–1109.

<a id="3">[3]</a> Redberg, R. E., Koskela, A., & Wang, Y.-X. (2023). Improving the Privacy and Practicality of Objective Perturbation for Differentially Private Linear Learners. Proceedings of the 37th Conference on Neural Information Processing Systems. 