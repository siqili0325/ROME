# Robust Mixture Models for Algorithmic Fairness Under Latent Heterogeneity

[![arXiv](https://img.shields.io/badge/arXiv-2509.17411-b31b1b.svg)](https://arxiv.org/abs/2509.17411)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Language](https://img.shields.io/badge/Language-R%20%7C%20Python-blue.svg)](#)

This repository contains the official R and Python implementations for **ROME** (**RO**bust **M**ixture **E**nsemble), a framework for group-robust prediction under latent population heterogeneity.

---

## 📌 Overview

Machine learning models optimized for average performance can perform poorly on vulnerable subpopulations.
Existing group-robust approaches often rely on groups specified in advance, yet fairness-relevant subgroup structure may be **latent**, **intersectional**, and driven by complex interactions among continuous and discrete attributes.
At the same time, directly conditioning an outcome model on sensitive attributes **S** may be restricted by institutional, policy, or legal requirements — even when **S** is valuable for identifying vulnerable subpopulations.

**ROME** addresses this tension by learning latent group structure from data while optimizing worst-group predictive performance, connecting latent-variable modeling with distributionally robust optimization (DRO).

### 🌟 Key Features

* **No predefined group labels required.** ROME discovers latent subpopulations directly from data, handling intersectionality without combinatorial explosion.
* **Principled use of sensitive attributes.** Sensitive features **S** inform subgroup discovery through gating or membership models but are excluded from outcome prediction, respecting fairness constraints by design.
* **Two complementary formulations:**
  - **ROME-EM** — Expectation-Maximization with closed-form DRO aggregation for linear models, grounded in the maximin aggregation theory of [Wang et al. (2023)](https://arxiv.org/abs/2309.02211).
  - **ROME-MoE** — Neural Mixture-of-Experts with a DRO training objective for nonlinear settings, jointly learning soft subgroup structure end-to-end.
* **Competitive performance.** Evaluated against six baselines spanning standard prediction, oracle group-DRO, fair regression (BGL), adversarial reweighting (ARL), and mixture-of-experts, across three real-world regression datasets.

## ✍️ Citation

If you find ROME useful for your research, please consider citing:

```bibtex
@article{li2025rome,
  title={Robust Mixture Models for Algorithmic Fairness Under Latent Heterogeneity},
  author={Li, Siqi and Liu, Molei and Tian, Ziye and Hong, Chuan and Liu, Nan},
  journal={arXiv preprint arXiv:2509.17411},
  year={2025}
}
```

---

## 👥 Contact

* **Siqi Li** — <siqili@u.duke.nus.edu>

For questions, issues, or collaborations, please open a GitHub Issue or reach out via email.
