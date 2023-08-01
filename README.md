# AUQantO-Method

In this repository, we present a model agnostic mechanism, coined Actionable Uncertainty Quantification Optimization (AUQantO), to optimize the performance of deep learning architectures for medical image classification. AUQantO is guided by uncertainty measurements that help clinical experts refine annotations to develop more reliable DCNN models. AUQantO employs either an entropy-based mechanism or a Monte-Carlo (MC) dropout technique to measure uncertainty in images, where a new hyperparameter (i.e., a threshold) is introduced and optimized. Our motivation stems from the notion that, despite the abundance of deep learning architectures and their significant potential for reducing the workload strain on medical experts, a small percentage of low quality or indecisive medical images would necessitate the aid of medical experts.



![AUQantO_Diagram](https://user-images.githubusercontent.com/20457990/232609227-6281e6d6-fbfa-4f2e-88c9-ced56a0a0863.png)


## Citation
If you use this code for your research, please cite our paper: [AUQantO: Actionable Uncertainty Quantification Optimization in deep learning architectures for medical image classification](https://doi.org/10.1016/j.asoc.2023.110666)


```
@article{SENOUSY2023110666,
title = {AUQantO: Actionable Uncertainty Quantification Optimization in deep learning architectures for medical image classification},
journal = {Applied Soft Computing},
pages = {110666},
year = {2023},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2023.110666},
url = {https://www.sciencedirect.com/science/article/pii/S1568494623006841},
author = {Zakaria Senousy and Mohamed Medhat Gaber and Mohammed M. Abdelsamea}
}

```
