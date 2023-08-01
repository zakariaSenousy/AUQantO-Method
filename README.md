# AUQantO-Method

In this repository, we present a model agnostic mechanism, coined Actionable Uncertainty Quantification Optimization (AUQantO), to optimize the performance of deep learning architectures for medical image classification. AUQantO is guided by uncertainty measurements that help clinical experts refine annotations to develop more reliable DCNN models. AUQantO employs either an entropy-based mechanism or a Monte-Carlo (MC) dropout technique to measure uncertainty in images, where a new hyperparameter (i.e., a threshold) is introduced and optimized. Our motivation stems from the notion that, despite the abundance of deep learning architectures and their significant potential for reducing the workload strain on medical experts, a small percentage of low quality or indecisive medical images would necessitate the aid of medical experts.



![AUQantO_Diagram](https://user-images.githubusercontent.com/20457990/232609227-6281e6d6-fbfa-4f2e-88c9-ced56a0a0863.png)


## Updates and Citation

Please cite this following paper if you use the code.

@article{SENOUSY2023110666,
title = {AUQantO: Actionable Uncertainty Quantification Optimization in deep learning architectures for medical image classification},
journal = {Applied Soft Computing},
pages = {110666},
year = {2023},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2023.110666},
url = {https://www.sciencedirect.com/science/article/pii/S1568494623006841},
author = {Zakaria Senousy and Mohamed Medhat Gaber and Mohammed M. Abdelsamea},
keywords = {Medical image analysis, Image classification, Deep learning, Convolutional neural networks, Uncertainty quantification, Actionability, XAI},
abstract = {Deep learning algorithms have the potential to automate the examination of medical images obtained in clinical practice. Using digitized medical images, convolution neural networks (CNNs) have demonstrated their ability and promise to discriminate among different image classes. As an initial step towards explainability in clinical diagnosis, deep learning models must be exceedingly precise, offering a measure of uncertainty for their predictions. Such uncertainty-aware models can help medical professionals in detecting complicated and corrupted samples for re-annotation or exclusion. This paper proposes a new model and data-agnostic mechanism, called Actionable Uncertainty Quantification Optimization (AUQantO) to improve the performance of deep learning architectures for medical image classification. This is achieved by optimizing the hyperparameters of the proposed entropy-based and Monte Carlo (MC) dropout uncertainty quantification techniques escorted by single- and multi-objective optimization methods, abstaining from the classification of images with a high level of uncertainty. This helps in improving the overall accuracy and reliability of deep learning models. To support the above claim, AUQantO has been validated with four deep learning architectures on four medical image datasets and using various performance metric measures such as precision, recall, Area Under the Receiver Operating Characteristic (ROC) Curve score (AUC), and accuracy. The study demonstrated notable enhancements in deep learning performance, with average accuracy improvements of 1.76% and 2.02% for breast cancer histology and 5.67% and 4.24% for skin cancer datasets, utilizing two uncertainty quantification techniques, and AUQantO further improved accuracy by 1.41% and 1.31% for brain tumor and 4.73% and 1.83% for chest cancer datasets while allowing exclusion of images based on confidence levels.}
}

```
