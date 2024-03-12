# Exmap
This repository contains code for the paper titled "ExMap: Leveraging Explainability Heatmaps for Unsupervised Group Robustness to Spurious Correlations", to appear at CVPR 2024.

_**Abstract** _
_Group robustness strategies aim to mitigate learned biases in deep learning models that arise from spurious correlations present in their training datasets. However, most existing methods rely on the access to the label distribution of the groups, which is time-consuming and expensive to obtain. As a result, unsupervised group robustness strategies are sought. Based on the insight that a trained model's classification strategies can be inferred accurately based on explainability heatmaps, we introduce ExMap, an unsupervised two stage mechanism designed to enhance group robustness in traditional classifiers. ExMap utilizes a clustering module to infer pseudo-labels based on a model's explainability heatmaps, which are then used during training in lieu of actual labels. Our empirical studies validate the efficacy of ExMap - We demonstrate that it bridges the performance gap with its supervised counterparts and outperforms existing partially supervised and unsupervised methods. Additionally, ExMap can be seamlessly integrated with existing group robustness learning strategies. Finally, we demonstrate its potential in tackling the emerging issue of multiple shortcut mitigation._

## Datasets
The setup for Waterbirds and CelebA are exactly the same as [here](https://github.com/anniesch/jtt/tree/master). For C-MNIST, we use the setup in [[1]. For UrbanCars, we use the setup in [2]. The latter dataset is available privately upon request. 

## Dependencies

Run 
```
pip install -r requirements.txt
```

A smoother alternative is to pull our image from Docker:

```
asl021/whacamole:latest
```

## Running ExMap

### C-MNIST

```
bash
```

### Waterbirds

```
bash
```

### CelebA

```
bash
```

### UrbanCars

```
bash
```

## References

[1] Arjovsky, Martín, Léon Bottou, Ishaan Gulrajani and David Lopez-Paz. “Invariant Risk Minimization.” ArXiv abs/1907.02893 (2019): n. pag.

[2] Li, Zhiheng, I. Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Cantón Ferrer, Chenliang Xu and Mark Ibrahim. “A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 20071-20082.

