# Exmap
This repository contains code for the paper titled "ExMap: Leveraging Explainability Heatmaps for Unsupervised Group Robustness to Spurious Correlations", to appear at CVPR 2024. We are updating this repo and the complete code will be available soon!

_**Abstract**_ - 
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

## Dataset access

### CMNIST
The Colored MNIST dataset does not need to be installed as it downloads the MNIST dataset automatically when running. The dataset is build following the procedure described in ["Invariant Risk Minimization"](https://arxiv.org/abs/1907.02893) .

### Waterbirds
To acquire the Waterbirds dataset we use the [following instructions](https://github.com/kohpangwei/group_DRO#waterbirds) from the group DRO code, where we install the tarball of the dataset.


### Foreground-only Waterbirds
In addition to the installed Waterbirds dataset, we download the segmentation masks for the birds (foreground objects) from [here](https://data.caltech.edu/records/w9d68-gec53). We then merge the directories with the original images (of type ".jpg") and the directories with the segmentation masks (of type ".png"). Below is an example of how the images are stored in the final directory:

```
waterbird_complete95_forest2water2\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg
waterbird_complete95_forest2water2\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.png
```

### CelebA
For CelebA we follow the [instructions](https://github.com/kohpangwei/group_DRO#celeba) from the group DRO code. Additionally, we copy the `celeba_metadata.csv` from the DFR code, described [here](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access).

### Urbancars
As of this time, the Urbancars dataset is not publicly available. It was made available to us after contacting the Whac-a-mole paper's authors.


### Running on each dataset

ERM, ExMap, and DFR can be run on each of the datasets by navigating to scripts folder and running the particular shell script (e.g. bash run_waterbirds.sh). This will automatically run all the methods on the particular dataset. 

## Save Models

If you wish to have access to our saved models on each dataset for better reproducibility, please reach out at rwiddhi.chakraborty@uit.no.

## References

[1] Arjovsky, Martín, Léon Bottou, Ishaan Gulrajani and David Lopez-Paz. “Invariant Risk Minimization.” ArXiv abs/1907.02893 (2019): n. pag.

[2] Li, Zhiheng, I. Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Cantón Ferrer, Chenliang Xu and Mark Ibrahim. “A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 20071-20082.

