from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import umap

def plot_scatter(data, y, groups, title):
    assert len(data) == len(y) == len(groups)
    assert y.max() <= 1 
    color_calls = -1
    plt.title(title)
    for i in np.unique(groups):
        # cycle colors until desired is reached
        while color_calls != i:
            color = plt.scatter(1,1).get_facecolor()[0]
            color_calls += 1

        for j, m in enumerate(["x", "o"]):
            mask = (groups==i) & (y==j)
            color_array = np.repeat(np.array([color]), len(data[mask]), axis=0)
            if mask.sum() > 0:
                plt.scatter(data[mask,0], data[mask,1], label=f"Class {j} cluster {i}", marker=m, c=color_array)
    plt.legend() 

def plot_pred(data, y, preds, title):
    assert len(data) == len(y) == len(preds)
    assert y.max() <= 1
    assert preds.max() <= 1
    plt.title(title)
    plt.scatter(data[y==preds,0], data[y==preds,1], label="Correct", c="black")
    plt.scatter(data[y!=preds,0], data[y!=preds,1], label="Incorrect", c="red")
    plt.legend()

def plot_data(args, heatmaps, y, preds, g, local_groups, global_groups):
    heatmaps = np.array([heatmaps[i] / np.abs(heatmaps[i]).max() for i in range(len(heatmaps))])
    
    if args.plot_type == "tsne":
        # number of knn used in spectral divided by 3: 
        perplex = lambda n: int(np.log(n)) / 3  
        #https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/manifold/_t_sne.py#L939

        X_y0 = TSNE(n_components=2, perplexity=perplex(len(heatmaps[y==0])), n_iter=1000, random_state=args.seed, n_jobs=1).fit_transform(heatmaps[y==0])
        X_y1 = TSNE(n_components=2, perplexity=perplex(len(heatmaps[y==1])), n_iter=1000, random_state=args.seed, n_jobs=1).fit_transform(heatmaps[y==1])
        X_all = TSNE(n_components=2, perplexity=perplex(len(heatmaps)), n_iter=1000, random_state=args.seed, n_jobs=1).fit_transform(heatmaps)
    elif args.plot_type == "umap":
        knn = lambda n: int(np.log(n))
        X_y0 = umap.UMAP(n_components=2, n_neighbors=knn(len(heatmaps[y==0]))).fit_transform(heatmaps[y==0])
        X_y1 = umap.UMAP(n_components=2, n_neighbors=knn(len(heatmaps[y==1]))).fit_transform(heatmaps[y==1])
        X_all = umap.UMAP(n_components=2, n_neighbors=knn(len(heatmaps))).fit_transform(heatmaps)

    plt.figure(figsize=(40,30))
    plt.tight_layout()
    plt.subplot(3,4,1)
    plot_scatter(X_all, y, g, "Ground truth all")
    plt.subplot(3,4,2)
    plot_scatter(X_all, y, local_groups, "Pseudolabels all (per class)")
    plt.subplot(3,4,3)
    plot_scatter(X_all, y, global_groups, "Pseudolabels all")
    plt.subplot(3,4,4)
    plot_pred(X_all, y, preds, "Predictions all")

    plt.subplot(3,4,5)
    plot_scatter(X_y0, y[y==0], g[y==0], "Ground truth for class y0")
    plt.subplot(3,4,6)
    plot_scatter(X_y0, y[y==0], local_groups[y==0], "Pseudolabels for class y0 (per class)")
    plt.subplot(3,4,7)
    plot_scatter(X_y0, y[y==0], global_groups[y==0], "Pseudolabels for class y0")
    plt.subplot(3,4,8)
    plot_pred(X_y0, y[y==0], preds[y==0], "Predictions for class y0")

    plt.subplot(3,4,9)
    plot_scatter(X_y1, y[y==1], g[y==1], "Ground truth for class y1")
    plt.subplot(3,4,10)
    plot_scatter(X_y1, y[y==1], local_groups[y==1], "Pseudolabels for class y1 (per class)")
    plt.subplot(3,4,11)
    plot_scatter(X_y1, y[y==1], global_groups[y==1], "Pseudolabels for class y1")
    plt.subplot(3,4,12)
    plot_pred(X_y1, y[y==1], preds[y==1], "Predictions for class y1")


    plt.savefig(os.path.join(args.results_dir, f"{args.plot_type}.png"))