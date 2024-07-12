"""Evaluate DFR on spurious correlations datasets."""

import torch

import numpy as np
import os
import sys
import tqdm
import json
import pickle

from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from contextlib import redirect_stdout, redirect_stderr

import models
import utils
from utils import supervised_utils


# WaterBirds
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
REG = "l1"


def get_args():
    parser = utils.get_model_dataset_args()


    parser.add_argument(
        "--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--result_path", type=str, default="logs/",
        help="Path to save results")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--save_embeddings", action='store_true',
        help="Save embeddings on disc")
    parser.add_argument(
        "--predict_spurious", action='store_true',
        help="Predict spurious attribute instead of class label")
    parser.add_argument(
        "--drop_group", type=int, default=None, required=False,
        help="Drop group from evaluation")
    parser.add_argument(
        "--log_dir", type=str, default="", help="For loading wandb results")
    parser.add_argument(
        "--save_linear_model", action='store_true', help="Save linear model weights")
    parser.add_argument(
        "--save_best_epoch", action='store_true', help="Save best epoch num to pkl")
    # DFR TR
    parser.add_argument(
        "--use_train", action='store_true', help="Use train data for reweighting")
    # modify group labels 
    parser.add_argument(
        "--group_label_type", type=str, nargs="+", default="default", choices=["default", "classes", "random", "random_within_class", "local_cluster", "global_cluster"], 
        help="Group label modification for validation set. [default is no modification]") 
    parser.add_argument(
        "--quiet", action="store_true", help="Don't print all results, just final WGA and mean")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to train on")


    args = parser.parse_args()
    args.num_minority_groups_remove = 0
    args.reweight_groups = False
    args.reweight_spurious = False
    args.reweight_classes = False
    args.no_shuffle_train = True
    args.mixup = False
    args.load_from_checkpoint = True
    return args


def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True, num_retrains=1, seed=1):

    worst_accs = {}
    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1

        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_train = x_val[idx[n_val:]]
        y_train = y_val[idx[n_val:]]
        g_train = g_val[idx[n_val:]]

        n_groups = np.max(g_train) + 1
        g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx if len(g) > 5])
        for g in g_idx:
            np.random.shuffle(g)
        x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
        y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
        g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        print("Val tuning:", np.bincount(g_train))
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear", random_state=seed)
            logreg.fit(x_train, y_train)
            preds_val = logreg.predict(x_val)
            group_accs = np.array(
                [(preds_val == y_val)[g_val == g].mean()
                 for g in range(n_groups) if (g_val == g).sum() > 0])
            worst_acc = np.min(group_accs)
            if i == 0:
                worst_accs[c] = worst_acc
            else:
                worst_accs[c] += worst_acc
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        args, c, all_embeddings, all_y, all_g, group_weights, target_type="target", num_retrains=20,
        preprocess=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["val"])
    for i in range(num_retrains):
        for _ in range(20):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]
            n_groups = np.max(g_val) + 1
            g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx if len(g) > 5])
            for g in g_idx:
                np.random.shuffle(g)
            x_train = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_train = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_train = np.concatenate([g_val[g[:min_g]] for g in g_idx])

            if np.any(np.unique(y_train) != np.unique(all_y["val"])):
                # do we need the same thing in tuning?
                print("missing classes, reshuffling...")
                continue
            else:
                break

        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear", random_state=args.seed)
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    print(np.bincount(g_train))
    print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear", random_state=args.seed)
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups_train = np.max(g_train) + 1
    n_groups_test = np.max(g_test) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups_test)]
    # test_mean_acc = (preds_test == y_test).mean()  #TODO: make this a weighted average based on train group sizes
    test_mean_acc = np.nansum([test_accs[g] * group_weights[g] for g in range(n_groups_test)])
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups_train) if (g_train == g).sum() > 0]

    if preprocess:
        x_val = scaler.transform(x_val)
    preds_val = logreg.predict(x_val)
    n_groups_val = np.max(g_val) + 1
    val_accs = [(preds_val == y_val)[g_val == g].mean()
                for g in range(n_groups_val) if (g_val == g).sum() > 0]
    # val_accs = np.nanmin(val_accs)
    val_accs = (preds_val == y_val).mean()
    

    if args.save_linear_model:
        linear_model = {
            'coef': logreg.coef_,
            'intercept': logreg.intercept_,
            'scaler': scaler
        }
        dir_linear_model = os.path.join(os.path.dirname(args.result_path), 'dfr_linear_models')
        if not os.path.isdir(dir_linear_model):
            os.makedirs(dir_linear_model)
        linear_model_path = os.path.join(dir_linear_model,
                                         os.path.basename(args.result_path)[:-4] + f'_linear_model_{target_type}.pkl')
        with open(linear_model_path, 'wb') as f:
            pickle.dump(linear_model, f)

    return test_accs, test_mean_acc, train_accs, val_accs


def modify_val_group_label_types(args, val_split):
    args.max_samples = len(val_split.dataset) if args.max_samples is None else args.max_samples
    val_split.dataset.y_array = val_split.dataset.y_array[:args.max_samples]
    val_split.dataset.group_array = val_split.dataset.group_array[:args.max_samples]
    val_split.dataset.spurious_array = val_split.dataset.spurious_array[:args.max_samples]
    val_split.dataset._count_attributes()

    g = val_split.dataset.group_array
    y = val_split.dataset.y_array

    if args.group_label_type == "default":
        return val_split
    elif args.group_label_type == "classes":
        val_split.dataset.group_array = y 
        val_split.dataset.n_groups = 2
        val_split.dataset.n_spurious = 1
    elif args.group_label_type == "local_cluster":
        pseudo_labels = np.load(os.path.join(args.output_dir, "local_cluster_estimation.npy"))
        assert len(pseudo_labels) == len(g)
        val_split.dataset.group_array = pseudo_labels
        val_split.dataset.n_spurious = pseudo_labels[y==0].max() + 1
    elif args.group_label_type == "global_cluster":
        val_split.dataset.spurious_array = np.load(os.path.join(args.output_dir, "global_cluster_estimation.npy"))
        val_split.dataset.n_spurious = val_split.dataset.spurious_array.max() + 1
        val_split.dataset._get_class_spurious_groups()
    else:
        raise ValueError(f"Unknown group label type {args.group_label_type}")
    
    val_split.dataset._count_groups() #update group_counts and n_groups
    print("\n\nModifying val group label types to ", args.group_label_type)
    return val_split

def log_group_info(logger, data, name, get_ys_func):
    logger.write(f'Modified {name} Data (total {len(data)})\n')
    print("N groups ", data.n_groups)
    for group_idx in range(data.n_groups):
        if data.group_counts[group_idx] == 0:
            continue
        y_idx, s_idx = get_ys_func(group_idx)
        logger.write(
            f'    Group {group_idx} (y={y_idx}, s={s_idx}):'
            f' n = {data.group_counts[group_idx]:.0f}\n')


def main(args):
    print(args)

    # Load data
    logger = utils.Logger() #if not has_wandb else None
    train_loader, test_loader_dict, get_ys_func = (
        utils.get_data(args, logger, contrastive=False))

    # train_loader.dataset.transform = test_loader_dict['val'].dataset.transform # remove left right flip
    
    n_classes = train_loader.dataset.n_classes
    train_distribution = np.bincount(train_loader.dataset.group_array) / len(train_loader.dataset.group_array)

    # modify group label types
    if args.use_train:
        retrain_loader = train_loader 
        name = "train"
    else:
        retrain_loader = test_loader_dict['val']
        name = "val"

    retrain_loader = modify_val_group_label_types(args, retrain_loader)
    get_ys_func2 = partial(utils.get_y_s, n_spurious_first_class=retrain_loader.dataset.n_spurious)

    log_group_info(logger, retrain_loader.dataset, name, get_ys_func2)


    # Model
    model_cls = getattr(models, args.model)
    model = model_cls(n_classes)
    if args.ckpt_path and args.load_from_checkpoint:
        print(f"Loading weights {args.ckpt_path}")
        ckpt_dict = torch.load(args.ckpt_path)
        try:
            model.load_state_dict(ckpt_dict)
        except:
            print("Loading one-output Checkpoint")
            w = ckpt_dict["fc.weight"]
            w_ = torch.zeros((2, w.shape[1]))
            w_[1, :] = w
            b = ckpt_dict["fc.bias"]
            b_ = torch.zeros((2,))
            b_[1] = b
            ckpt_dict["fc.weight"] = w_
            ckpt_dict["fc.bias"] = b_
            model.load_state_dict(ckpt_dict)
    else:
        print("Using initial weights")
    model.cuda()
    model.eval()

    # Evaluate model
    print("Base Model")
    base_model_results = supervised_utils.eval(model, test_loader_dict)
  
    temp = {}
    for name, accs in base_model_results.items():
        if name == "test":
            temp[name] = utils.get_results(accs, get_ys_func, weighting=train_distribution)
        elif name == "val":
            if args.use_train:
                temp[name] = utils.get_results(accs, get_ys_func, weighting=train_distribution)
            else:
                temp[name] = utils.get_results(accs, get_ys_func2)
        else:
            raise ValueError(f"Unknown split {name}")
    
    base_model_results = temp

    print(base_model_results)
    print()
    
    model.fc = torch.nn.Identity()
    #splits = ["test", "val"]
    splits = {
        "test": test_loader_dict["test"],
        "val": test_loader_dict["val"]
    }
    if args.use_train:
        splits["train"] = train_loader
    print(splits.keys())
    if os.path.exists(f"{args.result_path[:-4]}.npz"):
        print("Loading embeddings, labels and groups from existing file. Note this may lead to incorrect results if the data has changed.")
        arr_z = np.load(f"{args.result_path[:-4]}.npz")

        all_embeddings = {split: arr_z[f"embeddings_{split}"] for split in splits}
        all_y = {split: arr_z[f"y_{split}"] for split in splits}
        all_p = {split: arr_z[f"p_{split}"] for split in splits}
        all_g = {split: arr_z[f"g_{split}"] for split in splits}
    else:
        all_embeddings = {}
        all_y, all_p, all_g = {}, {}, {}
        for name, loader in splits.items():
            all_embeddings[name] = []
            all_y[name], all_p[name], all_g[name] = [], [], []
            for x, y, g, p in tqdm.tqdm(loader):
                with torch.no_grad():
                    all_embeddings[name].append(model(x.cuda()).detach().cpu().numpy())
                    all_y[name].append(y.detach().cpu().numpy())
                    all_g[name].append(g.detach().cpu().numpy())
                    all_p[name].append(p.detach().cpu().numpy())
            all_embeddings[name] = np.vstack(all_embeddings[name])
            all_y[name] = np.concatenate(all_y[name])
            all_g[name] = np.concatenate(all_g[name])
            all_p[name] = np.concatenate(all_p[name])

        if args.save_embeddings:
            np.savez(f"{args.result_path[:-4]}.npz",
                     embeddings_test=all_embeddings["test"],
                     embeddings_val=all_embeddings["val"],
                     y_test=all_y["test"],
                     y_val=all_y["val"],
                     g_test=all_g["test"],
                     g_val=all_g["val"],
                     p_test=all_p["test"],
                     p_val=all_p["val"],
                    )

    if args.drop_group is not None:
        print("Dropping group", args.drop_group)
        all_masks = {name: all_g[name] != args.drop_group for name in splits}
        for name in splits:
            all_y[name] = all_y[name][all_masks[name]]
            all_g[name] = all_g[name][all_masks[name]]
            all_p[name] = all_p[name][all_masks[name]]
            all_embeddings[name] = all_embeddings[name][all_masks[name]]
    
    if args.use_train:
        print("Reweighting on training data")
        all_y["val"] = all_y["train"]
        all_g["val"] = all_g["train"]
        all_p["val"] = all_p["train"]
        all_embeddings["val"] = all_embeddings["train"]


    # DFR on validation
    print("DFR")
    dfr_results = {}
    c = dfr_on_validation_tune(
        all_embeddings, all_y, all_g, num_retrains=5, seed=args.seed)
    dfr_results["best_hypers"] = c
    print("Hypers:", (c))

    test_min_lst = []
    test_mean_lst = []
    val_accs_lst = []
    for _ in tqdm.trange(5):
        test_accs, test_mean_acc, train_accs, val_accs = dfr_on_validation_eval(
            args, c, all_embeddings, all_y, all_g, group_weights=train_distribution, target_type="target", num_retrains=20)
        test_min_lst.append(np.nanmin(test_accs))
        test_mean_lst.append(test_mean_acc)
        val_accs_lst.append(val_accs)

    dfr_results["test_accs"] = test_accs
    dfr_results["train_accs"] = train_accs
    dfr_results["val_accs"] = np.mean(val_accs)
    dfr_results["test_worst_acc"] = np.nanmin(test_accs)
    dfr_results["test_mean_acc"] = test_mean_acc

    if len(test_accs) == 8 and "Urbancars" in args.dataset:
        gaps = {}
        gaps["bg_gap"] = (test_accs[2] + test_accs[5])/2 - test_mean_acc
        gaps["coObj_gap"] = (test_accs[1] + test_accs[6])/2 - test_mean_acc
        gaps["bg+coObj_gap"] = (test_accs[3] + test_accs[4])/2 - test_mean_acc
        dfr_results["test_gaps"] = gaps
    else:
        print("Warning: not all groups present, not reporting group results.")

    print(dfr_results)
    print()

    all_results = {}
    all_results["base_model_results"] = base_model_results
    all_results["dfr_val_results"] = dfr_results

    if args.predict_spurious:
        print("Predicting spurious attribute")
        all_y = all_p

        # DFR on validation
        print("DFR (spurious)")
        dfr_spurious_results = {}
        c = dfr_on_validation_tune(
            all_embeddings, all_y, all_g, seed=args.seed)
        dfr_spurious_results["best_hypers"] = c
        print("Hypers:", (c))
        test_accs, test_mean_acc, train_accs, val_accs = dfr_on_validation_eval(
            args, c, all_embeddings, all_y, all_g, group_weights=train_distribution, target_type="spurious")
        dfr_spurious_results["test_accs"] = test_accs
        dfr_spurious_results["train_accs"] = train_accs
        dfr_spurious_results["test_worst_acc"] = np.min(test_accs)
        dfr_spurious_results["test_mean_acc"] = test_mean_acc
        print(dfr_spurious_results)
        print()

        all_results["dfr_val_spurious_results"] = dfr_spurious_results
    
    print(all_results)


    command = " ".join(sys.argv)
    all_results["command"] = command
    all_results["model"] = args.model

    if args.ckpt_path:
        if os.path.exists(os.path.join(os.path.dirname(args.ckpt_path), 'args.json')):
            base_model_args_file = os.path.join(os.path.dirname(args.ckpt_path), 'args.json')
            with open(base_model_args_file) as fargs:
                base_model_args = json.load(fargs)
                all_results["base_args"] = base_model_args
        if args.save_best_epoch:
            if os.path.exists(os.path.join(os.path.dirname(args.ckpt_path), 'best_epoch_num.npy')):
                base_epoch_file = os.path.join(os.path.dirname(args.ckpt_path), 'best_epoch_num.npy')
                best_epoch_num = np.load(base_epoch_file)[0]
                all_results["base_model_best_epoch"] = best_epoch_num

    with open(args.result_path, 'wb') as f:
        pickle.dump(all_results, f)

    return test_mean_lst, test_min_lst, logger, all_results

def run(args):
    list_of_group_label_types = args.group_label_type.copy()
    for group_label_type in list_of_group_label_types:
        args.group_label_type = group_label_type

        utils.set_seed(args.seed)

        # put all prints into a log file
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        args.result_path = f"{os.path.dirname(args.result_path)}/{args.group_label_type}-dfr-seed{args.seed}.pkl"
        with open(os.path.join(os.path.dirname(args.result_path), f"logs-{args.group_label_type}-seed{args.seed}.txt"), 'w') as f:
            with redirect_stdout(f), redirect_stderr(f):
                test_mean_lst, test_min_lst, logger, all_results = main(args)

        print("Label type:", args.group_label_type)
        print(f"Mean test acc: {np.mean(test_mean_lst):.3f} \u00B1 {np.std(test_mean_lst) :.3f}")
        print(f"Worst group test acc: {np.mean(test_min_lst):.3f} \u00B1 {np.std(test_min_lst) :.3f}")

        # print("val accs:" all_results["dfr_val_results"]["val_accs"])
        print("mean val acc:", all_results["dfr_val_results"]["val_accs"])

        if all_results["dfr_val_results"].get("test_gaps") is not None:
            print("Gaps:", end=" ")
            for k, v in all_results["dfr_val_results"]["test_gaps"].items():
                print(f"{k}: {v:.5f}", end=", ")
        print("\n")


if __name__ == '__main__':
    args = get_args()
    run(args)