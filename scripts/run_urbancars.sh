# path to data dir
DATA_DIR= ""
DATASET_NAME=BothUrbancarsDataset #or use BgUrbancarsDataset, CoOccurObjUrbancarsDataset for the single shortcut cases
DATASET_TRANSFORM=UrbancarsTransform

# path to model checkpoint dir. Where model checkpoints will be stored
MODEL_TRAIN_DIR= ""
CHECKPOINT_PATH=$MODEL_TRAIN_DIR/best_weighted_acc_checkpoint.pt
MODEL=imagenet_resnet50_pretrained

# Where to store exmap results 
RESULTS_DIR= ""


# train model
python3 src/run_erm.py --output_dir=$MODEL_TRAIN_DIR \
	--num_epochs=100 --eval_freq=1 --save_freq=100 \
	--weight_decay=1e-4 --batch_size=32 --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=$DATA_DIR \
    --data_transform=$DATASET_TRANSFORM \
    --dataset=$DATASET_NAME --model=$MODEL  


# run exmap using spectral clustering and the eigengap heuristic (--auto)
python src/run_exmap.py --base_dir $DATA_DIR --dataset $DATASET_NAME\
    --model $MODEL --ckpt_path $CHECKPOINT_PATH \
    --downsize 56 --clustering_type spectral --plot_type umap\
    --results_dir $RESULTS_DIR --batch_size 32 \
    --auto


# run dfr (default), G-Exmap+dfr (global_cluster), L-Exmap+dfr (local_cluster
python src/run_dfr.py --data_dir $DATA_DIR --dataset $DATASET_NAME \
    --output_dir $RESULTS_DIR --model $MODEL --ckpt_path $CHECKPOINT_PATH \
    --group_label_type default global_cluster local_cluster