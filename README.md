# JointDiff

***

## Introduction

***

## Environment

The packages used for this project are listed in **environment.yml**. The environment can be constructed with:
```
conda env create -f environment.yml
```

To activate the environment, run:
```
conda activate jointdiff
```

***

## Data Process 

Our processed training data (3 *.lmdb files) can be downloaded with this [link](https://drive.google.com/drive/folders/1DaITD4DOu7EJt6Me1lgffOprv39U2ltw?usp=drive_link). To train JointDiff with our data, please download the files and move them to the folder **data/**.

To train the model with you own data, please get the *.tsv file ready following the format of **data/cath_summary_all.tsv**, and then update the paths of the dataset in the configuration file (e.g. **configs/jointdiff-x_dim-128-64-4_step100_lr1.e-4_wd0._posiscale50.0.yml**).

***

## Training

To train JointDiff of JointDiff-x, go to the folder **src/** and run:
```
python train_jointdiff.py \
--config <path of the configuration file> \
--logdir <path to save the checkoints> \
--centralize <whether centralization; 0 or 1, 1 for True> \
--random_mask <whether do random masking; 0 or 1, 1 for True> \
--with_dist_loss <whether add the pairwise distance loss; 0 or 1, 1 for True> \
--with_clash <whether add the clash loss; 0 or 1, 1 for True> 
```

Example:
```
# Train a JointDiff model (lr = 1e-4, wd = 0.0, sw = 1 / posiscale = 0.02)
# with sample centralization, distance loss, clash loss; without random masking

mkdir ../Logs/

python train_jointdiff.py \
--config ../configs/jointdiff_dim-128-64-4_step100_lr1.e-4_wd0._posiscale50.0.yml \
--logdir ../Logs/ \
--centralize 1 \
--random_mask 0 \
--with_dist_loss 1 \
--with_clash 1
```

***

## Inference
