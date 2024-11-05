# JointDiff

***

## Introduction

Computational design of functional proteins is of fundamental and applied interests.  Data-driven methods, especially generative deep learning, has seen a surge recently.  In this study, we aim at learning joint distribution of protein sequence and structure for their simultaneous co-design. To that end, we treat protein sequence and structure as three distinct modalities (amino-acid type plus positions and orientations of backbone residue frames) and learn three distinct diffusion processes (multinomial, Cartesian, and $SO(3)$ diffusions).  To bridge the three modalities, we introduce a graph attention encoder shared across modalities, whose inputs include all modalities and outputs are projected to predict individual modalities. 

Benchmark evaluations indicate that resulting JointDiff simultaneously generates protein sequence-structure pairs of better functional consistency compared to popular two-stage protein designers   Chroma (structure first) and ProteinGenerator (sequence first), while being more than $10$-times faster.  Meanwhile, they show room to improve in certain self- and cross-consistency. 

![architecture](/Architecture_JointDiff.png)

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
--centralize <whether to do centralization; 0 or 1, 1 for True> \
--random_mask <whether to do random masking; 0 or 1, 1 for True> \
--with_dist_loss <whether to add the pairwise distance loss; 0 or 1, 1 for True> \
--with_clash <whether to add the clash loss; 0 or 1, 1 for True> 
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
Our pretrained models (two *.pt files for JointDiff and JoinDiff-x) can be downloaded with this [link](https://drive.google.com/drive/folders/1wVBigdhMDL3FTX_u1--g1a4gkYAFjiG1?usp=drive_link). To do the inference sampling, go to the folder **src/** and run:
```
python infer_jointdiff.py \
--model_path <path of the checkpoint> \
--result_path <path to save the samples> \
--size_range <list of length_min, length_max, length_interval> \
--num <sampling amount for each length> \
--save_type <'last' for saving the sample of t=0; 'all' for saving the whole reverse trajectory> 
```

Example:
```
mkdir ../checkpoints/
# then download the models and save them in the folder ../checkpoints/
mkdir ../samples/

# Inference with our JointDiff-x;
# for each protein size in {100, 120, 140, 160, 180, 200}, get 5 samples;
# save the while trajectory, i.e. (T+1) pdb files for each sample;
# the samples will be saved as ../samples/len<l>_<t>_<idx>.pdb, while l is the length,
# t is the diffusion step and idx is the sample index;
# e.g. len100_0_1.pdb refers to sample #1 of protein size 100 at t=0.

python infer_jointdiff.py \
--model_path ../checkpoints/JointDiff-x_model.pt \
--result_path ../samples/ \
--size_range [100, 200, 20] \
--num 5 \
--save_type 'all'

```
