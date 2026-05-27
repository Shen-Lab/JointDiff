# JointDiff

***

## Introduction

Computational design of functional proteins is of fundamental and applied interests.  Data-driven methods, especially generative deep learning, has seen a surge recently.  In this study, we aim at learning joint distribution of protein sequence and structure for their simultaneous co-design. To that end, we treat protein sequence and structure as three distinct modalities (amino-acid type plus positions and orientations of backbone residue frames) and learn three distinct diffusion processes (multinomial, Cartesian, and $SO(3)$ diffusions).  To bridge the three modalities, we introduce a graph attention encoder shared across modalities, whose inputs include all modalities and outputs are projected to predict individual modalities. 

Benchmark evaluations indicate that resulting JointDiff simultaneously generates protein sequence-structure pairs of better functional consistency compared to popular two-stage protein designers   Chroma (structure first) and ProteinGenerator (sequence first), while being more than $10$-times faster.  Meanwhile, they show room to improve in certain self- and cross-consistency. 

![architecture](/Architecture_JointDiff.png)
[paper](https://onlinelibrary.wiley.com/doi/10.1002/pro.70340)

***

## Environment

### Conda

The packages used for this project are listed in [environments/environment.yml](https://github.com/Shen-Lab/JointDiff/blob/main/environments/environment.yml). The environment can be constructed with:
```
conda env create -f environments/environment.yml
```

To activate the environment, run:
```
conda activate jointdiff
```

### PIP
For other environment manage tools, we also provide [environments/requirements.txt](https://github.com/Shen-Lab/JointDiff/blob/main/environments/requirements.txt):
```
pip install -r environments/requirements.txt
```

***

## Data Process 

Our processed training data (3 *.lmdb files) can be downloaded with this [link](https://zenodo.org/records/14517007). To train JointDiff with our data, please download the files and move them to the folder **data/**.

To train the model with you own data, please get the *.tsv file ready following the format of **data/cath_summary_all.tsv**, and then update the paths of the dataset in the configuration file (e.g. **configs/jointdiff-x_dim-128-64-4_step100_lr1.e-4_wd0._posiscale50.0.yml**).

***

## Inference

We restructured the project during the revision process to improve readability and facilitate future development. We also reduced model redundancy to improve efficiency. The latest scripts can be found in [src/](https://github.com/Shen-Lab/JointDiff/tree/main/src), which support all JointDiff-x models, confidence models and all retrained models (including retrained JointDiff). The released JointDiff models were developed using our original version (v0) based on [DiffAb](https://github.com/luost26/diffab), and the corresponding inference code is provided in [src_v0/](https://github.com/Shen-Lab/JointDiff/tree/main/src_v0). We are currently working on merging the two implementations into a unified framework.

### Checkpoints
We released our major checkpoints [here](), including:

* **monomer design**
  * **JointDiff**: JointDiff_model.pt
  * **JointDiff-x (MSE)**: JointDiff-x_MSE-dist-distogram.pt
  * **JointDiff-x (FAPE)**: JointDiff-x_FAPE-distmse-distogram.pt
* **motif scaffolding**
  * **JointDiff-x (MSE)**: JointDiff-x_rm_mse-dist-disto.pt
  * **JointDiff-x (FAPE)**: JointDiff-x_rm_fape-dist-disto.pt
  * **JointDiff-x (FAPE; finetuned for GFP)**: JointDiff-x_rm_GFP-range50-fape-weighted.pt
* **confidence net**
  * **Regression-based confidence net**: confidence.pt
  * **Binary-classification-based confidence net**: confidence_binary.pt

To apply our released models, please download the models and move them to the desired path (default path: checkpoints/).


### Monomer Design

Go to [src_v0/](https://github.com/Shen-Lab/JointDiff/tree/main/src_v0) for JointDiff, and go to [src/](https://github.com/Shen-Lab/JointDiff/tree/main/src/) for JointDiff-x or your retrained JointDiff. Then run: 

```
python infer_jointdiff.py \
--model_path <str; path of the checkpoint> \
--result_path <str; path to save the samples> \
--size_range <list of length_min, length_max, length_interval> \
--num <int; sampling amount for each length> \
--save_type <str; 'last' for saving the sample of t=0; 'all' for saving the whole reverse trajectory> 
```

You may refer to [example_jointdiff_unconditional_sampling.sh](https://github.com/Shen-Lab/JointDiff/blob/main/src_v0/illustation/example_jointdiff_unconditional_sampling.sh) and [example_jointdiff-x_unconditional_sampling.sh](https://github.com/Shen-Lab/JointDiff/blob/main/src/illustation/example_jointdiff-x_unconditional_sampling.sh), or the example below:
```
mkdir ../samples/

# Inference with our JointDiff-x;
# for each protein size in {100, 120, 140, 160, 180, 200}, get 5 samples;
# save the last sample (t=0) of the trajectory;
# the samples will be saved as ../samples/len<l>_<t>_<idx>.pdb, while l is the length,
# t is the diffusion step and idx is the sample index;
# e.g. len100_0_1.pdb refers to sample #1 of protein size 100 at t=0.

python infer_jointdiff.py \
--model_path ../checkpoints/JointDiff-x_model.pt \
--result_path ../samples/ \
--size_range [100, 200, 20] \
--num 5 \
--save_type 'last'
```

### Motif-scaffolding

For motif-scaffolding, please down load the PDB files containing the motifs and prepare a CSV file containing the motif information (e.g. for GFP, it should be '0,1QY3_GFP,"55-55,A58-71,24-24,A96-96,125-125,A222-222,7-7",227-227'). Then run:

```
python infer_motifscaffolding_jointdiff.py  \
--model_path <str; path of the checkpoint> \
--data_path <str; path of the CSV file indicating the motif information> \
--pdb_path <str; path of the folder containing the pdb files> \
--info_dict_path <str; processed loadable data; if not exists, the processed dictionary will be saved to this path> \
--result_path <str; directory to save the samples> \
--attempt <int; sampling amount for each task>
```

Example:
```
### target description
data_path="../data/motif-scaffolding_benchmark/benchmark.csv"

### processed data; if not found, it will be automatically processed and saved at this path
info_dict_path="../data/motif-scaffolding_benchmark/benchmark_processed.pkl"

pdb_path="../data/motif-scaffolding_benchmark/pdbs_processed/"
ckpt_path="../checkpoints/JointDiff-x_rm_fape-dist-disto.pt"
out_path="../samples/motifscaffolding_jointdiff-x_samelength/"
mkdir -p -v ${out_path}

sample_length=0  # if 0, use the same protein size as the template structure; otherwise the protein size will be sampled following the target description

python infer_motifscaffolding_jointdiff.py \
--data_path ${data_path} \
--info_dict_path ${info_dict_path} \
--pdb_path ${pdb_path}  \
--model_path ${ckpt_path} \
--result_path ${out_path} \
--sample_length ${sample_length} \
--save_type 'last'
--attempt 10  # generate 10 samples for each motif
```

### Confidence Inference

To estimate the confidence value with our confidence net, run:

```
python infer_confidence.py \
--pdb_dir <str; path of the folder containing the pdb files>
--ckpt_path <str; path of the model checkpoints> \
--result_path <str; path to save the results>
```

***

## Training

### JointDiff & JointDiff-x
To train JointDiff of JointDiff-x, go to [src/](https://github.com/Shen-Lab/JointDiff/tree/main/src/) and run:

```
# The texts in the angled bracket refer to the indication rather than true values; users need to define them to run the scripts.

python train_jointdiff.py \
--config <str; path of the configuration file> \
--logdir <str; path to save the checkoints> \
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

### Confidence Net
To train the confidence net, run:

```
python train_confidence_net.py \
--processed_dir <str; path of the processed data> \
--logdir <str; path to save the checkpoints> \
--centralize <0 or 1; 1 to centralize the input structure> \
--binary <0 or 1; 1 for binary classication and 0 for regression> \
--label_norm <0 or 1; 1 to normalize the labels, only valid for regression task> \
--balance <0 or 1; 1 to balance the training process with weighted loss> \
--max_epoch <int; maximum training epochs> 
```

***

## Evaluation
To evaluate model performance with our published metrics, go to [src/evaluation](https://github.com/Shen-Lab/JointDiff/tree/main/src/evaluation) and follow the instructions below:

### Biological features (torsional angles and clashes)
```
python pdb_feature_cal.py \
--in_path <str; path of the pdb file> \
--out_path <str; path to save the output pickle dictionary>
```

### Amino acid repeating rate
```
python repeating_rate.py --in_path <str; fasta file> 
```

### Sequence-centered consistency & Foldability
For sequence and structure predictions, please refer to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and [ESMFold](https://github.com/facebookresearch/esm) to get the corresponding intermediate files.
```
python consistency-seq_cal.py \
--seq_path <str; fasta file of MPNN predictions> \
--gt_path <str; fasta file of the designed sequences> \
--out_path <str; path to save the output pickle dictionary>
```
* For sequence-centered consistency, MPNN predictions are based on designed structures.
* For foldability, MPNN predictions are based on ESMFold-predicted structures of the designed sequences.

### Structure-centered consistency & Designability
```
python consistency-struc_cal.py \
--ref_path <str; directory of designed structures/pdbs> \
--pred_path <str; directory of the ESMFold predictions> \
--out_path <str; path to save the output pickle dictionary>
```
* For structure-centered consistency, ESMFold predictions are based on designed sequences.
* For designability, ESMFold predictions are based on MPNN-infered sequences of the designed structures.

### Functional consistency
For the sequence and structure embeddings, please follow [ProTrek](https://github.com/westlake-repl/ProTrek) and save the results as dictionaries with key to be sample names and value to be embedding tensors.
```
python protrek_similarity.py \
--go_emb ../data/mf_go_all_emb.pkl  # dictionary containing GO terms embeddings \
--seq_emb  <str; path of the dictionary containing sequence embeddings> \
--struc_emb  <str; path of the dictionary containing structure embeddings> \
--out_path <str; path to save the output pickle dictionary>
```
