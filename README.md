# Diffusion Path Samplers via Sequential Monte Carlo

This repo is an implementation of our paper: [_Diffusion Path Samplers via Sequential Monte Carlo_](https://neurips.cc/virtual/2026/loc/sydney/poster/152139) (NeurIPS 2026).

In brief, this work expands on the existing literature surrounding training-free diffusion-based samplers through: (i) an SMC backbone for posterior sampling, (ii) control-variate score identities, and (iii) different instantiations of the diffusion path and their principled design. In this repo, we have in place an interface for running the different samplers we propose and testing them on various benchmark target distributions. 

## Directory Structure
```
dpsmc/
├── config                                # Config files
│   ├── experiment
│   │   └── sample.yaml
│   ├── sampler
│   │   └── ...
│   ├── target
│   │   └── ...
│   └── config.yaml
├── experiments                           # Logic for experiments
│   └── experiments.py
├── samplers                              # Implemented samplers
│   ├── dpsmc_dald_ll.py
│   ├── dpsmc_dald.py
│   ├── dpsmc_ou_rdsmc.py
│   ├── dpsmc_ou.py
│   ├── dpsmc_si.py
│   └── sampler.py
├── targets                               # Target distributions
│   ├── datasets
│   │   ├── ionosphere.pkl
│   │   └── sonar.pkl
│   └── ...
├── utils                                 # Some helper functions
│   ├── distributions.py
│   ├── path.py
│   ├── registry.py
│   └── score_identities.py
├── main.py                               # Main-entry point
├── README.md
└── requirements.txt                      # Packages required
```


## Installation

Install the necessary Python packages specified in our `requirements.txt` file. You can do this with the following command:
```bash
pip install -r requirements.txt
```
We used python 3.10 for maintaining this code, and it's recommended you use the same version. If you encounter problems with paths, consider setting your python path environment variable to the outermost directory of our repository. Upon `cd`-ing into it, you can run the following command:
```bash
export PYTHONPATH=$PWD
```


## Usage

The project uses hydra to flexibly handle config files. Below is a general recipe for how to run an instance of our sampler to sample from a benchmark target:
```bash
python3 main.py experiment=sample \
    sampler=[sampler] target=[target] device=[device] seed=[seed]
```
We have the following options for samplers and targets:

|Parameter|Available Options|
|:-:|:-|
|`sampler`|`dpsmc_dald`, `dpsmc_dald_ll`, `dpsmc_ou`, `dpsmc_si`, `dpsmc_ou_rdsmc`|
|`target`|`gmm40_dim_2`, `gmm40_dim_50`, `rings`, `funnel`, `ionosphere`, `sonar`|

By default, we produce $N_X=4096$ samples, each maintaining $N_Y=128$ auxiliary variables, with $K=1024$ discretisation steps. This can be modified by appending the updated setting in the command, e.g. `sampler.n_steps=2048`. Check the config files to see what other parameters you can change. After sampling, it runs the evaluation metrics associated with the target and saves the samples in the `outputs` folder.

Below, we describe how to run each version of DPSMC.


### DPSMC (DALD)

Each target has already been specified with the $\xi$ hyperparameter (for computing $\epsilon$ in DALD) in their configs, so you can run DALD without specifying any hyperparameters to run our benchmarks, e.g.
```bash
python3 main.py experiment=sample \
    sampler=dpsmc_dald target=ionosphere device="cuda:0" seed=0
```

### DPSMC (DALD) with Langevin-within-Langevin initialisation

The $\xi$ hyperparameter is set automatically as above, but you need to specify a value for the initial schedule/time $\lambda_0$ (see Table 6 in Appendix G1), e.g.
```bash
python3 main.py experiment=sample \
    sampler=dpsmc_dald_ll target=rings sampler.lambda_0=0.667 device="cuda:0" seed=0
```

### DPSMC (OU)

You need to specify a value for the initial schedule/time $\lambda_0$ (see Table 5 in Appendix G1), e.g.
```bash
python3 main.py experiment=sample \
    sampler=dpsmc_ou target=gmm40_dim_2 sampler.lambda_0=0.5 device="cuda:0" seed=0
```

### DPSMC (SI)

You can run SI without specifying any hyperparameters to run our benchmarks, e.g.
```bash
python3 main.py experiment=sample \
    sampler=dpsmc_si target=funnel device="cuda:0" seed=0
```

### DPSMC (OU) w/ RDSMC weights

In this setup, we also produce normalising constant estimates. By default, we set no. of discretisation steps $K=124$, no. of AIS steps to $32$, and no. of MALA steps to be $8$. The initial schedule/time $\lambda_0=4/6$ is set as per Table 8 in Appendix G3. To run experiments for Rings, do
```bash
python3 main.py experiment=sample \
    sampler=dpsmc_ou_rdsmc target=rings sampler.lambda_0=0.667 device="cuda:0" seed=0
```
For Funnel, we increase the no. of AIS steps to $160$ and no. of MALA steps to be $40$. To run experiments for Funnel, do
```bash
python3 main.py experiment=sample \
    sampler=dpsmc_ou_rdsmc target=funnel sampler.lambda_0=0.667 \
    sampler.n_ais_steps=160 sampler.n_mala_steps=40 device="cuda:0" seed=0
```

## Remarks

In `targets/gmm.py` lines 80-95, we specify a function for chunking large tensors into smaller batches before being evaluated of its log probability (and later the gradient). If your GPU allows for it, you can comment out our batching operation to reap the full benefits of our sampler's batch parallelism.

If you would like to modify the visible GPU devices to avoid extra memory sometimes being allocated to GPU 0 even if it is unused, you can set, e.g. `visible_devices=\'5,6,7\'`.

Notably, even with fixed seeds, results may vary slightly from those in the paper due to hardware, library versions (e.g. PyTorch/CUDA), and `torch.compile`, since small numerical differences can compound through resampling and step-size adaptation.


## Citation

If you would like to reference our work, here is a BibTeX citation you can use.

```bibtex
@inproceedings{young2026diffusion,
  title     = {Diffusion Path Samplers via Sequential {M}onte {C}arlo},
  author    = {Young, James Matthew and {Cordero-Encinar}, Paula and Reich, Sebastian and Duncan, Andrew B. and Akyildiz, O. Deniz},
  booktitle = {Proceedings of the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026}
}
```