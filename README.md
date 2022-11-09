![Bird-of-Feather Workshop](BoF-workshop.png)

# Fundamentals of Accelerated Neural Network Training with Multi-GPUs on HiPerGator-AI

AI support team
UF Research Computing

[slides]FundamentalOfMultiGPUTraining.pdf]

[GitHub repo](https://github.com/YunchaoYang/BoF-MultiGPUTutorial)

# code is adapted from pytorch distributed-pytorch
code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.


## Files
* [single_gpu.py](single_gpu.py): Non-distributed training script on a single GPU

* [multigpu.py](multigpu.py): DDP on a single node

* [multigpu_torchrun.py](multigpu.py): DDP on a single node using Torchrun

* [multinode.py](multigpu.py): DDP on multiple nodes using Torchrun (and optionally Slurm)
    * [slurm/setup_pcluster_slurm.md](slurm/setup_pcluster_slurm.md): instructions to set up an AWS cluster
    * [slurm/config.yaml.template](slurm/config.yaml.template): configuration to set up an AWS cluster
    * [slurm/sbatch_run.sh](slurm/sbatch_run.sh): slurm script to launch the training job





## learn more 
please follow the [Distributed Data Parallel in PyTorch Tutorial Series](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj) by Pytorch