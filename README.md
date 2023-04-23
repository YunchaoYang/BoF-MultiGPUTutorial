<img src="./img/BoF-workshop.png" alt="drawing" width="400"/>

# Fundamentals of Accelerated Neural Network Training with Multi-GPUs on HiPerGator-AI

- Yunchao Yang
- AI support team
- UF Research Computing

[slides](FundamentalOfMultiGPUTraining.pdf)

[GitHub repo](https://github.com/YunchaoYang/BoF-MultiGPUTutorial)

Code is adapted for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

## How to use the Files
Step 0: navigate to `ood.rc.ufl.edu` and request a open ondemand console terminal with 1 node and 2 GPUs:  
  - Cluster partition = "hpg-ai"
  - Generic Resource Request = "gpu:2"

Step 1: single GPU code
* [single_gpu.py](single_gpu.py): Non-distributed training script on a single GPU
* [How to run]: `python single_gpu.py` 

Step 2.1: single node parallel with mp.spawn utility  
* [multigpu.py](multigpu.py): DDP on a single node, with `mp.spawn`  
* [run_multigpu.sh](run_multigpu.sh): runner

Step 2.2: single node paralel with torchrun utility
* [multigpu_torchrun.py](multigpu.py): DDP on a single node using `torchrun`
* [run_multigpu_torchrun.sh](run_multigpu_torchrun.sh): runner

Step 3 multiNode parallel with torchrun utility
* [multinode.py](multinode.py): DDP on multiple nodes using Torchrun (and optionally Slurm)

Step 4. Run SLURM jobs on HPG
    * [slurm/multigpu_torchrun.py](slurm/multigpu_torchrun.py): training script for multiGPU
    * [slurm/datautils.py](slurm/datautils.py): Dataset helper function 
    * [slurm/launch_ddp_2N4G.sh](slurm/launch_ddp_2N4G.sh): Sample slurm script to launch a trining script using torchrun on 2Nodes with 2GPUs no each node.
    * [slurm/launch_ddp_4N4G.sh](slurm/launch_ddp_4N4G.sh): Sample slurm script to launch a trining script using torchrun on 4Nodes with 1GPUs no each node.

## learn more about the code walkthrough
please follow the [Distributed Data Parallel in PyTorch Tutorial Series](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj) by Pytorch