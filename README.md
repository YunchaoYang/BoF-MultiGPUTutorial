<img src="BoF-workshop.png" alt="drawing" width="400"/>

# Fundamentals of Accelerated Neural Network Training with Multi-GPUs on HiPerGator-AI

- AI support team
- UF Research Computing

[slides](FundamentalOfMultiGPUTraining.pdf)

[GitHub repo](https://github.com/YunchaoYang/BoF-MultiGPUTutorial)

Code is adapted for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

## Files

Step 1
* [single_gpu.py](single_gpu.py): Non-distributed training script on a single GPU

Step 2.1
* [multigpu.py](multigpu.py): DDP on a single node

Step 2.2
* [multigpu_torchrun.py](multigpu.py): DDP on a single node using Torchrun

Step 3
* [multinode.py](multigpu.py): DDP on multiple nodes using Torchrun (and optionally Slurm)


Step 4. Run SLURM jobs on HPG
    * [slurm/multigpu_torchrun.py](slurm/multigpu_torchrun.py): training script for multiGPU
    * [slurm/datautils.py](slurm/datautils.py): Dataset helper function 
    * [slurm/launch_ddp_2N4G.sh](slurm/launch_ddp_2N4G.sh): Sample slurm script to launch a trining script using torchrun on 2Nodes with 2GPUs no each node.
    * [slurm/launch_ddp_4N4G.sh](slurm/launch_ddp_4N4G.sh): Sample slurm script to launch a trining script using torchrun on 4Nodes with 1GPUs no each node.



## learn more 
please follow the [Distributed Data Parallel in PyTorch Tutorial Series](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj) by Pytorch