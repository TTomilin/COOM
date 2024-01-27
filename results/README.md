## Result Module
This module provides scripts for downloading and plotting the results of the experiments in our paper.
The results are stored in [Weights | Biases](https://wandb.ai/) and can be downloaded using the scripts in the [download](download) folder.
The plotting scripts are located in the [plotting](plotting) folder.
Calculating the core results in the paper can be done using the [cl_metrics.py](tables/cl_metrics.py) script.

## Installation
To install the results module, run the following command:
```bash 
$ pip install COOM[results]
```

### Running experiments
For running the experiments in our paper please follow the instructions in the continual learning (CL) module [README.md](../CL/README.md).

### Downloading results
We recommend using [Weights | Biases](https://wandb.ai/) to log your experiments. 
Having done so, you can use the following scripts to download the results:
1. Continual Learning Data - [cl_data.py](download/cl_data.py)  
`python cl_data.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE>`
2. Single Run Data - [single_data.py](download/single_data.py)  
`python single_data.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE>`
3. Single Run Data of hard (COC) environments - [single_data.py](download/single_data_hard.py)  
`python single_data_hard.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE>`
4. Action Distribution Data - [action_data.py](download/action_data.py)  
- For training data run  
`python single_data_hard.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE>`  
- For evaluation data run  
`python single_data_hard.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE> --test_envs <TEST_ENVS>`  
- For example, getting the data of all tasks of the CO8 sequence can be done by running
`python single_data_hard.py --project <YOUR_WANDB_PROJECT> --sequence CO8 --test_envs 0 1 2 3 4 5 6 7`
5. Runtime Data - [runtime_data.py](download/runtime_data.py)  
- For memory usage data run
`python runtime_data.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE> --metric system.proc.memory.rssMB`  
- For walltime data run  
`python runtime_data.py --project <YOUR_WANDB_PROJECT> --sequence <SEQUENCE> --metric walltime`  

### Plotting figures

Figures from the paper can be drawn using the [plotting scripts](https://github.com/TTomilin/COOM/tree/main/experiments/results).  
```
python plot_results_envs.py --sequence <SEQUENCE> --metric <METRIC>
python plot_results_methods.py --sequence <SEQUENCE> --metric <METRIC>
```

### Calculating metrics
The main results displayed in our paper can be calculated using [cl_metrics.py](tables/cl_metrics.py).  
```
python cl_metrics.py --sequences CD4 CO4 CD8 CO8 COC --methods packnet mas agem l2 ewc vcl fine_tuning clonex perfect_memory
```
