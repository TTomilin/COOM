SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEQUENCE = [CD4, CD8, CO4, CO8, CD16, CO16, COC]

python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method mas --cl_reg_coef=10000
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method agem --regularize_critic --episodic_mem_per_task 10000 --episodic_batch_size 128
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method ewc --cl_reg_coef=250
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python run_cl.py --sequence [SEQUENCE] --seed [SEED]  # Fine-tuning
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5
python run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method clonex --exploration_kind 'best_return' --cl_reg_coef=100 --episodic_mem_per_task 10000 --episodic_batch_size 128

# COC
python run_cl.py --sequence COC --sparse_rewards --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05

# Network plasticity
python CL/run_continual.py --sequence CO8 --seed [SEED] --repeat_sequence 10 --no_test --steps_per_env 100000

# Method Variations
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation conv
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation shift
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation noise
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --buffer_type prioritized
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --use_lstm
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --regularize_critic