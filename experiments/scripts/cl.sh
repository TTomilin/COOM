# CD4
python3 run_cl.py --sequence CD4 --seed 1 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD4 --seed 2 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD4 --seed 3 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD4 --seed 1 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD4 --seed 2 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD4 --seed 3 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD4 --seed 1 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD4 --seed 2 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD4 --seed 3 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD4 --seed 1 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD4 --seed 2 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD4 --seed 3 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD4 --seed 1 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CD4 --seed 2 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CD4 --seed 3 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CD4 --seed 1
python3 run_cl.py --sequence CD4 --seed 2
python3 run_cl.py --sequence CD4 --seed 3
python3 run_cl.py --sequence CD4 --seed 1 --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5
python3 run_cl.py --sequence CD4 --seed 2 --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5
python3 run_cl.py --sequence CD4 --seed 3 --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5

# CD8
python3 run_cl.py --sequence CD8 --seed 1 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD8 --seed 2 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD8 --seed 3 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD8 --seed 1 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD8 --seed 2 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD8 --seed 3 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD8 --seed 1 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD8 --seed 2 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD8 --seed 3 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD8 --seed 1 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD8 --seed 2 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD8 --seed 3 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD8 --seed 1 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CD8 --seed 2 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CD8 --seed 3 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CD8 --seed 1
python3 run_cl.py --sequence CD8 --seed 2
python3 run_cl.py --sequence CD8 --seed 3

# CO4
python3 run_cl.py --sequence CO4 --seed 1 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO4 --seed 2 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO4 --seed 3 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO4 --seed 1 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO4 --seed 2 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO4 --seed 3 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO4 --seed 1 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO4 --seed 2 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO4 --seed 3 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO4 --seed 1 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO4 --seed 2 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO4 --seed 3 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO4 --seed 1 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CO4 --seed 2 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CO4 --seed 3 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CO4 --seed 1
python3 run_cl.py --sequence CO4 --seed 2
python3 run_cl.py --sequence CO4 --seed 3
python3 run_cl.py --sequence CO4 --seed 1 --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 2e5
python3 run_cl.py --sequence CO4 --seed 2 --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 2e5
python3 run_cl.py --sequence CO4 --seed 3 --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 2e5

# CO8
python3 run_cl.py --sequence CO8 --seed 1 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO8 --seed 2 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO8 --seed 3 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO8 --seed 1 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO8 --seed 2 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO8 --seed 3 --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO8 --seed 1 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO8 --seed 2 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO8 --seed 3 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO8 --seed 1 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO8 --seed 2 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO8 --seed 3 --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO8 --seed 1 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CO8 --seed 2 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CO8 --seed 3 --cl_method vcl --cl_reg_coef=1.0 --vcl_first_task_kl False
python3 run_cl.py --sequence CO8 --seed 1
python3 run_cl.py --sequence CO8 --seed 2
python3 run_cl.py --sequence CO8 --seed 3

# COC
python3 run_cl.py --sequence COC --sparse_rewards --seed 1 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence COC --sparse_rewards --seed 2 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence COC --sparse_rewards --seed 3 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05