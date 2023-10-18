SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# CD4
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method agem --regularize_critic --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method ewc --cl_reg_coef=250
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python3 run_cl.py --sequence CD4 --seed [SEED]  # Fine-tuning
python3 run_cl.py --sequence CD4 --seed [SEED] --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5
python3 run_cl.py --sequence CD4 --seed [SEED] --cl_method clonex --exploration_kind 'best_return' --cl_reg_coef=100 --episodic_mem_per_task 10000 --episodic_batch_size 128

# CD8
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method agem --regularize_critic --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method ewc --cl_reg_coef=250
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python3 run_cl.py --sequence CD8 --seed [SEED]  # Fine-tuning
python3 run_cl.py --sequence CD8 --seed [SEED] --cl_method clonex --exploration_kind 'best_return' --cl_reg_coef=100 --episodic_mem_per_task 10000 --episodic_batch_size 128

# CO4
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method agem --regularize_critic --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method ewc --cl_reg_coef=250
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python3 run_cl.py --sequence CO4 --seed [SEED]  # Fine-tuning
python3 run_cl.py --sequence CO4 --seed [SEED] --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 2e5
python3 run_cl.py --sequence CO4 --seed [SEED] --cl_method clonex --exploration_kind 'best_return' --cl_reg_coef=100 --episodic_mem_per_task 10000 --episodic_batch_size 128

# CO8
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method mas --cl_reg_coef=10000
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method agem --regularize_critic --episodic_mem_per_task 10000 --episodic_batch_size 128
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method ewc --cl_reg_coef=250
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python3 run_cl.py --sequence CO8 --seed [SEED]  # Fine-tuning
python3 run_cl.py --sequence CO8 --seed [SEED] --cl_method clonex --exploration_kind 'best_return' --cl_reg_coef=100 --episodic_mem_per_task 10000 --episodic_batch_size 128

# CD16
python3 run_cl.py --sequence CD16 --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05

# CO16
python3 run_cl.py --sequence CO16 --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05

# COC
python3 run_cl.py --sequence COC --sparse_rewards --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05