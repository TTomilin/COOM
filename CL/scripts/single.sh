SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Default environments of each scenario
python run_single.py --seed [SEED] --scenario pitfall
python run_single.py --seed [SEED] --scenario arms_dealer
python run_single.py --seed [SEED] --scenario hide_and_seek
python run_single.py --seed [SEED] --scenario floor_is_lava
python run_single.py --seed [SEED] --scenario chainsaw
python run_single.py --seed [SEED] --scenario raise_the_roof
python run_single.py --seed [SEED] --scenario run_and_gun
python run_single.py --seed [SEED] --scenario health_gathering

# Hard environments of each scenario
python run_single.py --seed [SEED] --envs hard --scenario pitfall
python run_single.py --seed [SEED] --envs hard --scenario arms_dealer
python run_single.py --seed [SEED] --envs hard --scenario hide_and_seek
python run_single.py --seed [SEED] --envs hard --scenario floor_is_lava
python run_single.py --seed [SEED] --envs hard --scenario chainsaw
python run_single.py --seed [SEED] --envs hard --scenario raise_the_roof
python run_single.py --seed [SEED] --envs hard --scenario run_and_gun
python run_single.py --seed [SEED] --envs hard --scenario health_gathering

# Run and Gun variations
python run_single.py --seed [SEED] --scenario run_and_gun --envs obstacles
python run_single.py --seed [SEED] --scenario run_and_gun --envs green
python run_single.py --seed [SEED] --scenario run_and_gun --envs resized
python run_single.py --seed [SEED] --scenario run_and_gun --envs monsters
python run_single.py --seed [SEED] --scenario run_and_gun --envs default
python run_single.py --seed [SEED] --scenario run_and_gun --envs red
python run_single.py --seed [SEED] --scenario run_and_gun --envs blue
python run_single.py --seed [SEED] --scenario run_and_gun --envs shadows
