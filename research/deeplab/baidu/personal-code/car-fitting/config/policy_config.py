prefix='./output/policy-net'

gpu_id = 1
gpu_flag = True

batch_size = 32
grad_dir_num = 8
n_epochs = 1000
epoch = 1000
discount = 0.99
max_path_length = 1000
eval_samples = 10000

actor_updater = "nadam"
actor_lr = 1e-5
actor_wd = 1e-4
soft_target_tau = 1e-3
n_updates_per_sample = 1
include_horizon_terminal = False
seed = 12345
scale = 0.5 # scale for image size
init_scale = 0.005 # scale for parameter init

is_crop = True
is_rel = True
height = 256
width = 256

GAMMA = 0.99
pose_dim = 6

EPISODES = 1000000
TEST = 100
# reward_name = "mask_err"
reward_name = "mcmc"

# for debugging
memory_size = 10000
memory_start_size = 64
mcmc_sample_num = 200
timestep_limit = 30
eval_iter = 800

