prefix='./output/policy-net-apollo'

gpu_id = 1
gpu_flag = True

batch_size = 16
batch_process = 8
grad_dir_num = 8
n_epochs = 500
epoch = 500
discount = 0.99
max_path_length = 1000
eval_samples = 10000

network = 'deamon'
actor_updater = "nadam"
actor_lr = 5e-4
actor_wd = 1e-4
soft_target_tau = 1e-3
n_updates_per_sample = 1
include_horizon_terminal = False
seed = 12345
scale = 1.0 # scale for image size
init_scale = 0.005 # scale for parameter init

# for cropping
is_crop = True
is_rel = True
height = 256
width = 256

GAMMA = 0.99
pose_dim = 6

EPISODES = 1000000
TEST = 100

# for debugging
reward_name = 'mcmc'
memory_size = 5000
memory_start_size = 32
mcmc_sample_num = 100
timestep_limit = 30
eval_iter = 800

