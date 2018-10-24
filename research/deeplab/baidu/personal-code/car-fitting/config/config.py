dataset='kitti'
prefix='./output/ddpg-car-fit'

gpu_ids = '0'
gpu_id = 3
gpu_flag = True
batch_size = 8
n_epochs = 1000
epoch_length = 1000
memory_size = 10000

discount = 0.99
max_path_length = 1000
eval_samples = 10000

critic_updater = "adam"
critic_lr = 1e-4
weight_decay = 5e-3

actor_updater = "adam"
actor_lr = 1e-4
soft_target_tau = 1e-3
n_updates_per_sample = 1
include_horizon_terminal = False
seed = 12345
scale = 0.5 # scale for image size
init_scale = 0.01 # scale for parameter init

GAMMA = 0.99
pose_dim = 6
update_pose = False

EPISODES = 1000000
TEST = 100

# for debugging data, change to larger mount when release
memory_start_size = 200
timestep_limit = 100
eval_iter = 200

is_crop = False
