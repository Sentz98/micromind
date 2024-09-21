"""
Configuration file image classification with PhiNet.

Authors:
    - Francesco Paissan, 2023
"""

# Model configuration
experiment_name = "phinet_375_cifar100_float"
model = "phinet"
input_shape = (3, 32, 32)
alpha = 3
num_layers = 7
beta = 1
t_zero = 5
divisor = 8
downsampling_layers = [5, 7]
return_layers = None

# Logging
log_level = "INFO"
wandb_log = True
wandb_name = experiment_name
wandb_id = 0000
wandb_resume = True

ckpt_pretrained = ""

# Basic training loop
epochs = 20

# Basic data
data_dir = "data/cifar100/"
dataset = "torch/cifar100"
batch_size = 256
dataset_download = True
calib_percentage = 30

# Dataloading config
num_workers = 4
pin_memory = True
persistent_workers = True

# Loss function
bce_loss = False
bce_target_thresh = None


# Quantization config
quantize = False
quantizer = "QAT"  #QAT, or DIFFQ

q_min_size = 0.0001  # minimum param size in MB to be quantized>
q_bits = 2  # number of bits used for uniform quantization
q_penalty = 5  # model weight penalty for DiffQ
q_group_size = 4  # group size for DiffQ
q_min_bits = 2  # minimal number of bits for DiffQ
q_init_bits = 8  # initial number of bits for DiffQ
q_max_bits = 15  # max number of bits for DiffQ
q_exclude = []  # exclude patterns, e.g. bias
q_qat = True  # quantization aware training to be used with uniform qunatization
q_lr = 1e-3  # learning rate for the bits parameters
q_adam = True  # use a separate optimizer for the bits parameters
q_lsq = False  # use LSQ


# Data augmentation config
aa = "rand-m8-inc1-mstd101"
aug_repeats = 0
aug_splits = 0
class_map = ""
color_jitter = 0.4
cutmix = 0.0
cutmix_minmax = None
drop = 0.0
drop_block = None
drop_connect = None
drop_path = 0.1
epoch_repeats = 0.0
hflip = 0.5
img_size = None
in_chans = None
initial_checkpoint = ""
interpolation = "bilinear"
jsd_loss = False
layer_decay = 0.65
local_rank = 0
log_interval = 50
log_wandb = True
lr = 0.001
lr_base = 0.1
lr_base_scale = ""
lr_base_size = 256
lr_cycle_decay = 0.5
lr_cycle_limit = 1
lr_cycle_mul = 1.0
lr_k_decay = 1.0
lr_noise = None
lr_noise_pct = 0.67
lr_noise_std = 1.0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mixup = 0.0
mixup_mode = "batch"
mixup_off_epoch = 0
mixup_prob = 1.0
mixup_switch_prob = 0.5
no_aug = False
num_classes = 100
ratio = [0.75, 1.3333333333333333]
recount = 1
recovery_interval = 0
remode = "pixel"
reprob = 0.3
scale = [0.08, 1.0]
smoothing = 0.1
train_interpolation = "bilinear"
train_split = "train"
use_multi_epochs_loader = False
val_split = "validation"
vflip = 0.0
