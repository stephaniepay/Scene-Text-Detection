from easydict import EasyDict
import torch

config = EasyDict()

# dataloader jobs number
config.num_workers = 8


# batch_size
config.batch_size = 4

# training epoch number
config.max_epoch = 201

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True
# config.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

config.n_disk = 15

config.output_dir = 'output'

config.input_size = 512

# max polygon per image
config.max_annotation = 200

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.6

# demo tcl threshold
config.tcl_thresh = 0.4

# expand ratio in post processing
config.post_process_expand = 0.3

# merge joined text instance when predicting
config.post_process_merge = False

config.dataset = "total-text" 

config.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # added this line
config.vis_dir = "./vis/"
# config.exp_name = "vgg123456"
config.save_dir = './save/'
config.data_custom=False
config.data_root="./data/total-text"



def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
        
    # config.device = torch.device('cuda') if config.cuda else torch.device('cpu')
    
    #vgg
    if 'cuda' in config:  # Check if 'cuda' is a key in the `config` dict
        config['device'] = torch.device('cuda') if config['cuda'] else torch.device('cpu')  # Use the bracket notation
    else:
        config['device'] = torch.device('cpu')  # Fallback to 'cpu' if 'cuda' is not a key in the `config` dict
        
    if 'dataset' in config:
        print(config['dataset'])
    else:
        print("Dataset key is not in cfg.")
    ##############

def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
