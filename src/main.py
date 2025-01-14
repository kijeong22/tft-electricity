#%%
import torch
import numpy as np
import os
import argparse
import json
import numpy as np
from datetime import datetime
import random
import wandb
from model import TFT
from datetime import datetime
import uuid

#%%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#%%
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="TFT")
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument("--mode", 
                        type=str, 
                        default='train', 
                        help="training or evaluation")
    parser.add_argument("--key", 
                        type=str, 
                        default=None, 
                        help="wandb key")
    parser.add_argument("--d", 
                        type=int,
                        default=32, 
                        help="d_model")
    parser.add_argument("--nh", 
                        type=int, 
                        default=4, 
                        help="num_heads")
    parser.add_argument("--dr", 
                        type=float, 
                        default=0.1, 
                        help="dropout")
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-3, 
                        help="learning_rate")
    parser.add_argument("--q", 
                        type=list, 
                        default=[0.1,0.5,0.9], 
                        help="quantiles")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('./config/config.json', 'r') as f:
        config = json.load(f)

    seed = args.seed
    mode = args.mode
    key = args.key
    
    config['model']['d_model'] = args.d
    config['model']['num_heads'] = args.nh
    config['model']['dropout'] = args.dr
    config['model']['quantiles'] = args.q
    config['model']['device'] = device

    config['train']['learning_rate'] = args.lr

    seed_everything(seed)

    project_name = 'TFT'
    current_time = datetime.now()
    run_name = current_time.strftime("%y%m%d-%H%M")# + "-" + str(uuid.uuid4())[:3]

    if key is not None:

        wandb.login(key=key)
        wandb.init(project=project_name, entity='99rlwjd', name=run_name)
        wandb.config.update(args)

    config['run_name'] = run_name

    # Model Initialization
    model = TFT(config=config)

    if mode == 'train':
        model.fit()

    # elif mode == 'eval':
    #     model.evaluate()
    
    if key is not None:
        wandb.finish()

    # post_training_cleanup(
    #     model_directory=config['train']['save_path'],
    #     run_name=run_name
    # )