import argparse
import yaml
import torch
from scripts.MGN_train import run_training
from scripts.MGN_evaluate import run_evaluation
from tools.MGN_rollout import run_rollout

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Arguments
    parser = argparse.ArgumentParser(description="MeshGraphNets Simulator")
    parser.add_argument("--config", default='config/MGN_config.yaml', help="Path to config file")
    # Make mode optional in CLI
    parser.add_argument("--mode", default=None, choices=['train', 'eval', 'rollout'], help="Override execution mode")
    args = parser.parse_args()
    
    # 2. Load Config
    config = load_config(args.config)
    
    # PRIORITY: CLI > Config
    if args.mode is not None:
        current_mode = args.mode
    else:
        current_mode = config['mode'] 

    device = torch.device(config['train'].get('device', 'cpu'))
    
    print(f"Running in {current_mode.upper()} mode on {device}")

    # 3. Dispatch
    if current_mode == 'train':
        run_training(config, device)
    elif current_mode == 'eval':
        avg_loss = run_evaluation(config, device)
        print(f"Final Test Set MSE Loss: {avg_loss:.6f}")
    elif current_mode == 'rollout':
        run_rollout(config, device)

if __name__ == "__main__":
    main()