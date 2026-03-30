import yaml
import argparse
import multiprocessing as mp


def main():
    parser = argparse.ArgumentParser(description="PPO Crush Structure Optimizer")
    parser.add_argument('--config', type=str, default='config/PPG_config.yaml')
    parser.add_argument('--mode', type=str, default=None, help='Override config mode: train / eval')
    parser.add_argument('--trials', type=int, default=16, help='Number of eval trials')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mode = args.mode or config.get('mode', 'train')

    if mode == 'train':
        from scripts.PPG_train import run_training
        run_training(config)

    elif mode == 'eval':
        from scripts.PPG_evaluate import run_evaluation
        run_evaluation(config, num_trials=args.trials)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'eval'.")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()