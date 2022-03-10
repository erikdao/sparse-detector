"""
Entry point for training DETR Baseline
"""
import pprint
import click

from sparse_detector.config import load_config


@click.command("train_baseline")
@click.option('--config', default='', help="Path to config file")
@click.option('--exp_name', default=None, help='Experiment name. Need to be set')
def main(config, exp_name):
    configs = load_config(config)
    pprint.pprint(configs)

if __name__ == "__main__":
    main()
