"""
Entry point for training DETR Baseline
"""
import sys
import pprint
from pathlib import Path
import click

package_root = Path(__file__).parent.parent
sys.path.insert(0, package_root)

from sparse_detector.configs import load_config


@click.command("train_baseline")
@click.option('--config', default='', help="Path to config file")
@click.option('--exp-name', default=None, help='Experiment name. Need to be set')
@click.option('--seed', default=42, type=int)
@click.option('--lr', default=1e-4, type=float, help="Transformer detector learing rate")
@click.option('--lr-backbone', default=1e-5, type=float, help="Backbone learning rate")
@click.option('--batch-size', default=8, type=int, help="Batch size per GPU")
@click.option('--weight-decay', default=1e-4, type=float, help="Optimizer's weight decay")
@click.option('--epochs', default=300, type=int, help="Number of epochs")
@click.option('--lr-drop', default=200,type=int, help="Epoch after which learning rate is dropped")
@click.option('--clip-max-norm', default=0.1, type=float, help='gradient clipping max norm')
@click.option('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
@click.option('--dilation/--no-dilation', default=False, help="If true, we replace stride with dilation in the last convolutional block (DC5)")
@click.option('--position-embedding', default='sine', type=str, help="Type of positional embedding to use on top of the image features")
@click.option('--enc-layers', default=6, type=int, help="Number of encoding layers in the transformer")
@click.option('--dec-layers', default=6, type=int, help="Number of decoding layers in the transformer")
@click.option('--dim-feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
@click.option('--hidden-dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
@click.option('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
@click.option('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
@click.option('--num-queries', default=100, type=int, help="Number of query slots")
@click.option('--pre-norm/--no-pre-norm', default=True)
@click.option('--aux-loss/--no-aux-loss', default=True, help="Whether to use auxiliary decoding losses (loss at each layer)")
@click.option('--set-cost-class', default=1, type=float, help="Class coefficient in the matching cost")
@click.option('--set-cost-bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
@click.option('--set-cost-giou', default=2, type=float, help="giou box coefficient in the matching cost")
@click.option('--bbox-loss-coef', default=5, type=float)
@click.option('--giou-loss-coef', default=2, type=float)
@click.option('--eos-coef', default=0.1, type=float, help="Relative classification weight of the no-object class")
@click.option('--dataset-file', default='coco')
@click.option('--coco-path', type=str)
@click.option('--output-dir', default='', help='path where to save, empty for no saving')
@click.option('--resume', default='', help='resume from checkpoint')
@click.option('--start-epoch', default=0, type=int, help='start epoch')
@click.option('--num_workers', default=12, type=int)
@click.option('--dist_url', default='env://', help='url used to set up distributed training')
def main(
    config, exp_name, seed, backbone, lr_backbone, lr, batch_size, weight_decay, epochs,
    lr_drop, clip_max_norm, dilation, postion_embedding, enc_layers, dec_layers, dim_feedforward,
    hidden_dim, dropout, nheads, num_queries, pre_norm, aux_loss, set_cost_class,
    set_cost_bbox, set_cost_giou, bbox_loss_coef, giou_loss_coef, eos_coef,
    dataset_file, coco_path, output_dir, resume, start_epoch, num_workers, dist_url
):
    args = locals()
    pprint.pprint(args)
    # configs = load_config(config)
    # pprint.pprint(configs)

if __name__ == "__main__":
    main()

