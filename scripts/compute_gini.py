"""
This script is used to compute the Gini score for all decoder layers' attention maps
It's another adhoc script
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore', 'UserWarning')

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch
import numpy as np
from tqdm import tqdm

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.utils.metrics import gini
from sparse_detector.utils import distributed  as dist_utils
from sparse_detector.configs import build_detr_config
from sparse_detector.datasets.loaders import build_dataloaders

@click.command()
@click.option('--seed', type=int, default=42)
@click.option('--decoder-act', type=str, default='sparsemax')
@click.option('--coco-path', type=str, default="./data/COCO")
@click.option('--num-workers', default=12, type=int)
@click.option('--batch-size', default=6, type=int, help="Batch size per GPU")
@click.option('--dist_url', default='env://', help='url used to set up distributed training')
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--detection-threshold', default=0.7, help='Threshold to filter detection results')
def main(resume_from_checkpoint, seed, decoder_act, coco_path, num_workers, batch_size, dist_url, detection_threshold):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dist_config = dist_utils.init_distributed_mode(dist_url)

    click.echo("Loading configs")
    configs = build_detr_config(device=device)
    if decoder_act:
        configs['decoder_act'] = decoder_act
    print(configs)

    click.echo("Building model with configs")
    model, criterion, postprocessors = build_model(**configs)

    click.echo("Load model from checkpoint")
    checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    describe_model(model)
    criterion.eval()

    click.echo("Building dataset")
    data_loader, base_ds = build_dataloaders('val', coco_path, batch_size, dist_config.distributed, num_workers)
    
    click.echo("Computing gini")
    dataset_gini = []
    for batch_idx, (samples, targets) in tqdm(enumerate(data_loader)):
        attentions = []
        conv_features = []
        hooks = [model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),]

        for i in range(6):
            hooks.append(
                model.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
                    lambda self, input, output: attentions.append(output[1])
                )
            )

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        for hook in hooks:
            hook.remove()
        h, w = conv_features[0]['3'].tensors.shape[-2:]

        pred_logits = outputs['pred_logits'].detach().cpu()  # [B, num_queries, num_classes]
        probas = pred_logits.softmax(-1)[:, :, :-1]
        batch_keep = probas.max(-1).values > detection_threshold

        batch_gini = []
        # For each image in the batch
        for img_idx, keep in enumerate(batch_keep):
            assert keep.shape == (100,)
            queries = keep.nonzero().squeeze(-1)
            if len(queries) == 0:
                continue

            # List of attention maps from all decoder's layer for this particular image
            img_attentions = [attn[img_idx].detach().cpu() for attn in attentions]
            assert len(img_attentions) == 6
            
            image_gini = []
            for layer_idx, layer_attn in enumerate(img_attentions):
                attn_gini = 0.0
                for query in queries:
                    attn = layer_attn[query].view(w, h).detach().cpu()
                    attn_gini += gini(attn)
                
                attn_gini /= len(queries)
                image_gini.append(attn_gini)
            
            image_gini_t = torch.stack(image_gini)
            batch_gini.append(image_gini_t)
        
        dataset_gini.extend(batch_gini)
        del attentions
        del conv_features
    
    dataset_gini_t = torch.stack(dataset_gini)
    print(dataset_gini_t.shape)

    print(f"Mean: {torch.mean(dataset_gini_t, 0)}")
    print(f"Std: {torch.std(dataset_gini_t, 0)}")

if __name__ == "__main__":
    main()