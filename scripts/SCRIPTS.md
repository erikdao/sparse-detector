# Scripts

Various ad-hoc scripts and command lines during the development of this project.


**Attention visualizations**
Produce visualisations for attention maps from all decoder's layers.
```
for i in {0..5}; do python -m scripts.visualize_attentions misc/input_images/000000192670.jpg checkpoints/baseline_detr/checkpoint.pth --decoder-act softmax --decoder-layer "$i"; done

python -m scripts.visualize_attentions misc/input_images/000000002473.jpg checkpoints/baseline_detr/checkpoint.pth --decoder-act softmax

python -m scripts.visualize_attentions misc/input_images/000000002473.jpg checkpoints/decoder_sparsemax_cross-mha/checkpoint.pth --decoder-act sparsemax
```

```
python -m scripts.inspect_attentions misc/input_images/000000002473.jpg checkpoints/decoder_sparsemax_cross-mha/checkpoint.pth --decoder-act sparsemax

python -m scripts.inspect_attentions misc/input_images/000000002473.jpg checkpoints/baseline_detr/checkpoint.pth  --decoder-act softmax
```