from typing import Any, Optional
from sparse_detector.models.detr import build


def build_model(
    backbone: str,
    lr_backbone: float,
    dilation: bool,
    return_interm_layers: bool,
    position_embedding: str,
    hidden_dim: int,
    enc_layers: int,
    dec_layers: int,
    dim_feedforward: Optional[int] = 2048,
    dropout: Optional[float] = 0.1,
    num_queries: Optional[int] = 100,
    bbox_loss_coef: Optional[float] = 5,
    giou_loss_coef: Optional[float] = 2,
    eos_coef: Optional[float] = 0.1,
    aux_loss: Optional[bool] = False,
    set_cost_class: Optional[int] = 1,
    set_cost_bbox: Optional[int] = 5,
    set_cost_giou: Optional[int] = 2,
    nheads: Optional[int] = 8,
    pre_norm: Optional[bool] = True,
    dataset_file: Optional[str] = 'coco',
    device: Optional[Any] = None
):
    return build(
        backbone,
        lr_backbone,
        dilation,
        return_interm_layers,
        position_embedding,
        hidden_dim,
        enc_layers,
        dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=num_queries,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        eos_coef=eos_coef,
        aux_loss=aux_loss,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        nheads=nheads,
        pre_norm=pre_norm,
        dataset_file=dataset_file,
        device=device
    )
