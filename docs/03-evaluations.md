# Evaluation

- This codebase uses COCOAPI evaluation implementation. At the heart of it lies `CocoEvaluator` class.
- A `CocoEvaluator` object has an attribute name `coco_eval` whose `stats` attribute contains the metrics
```
(Pdb) coco_eval.stats
array([0.29048652, 0.49894725, 0.29046043, 0.10394502, 0.31172866,
       0.4657031 , 0.26193177, 0.43025718, 0.47090144, 0.19861693,
       0.5099824 , 0.72390345])
```
- In this project, we're only dealing with object detection. So if you have a `CocoEvaluator` instance, you can get the stats vector by `coco_evaluator.coco_eval['bbox']`
- The final summary is produced by calling `coco_eval.summarize()`. 
```
(Pdb) coco_eval.summarize()
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.104
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.262
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724
```