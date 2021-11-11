# DA-MRCNN
Detectron2 implementation of DA-MRCNN

## Installation
Follow the [guide on Detectron2's documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install Detectron2.

Then replace `detectron2/modeling/meta_arch/rcnn.py` with our `DA-MRCNN/rcnn.py`, and `detectron2/modeling/backbone/fpn.py` with our `DA-MRCNN/fpn.py`.

## Data Preparation
We use COCO format to register the dataset.
```
register_coco_instances("dataset_name_soruce_training",{},"path_annotations","path_images")
register_coco_instances("dataset_name_target_training",{},"path_annotations","path_images")
register_coco_instances("dataset_name_target_test",{},"path_annotations","path_images")
```

## Pretrained weight
A pretrained model is recommended. It can be prepared from [ReGion-Based Detector (RGBD)](https://github.com/SJ-Chuang/rgbd).

## Train
`tools/trainer.py`

_Remarks:_ If the last checkpoint is 30000 iteration (e.g. from the pretrained weight above), the MAX_ITER must be greater than 30000.

## Evaluation
`tools/evaluate.py` uses COCO AP evaluations.

## Inference
`tools/inference.py`

## Reference
<a id="1">[1]</a> 
Pasqualino, G., Furnari, A., Signorello, G., Farinella, G. M. (2021). 
[An unsupervised domain adaptation scheme for single-stage artwork recognition in cultural sites.](https://doi.org/10.1016/j.imavis.2021.104098)
Image and Vision Computing, 107, 104098.

<a id="2">[2]</a> 
Chuang, S. J. et al (2021). 
On-site Rebar Spacing Inspection using Deep-learning-based Image Segmentation. 
Comput Aided Civ Inf, 2021;00:1–9.

<a id="3">[3]</a> 
Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019). 
[Detectron2.](https://github.com/facebookresearch/detectron2)
