import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import torch, torchvision

import numpy as np

from detectron2.utils.logger import setup_logger
import logging
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer 


def main():
    register_coco_instances("2019_train", {}, "./2019_train/annotations_train.json", "./2019_train")
    register_coco_instances("2019_val", {}, "./2019_val/annotations_val.json", "./2019_val")
    # register_coco_instances("2019_da", {}, "./2019_da/annotations_train.json", "./2019_da")
    # register_coco_instances("2019_da_val", {}, "./2019_da_val/annotations_val.json", "./2019_da_val")

    register_coco_instances("2021_test", {}, "./2021_test/annotations_test.json", "./2021_test")
    # register_coco_instances("2021_da", {}, "./2021_da/annotations_train.json", "./2021_da")
    # register_coco_instances("2021_da_val", {}, "./2021_da_val/annotations_val.json", "./2021_da_val")

    setup_logger()
    logger = logging.getLogger("detectron2")

    cfg_source = get_cfg()
    cfg_source.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg_source.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))
    cfg_source.DATASETS.TRAIN = ("2019_train",)
    cfg_source.DATASETS.TEST = ()
    cfg_source.DATALOADER.NUM_WORKERS = 2
    cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")
    cfg_source.SOLVER.IMS_PER_BATCH = 4
    cfg_source.SOLVER.BASE_LR = 0.00005
    cfg_source.SOLVER.WEIGHT_DECAY = 0.001
    cfg_source.SOLVER.MAX_ITER = 30000
    cfg_source.SOLVER.STEPS = (300,)
    cfg_source.SOLVER.GAMMA = 0.5
    cfg_source.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_source.INPUT.MIN_SIZE_TEST = 0
    os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
    cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg_source.MODEL.WEIGHTS = "./output/model_final.pth"

    model = build_model(cfg_source)


    DetectionCheckpointer(model).load(cfg_source.MODEL.WEIGHTS)

    """ TEST """
    print("############ 2019 TRAIN ############")
    evaluator = COCOEvaluator("2019_train", cfg_source, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg_source, "2019_train")
    inference_on_dataset(model, val_loader, evaluator)

    print("############ 2019 VAL ############")
    evaluator = COCOEvaluator("2019_val", cfg_source, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg_source, "2019_val")
    inference_on_dataset(model, val_loader, evaluator)

    print("############ 2021 TEST ############")
    evaluator = COCOEvaluator("2021_test", cfg_source, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg_source, "2021_test")
    inference_on_dataset(model, val_loader, evaluator)


if __name__ == "__main__":
    main()