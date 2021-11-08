import detectron2
from detectron2.data.datasets import register_coco_instances
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

import torch, torchvision

import numpy as np

from detectron2.utils.logger import setup_logger
import logging

from detectron2.engine import launch

from detectron2.utils.visualizer import ColorMode



def main():
    register_coco_instances("2019_train", {}, "./train_2019/annotations_train.json", "./train_2019")
    # register_coco_instances("2019_val", {}, "./val_2019/annotations_val.json", "./val_2019")
    # register_coco_instances("2019_test", {}, "./test_2019/annotations_test.json", "./test_2019")

    register_coco_instances("2021_train", {}, "./train_2021/annotations_train.json", "./train_2021")
    # register_coco_instances("2021_val", {}, "./val_2021/annotations_val.json", "./val_2021")
    register_coco_instances("2021_test", {}, "./test_2021/annotations_test.json", "./test_2021")

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
    cfg_source.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_source.INPUT.MIN_SIZE_TEST = 0
    os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
    cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg_source.MODEL.WEIGHTS = "./output/model_final.pth"


    """ INFERENCE """
    predictor = DefaultPredictor(cfg_source)

    test_path = "./test_2021/JPEGImages"
    files = os.listdir(test_path)
    steel_metadata = MetadataCatalog.get("2019_train")
    os.makedirs("inference-target", exist_ok=True)

    for f in files:
        im = cv2.imread(os.path.join(test_path, f))
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=steel_metadata, 
                    # scale=0.5, 
                    # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite(os.path.join("inference", f), out.get_image()[:, :, ::-1])


    """ INFERENCE """
    predictor = DefaultPredictor(cfg_source)

    test_path = "./train_2019/JPEGImages"
    files = os.listdir(test_path)
    steel_metadata = MetadataCatalog.get("2019_train")
    os.makedirs("inference-source", exist_ok=True)

    for f in files:
        im = cv2.imread(os.path.join(test_path, f))
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=steel_metadata, 
                    # scale=0.5, 
                    # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite(os.path.join("inference", f), out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    launch(main, 2)