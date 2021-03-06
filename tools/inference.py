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
    register_coco_instances("2019_train", {}, "./2019_train/annotations_train.json", "./2019_train")
    register_coco_instances("2019_val", {}, "./2019_val/annotations_val.json", "./2019_val")
    # register_coco_instances("2019_da", {}, "./2019_da/annotations_train.json", "./2019_da")
    # register_coco_instances("2019_da_val", {}, "./2019_da_val/annotations_val.json", "./2019_da_val")

    # register_coco_instances("2021_test", {}, "./2021_test/annotations_test.json", "./2021_test")
    # register_coco_instances("2021_da", {}, "./2021_da/annotations_train.json", "./2021_da")
    register_coco_instances("2021_da_val", {}, "./2021_da_val/annotations_val.json", "./2021_da_val")

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

    test_path = "./2021_da_val/JPEGImages"
    files = os.listdir(test_path)
    steel_metadata = MetadataCatalog.get("2019_train")
    os.makedirs("inference-target")

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
        cv2.imwrite(os.path.join("inference-target", f), out.get_image()[:, :, ::-1])


    test_path = "./2019_val/JPEGImages"
    files = os.listdir(test_path)

    os.makedirs("inference-source")

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
        cv2.imwrite(os.path.join("inference-source", f), out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    main()
