import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

import detectron2.utils.comm as comm
import torch, torchvision

import numpy as np

from detectron2.utils.logger import setup_logger
import logging
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter #, TensorboardXWriter

from detectron2.engine import launch


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
    # cfg_source.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))    # Detectron2's new MRCNN baseline, to be investigated
    cfg_source.DATASETS.TRAIN = ("2019_train",)
    # cfg_source.DATASETS.TEST = ()
    cfg_source.DATASETS.TEST = ("2021_train",)
    # cfg_source.TEST.EVAL_PERIOD = 300
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

    # cfg_source.MODEL.WEIGHTS = "./pretrained-model/model_final.pth"     # Import pretrained weight from previous Steelscape models

    model = build_model(cfg_source)

    cfg_target = get_cfg()
    cfg_target.DATASETS.TRAIN = ("2021_train",)
    cfg_target.DATASETS.TEST = ("2021_test",)
    cfg_target.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_target.DATALOADER.NUM_WORKERS = 2
    cfg_target.SOLVER.IMS_PER_BATCH = 4


    """ TRAIN """
    resume = False
    
    model.train()
    print(model)
    
    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg_source.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume = resume).get("iteration", -1) + 1
    )
    max_iter = cfg_source.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg_source.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg_source.OUTPUT_DIR, "metrics.json")),
            # TensorboardXWriter(cfg_source.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    
    i = 1
    max_epoch = 545.45 # max iter / min(data_len(data_source, data_target))
    current_epoch = 0
    data_len = 55

    alpha3 = 0
    alpha4 = 0
    alpha5 = 0

    data_loader_source = build_detection_train_loader(cfg_source)
    data_loader_target = build_detection_train_loader(cfg_target) 
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data_source, data_target, iteration in zip(data_loader_source, data_loader_target, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            
            if (iteration % data_len) == 0:
                current_epoch += 1
                i = 1

            p = float( i + current_epoch * data_len) / max_epoch / data_len
            alpha = 2. / ( 1. + np.exp( -10 * p)) - 1
            i += 1

            alpha3 = alpha
            alpha4 = alpha
            alpha5 = alpha

            # if alpha3 > 0.5:
            #     alpha3 = 0.5

            # if alpha4 > 0.5:
            #     alpha4 = 0.5

            # if alpha5 > 0.1:
            #     alpha5 = 0.1
            
            loss_dict = model(data_source, False, alpha3, alpha4, alpha5)
            loss_dict_target = model(data_target, True, alpha3, alpha4, alpha5)
            loss_dict["loss_r3"] += loss_dict_target["loss_r3"]
            loss_dict["loss_r4"] += loss_dict_target["loss_r4"]
            loss_dict["loss_r5"] += loss_dict_target["loss_r5"]

            loss_dict["loss_r3"] *= 0.5
            loss_dict["loss_r4"] *= 0.5
            loss_dict["loss_r5"] *= 0.5

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            storage.put_scalar("alpha", alpha, smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


if __name__ == "__main__":
    launch(main, 2)    # Use 2 GPUs to prevent CUDA out of memory