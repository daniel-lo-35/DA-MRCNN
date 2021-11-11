import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
# import os

import torch, torchvision

import numpy as np

from detectron2.utils.logger import setup_logger
import logging
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer 


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
    # os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
    cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg_source.MODEL.WEIGHTS = "./output/model_final.pth"

    model = build_model(cfg_source)

    # cfg_target = get_cfg()
    # cfg_target.DATASETS.TRAIN = ("2021_train",)
    # cfg_target.DATASETS.TEST = ("2021_test",)
    # cfg_target.INPUT.MIN_SIZE_TRAIN = (0,)
    # cfg_target.DATALOADER.NUM_WORKERS = 2
    # cfg_target.SOLVER.IMS_PER_BATCH = 4

    DetectionCheckpointer(model).load(cfg_source.MODEL.WEIGHTS)

    data_loader_source = build_detection_test_loader(cfg_source, "2019_train")
    data_loader_target = build_detection_test_loader(cfg_source, "2021_train") 

    model.eval()

    ############ 2019 TRAIN ############
    for idx, batch in enumerate(data_loader_source):
        images = model.preprocess_image(batch)
        features, res = model.backbone(images.tensor)
        current_output3 = res["res3"].cpu().detach().numpy()
        current_output4 = res["res4"].cpu().detach().numpy()
        current_output5 = res["res5"].cpu().detach().numpy()

        if idx == 0:
            features_source3 = current_output3
            features_source4 = current_output4
            features_source5 = current_output5
        else:
            features_source3 = np.concatenate((features_source3, current_output3))
            features_source4 = np.concatenate((features_source4, current_output4))
            features_source5 = np.concatenate((features_source5, current_output5))

    ############ 2021 TRAIN ############
    for idx, batch in enumerate(data_loader_target):
        images = model.preprocess_image(batch)
        features, res = model.backbone(images.tensor)
        current_output3 = res["res3"].cpu().detach().numpy()
        current_output4 = res["res4"].cpu().detach().numpy()
        current_output5 = res["res5"].cpu().detach().numpy()

        if idx == 0:
            features_target3 = current_output3
            features_target4 = current_output4
            features_target5 = current_output5
        else:
            features_target3 = np.concatenate((features_target3, current_output3))
            features_target4 = np.concatenate((features_target4, current_output4))
            features_target5 = np.concatenate((features_target5, current_output5))

    features3 = np.concatenate((features_source3, features_target3))
    features4 = np.concatenate((features_source4, features_target4))
    features5 = np.concatenate((features_source5, features_target5))

    print(features3.shape)
    print(features4.shape)
    print(features5.shape)

    features3 = features3.reshape(features3.shape[0], features3.shape[1] * features3.shape[2] * features3.shape[3])
    features4 = features4.reshape(features4.shape[0], features4.shape[1] * features4.shape[2] * features4.shape[3])
    features5 = features5.reshape(features5.shape[0], features5.shape[1] * features5.shape[2] * features5.shape[3])

    from sklearn.manifold import TSNE
    tsne3 = TSNE(n_components=2).fit_transform(features3)
    tsne4 = TSNE(n_components=2).fit_transform(features4)
    tsne5 = TSNE(n_components=2).fit_transform(features5)

    print(tsne3.shape)
    print(tsne4.shape)
    print(tsne5.shape)

    from matplotlib import pylab

    pylab.figure(figsize=(10,10))
    for i in np.arange(len(tsne3)):
        c = 'r'
        if i < 176:    # len(data_loader_source) = 176
            c = 'b'
        x, y = (tsne3[i,0],tsne3[i,1])
        pylab.scatter(x,y,c=c)
        #pylab.annotate(str(Lab[i][0]), xy=(x,y), xytext=(0, 0), textcoords='offset points',
                    #ha='right', va='bottom')
    # pylab.show()
    pylab.savefig("tSNE-res3.png")

    pylab.clf()

    pylab.figure(figsize=(10,10))
    for i in np.arange(len(tsne4)):
        c = 'r'
        if i < 176:    # len(data_loader_source) = 176
            c = 'b'
        x, y = (tsne4[i,0],tsne4[i,1])
        pylab.scatter(x,y,c=c)
        #pylab.annotate(str(Lab[i][0]), xy=(x,y), xytext=(0, 0), textcoords='offset points',
                    #ha='right', va='bottom')
    # pylab.show()
    pylab.savefig("tSNE-res4.png")

    pylab.clf()

    pylab.figure(figsize=(10,10))
    for i in np.arange(len(tsne5)):
        c = 'r'
        if i < 176:    # len(data_loader_source) = 176
            c = 'b'
        x, y = (tsne5[i,0],tsne5[i,1])
        pylab.scatter(x,y,c=c)
        #pylab.annotate(str(Lab[i][0]), xy=(x,y), xytext=(0, 0), textcoords='offset points',
                    #ha='right', va='bottom')
    # pylab.show()
    pylab.savefig("tSNE-res5.png")

    # Plot Note: Blue is 2019, Red is 2021


if __name__ == "__main__":
    main()