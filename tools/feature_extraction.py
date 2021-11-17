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
    # register_coco_instances("2019_train", {}, "./2019_train/annotations_train.json", "./2019_train")
    # register_coco_instances("2019_val", {}, "./2019_val/annotations_val.json", "./2019_val")
    register_coco_instances("2019_da", {}, "./2019_da/annotations_train.json", "./2019_da")
    register_coco_instances("2019_da_val", {}, "./2019_da_val/annotations_val.json", "./2019_da_val")

    # register_coco_instances("2021_test", {}, "./2021_test/annotations_test.json", "./2021_test")
    register_coco_instances("2021_da", {}, "./2021_da/annotations_train.json", "./2021_da")
    register_coco_instances("2021_da_val", {}, "./2021_da_val/annotations_val.json", "./2021_da_val")

    setup_logger()
    logger = logging.getLogger("detectron2")

    cfg_source = get_cfg()
    cfg_source.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg_source.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))    # Detectron2's new MRCNN baseline, to be investigated
    cfg_source.DATASETS.TRAIN = ("2019_da",)
    # cfg_source.DATASETS.TEST = ()
    cfg_source.DATASETS.VAL = ("2019_da_val",)
    cfg_source.DATASETS.TEST = ("2021_da",)
    # cfg_source.TEST.EVAL_PERIOD = 300
    cfg_source.DATALOADER.NUM_WORKERS = 2
    cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")
    cfg_source.SOLVER.IMS_PER_BATCH = 4
    cfg_source.SOLVER.BASE_LR = 0.00005
    cfg_source.SOLVER.WEIGHT_DECAY = 0.001
    cfg_source.SOLVER.MAX_ITER = 60000
    cfg_source.SOLVER.STEPS = (300,)
    cfg_source.SOLVER.GAMMA = 0.5
    cfg_source.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_source.INPUT.MIN_SIZE_TEST = 0
    cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg_source.MODEL.WEIGHTS = "./output/model_final.pth"

    model = build_model(cfg_source)


    DetectionCheckpointer(model).load(cfg_source.MODEL.WEIGHTS)

    data_loader_source1 = build_detection_test_loader(cfg_source, "2019_da")
    data_loader_source2 = build_detection_test_loader(cfg_source, "2019_da_val")
    data_loader_target1 = build_detection_test_loader(cfg_source, "2021_da")
    data_loader_target2 = build_detection_test_loader(cfg_source, "2021_da_val") 

    model.eval()

    ############ 2019 TRAIN ############
    for idx, batch in enumerate(data_loader_source1):
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

    for idx, batch in enumerate(data_loader_source2):
        images = model.preprocess_image(batch)
        features, res = model.backbone(images.tensor)
        current_output3 = res["res3"].cpu().detach().numpy()
        current_output4 = res["res4"].cpu().detach().numpy()
        current_output5 = res["res5"].cpu().detach().numpy()

        features_source3 = np.concatenate((features_source3, current_output3))
        features_source4 = np.concatenate((features_source4, current_output4))
        features_source5 = np.concatenate((features_source5, current_output5))


    ############ 2021 TRAIN ############
    for idx, batch in enumerate(data_loader_target1):
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

    for idx, batch in enumerate(data_loader_target2):
        images = model.preprocess_image(batch)
        features, res = model.backbone(images.tensor)
        current_output3 = res["res3"].cpu().detach().numpy()
        current_output4 = res["res4"].cpu().detach().numpy()
        current_output5 = res["res5"].cpu().detach().numpy()

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
        if i < 302:    # len(data_loader_source) = 302
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
        if i < 302:    # len(data_loader_source) = 302
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
        if i < 302:    # len(data_loader_source) = 302
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