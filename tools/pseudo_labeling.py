import torch
import os, cv2, argparse, json, math #, datetime, imgviz
import numpy as np

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import pycocotools.mask

from detectron2.utils.visualizer import Visualizer

from detectron2.evaluation.coco_evaluation import instances_to_coco_json


def instance2mask(instances , colors):
    h, w = instances.image_size
    mask = np.zeros((h, w, 3), np.uint8)
    for c, m in zip(instances.pred_classes, instances.pred_masks):
        if colors[c]:
            m = cv2.cvtColor(m.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            mask = np.where(m==[True, True, True], colors[c], mask)
    return mask

def draw_masks(bg, masks):
    mask = bg.copy()    
    for m in masks:
        mask = np.where(m==[0, 0, 0], mask, m)
        
    return mask.astype(np.uint8)


class PseudoLabeling:
    def __init__(self, threshold=0.8, device='cuda', model_path="./pretrained_model/model_final.pth"):
        cfg = get_cfg()
        cfg.thing_classes = ['intersection', 'spacing']
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.DEVICE = device
        self.labels = ['Intersection', 'Spacing']
        self.colors = [(127, 0, 0), (0, 127, 0)]
        self.predictor = DefaultPredictor(cfg)
        self.visualizer = Visualizer
        
    def predict_mask(self, weighted):
        outputs = self.predictor(weighted)
        instances = outputs["instances"].to("cpu")
        mask = instance2mask(instances, self.colors)
        return mask, instances
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset_dir', type=str, help='Path to dataset includes color and depth folders', default='./data')
    parser.add_argument("-t", '--threshold', type=float, help='THRESHOLD NUMBER', default=0.5)
    parser.add_argument("-a", "--anno", help="COCO annotation file!", required=True)
    parser.add_argument("-o", "--output", type=str, help='Annotation file output name', default=None)
    args = parser.parse_args()

    if args.output == None:
        args.output = args.anno
    
    PseudoLabelingModule = PseudoLabeling(threshold=args.threshold, device='cuda') 

    with open(args.anno) as file:
        data = json.load(file)

    cocojson = []

    os.makedirs(os.path.join(args.dataset_dir, 'Masks'), exist_ok=True)

    for content in data["images"]:
        color = cv2.imread(os.path.join(args.dataset_dir, content["file_name"]))

        mask, instances = PseudoLabelingModule.predict_mask(color)

        _, fname = os.path.split(content["file_name"])
        cv2.imwrite(os.path.join(args.dataset_dir, 'Masks/mask-{}'.format(fname)), mask)

        image_id = content["id"]

        cocojson.extend(instances_to_coco_json(instances, image_id))


    for n, content in enumerate(cocojson):
        content['id'] = n
        content['iscrowd'] = 0
        content['bbox'][0] = math.ceil(content['bbox'][0])
        content['bbox'][1] = math.ceil(content['bbox'][1])
        content['bbox'][2] = math.ceil(content['bbox'][2])
        content['bbox'][3] = math.ceil(content['bbox'][3])
        # content['area'] = content['bbox'][2] * content['bbox'][3]
        content['area'] = float(pycocotools.mask.area(content['segmentation']))
        # del content['score']
        
        mask = pycocotools.mask.decode(content['segmentation'])

        # Code source below: https://github.com/facebookresearch/Detectron/issues/100
        # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []

        for contour in contours:
            contour = contour.flatten().tolist()
            # segmentation.append(contour)
            if len(contour) > 4:
                segmentation.append(contour)
        if len(segmentation) == 0:
            continue
        
        content['segmentation'] = segmentation

    data['annotations'] = cocojson

    with open(args.output, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
