from __future__ import division

import argparse
import json

from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for (img_path, input_img) in dataloader:

        # Configure input
        input_img = Variable(input_img.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # determine original image size
        img = Image.open(img_path[0])

        # format prediction
        out = []
        for detection in detections:
            detections = rescale_boxes(detection, opt.img_size, img.size[::-1])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection.tolist():
                out.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls_conf": cls_conf,
                    "cls": classes[int(cls_pred)]
                })

        with open(f"bbox.json", 'w') as fp:
            json.dump(out, fp, indent=4, sort_keys=True)
