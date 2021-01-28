import os, cv2
from detectron2.engine import DefaultPredictor
from matplotlib import pyplot as plt

from .cv2utils import plot_clearly
from .config import config

def predict(img_dir,out_dir ,img):
    predictor = DefaultPredictor(config())
    try:
        file = os.path.join(img_dir, img)
        im = cv2.imread(file)
        outputs = predictor(im)  
        plot_clearly(file, outputs["instances"].to("cpu"))
        plt.savefig(os.path.join(out_dir, img), bbox_inches = 'tight', pad_inches = 0)
        return output_parser(outputs["instances"].to("cpu")) 
    except Exception as e:
        print(e)
        return "Error occurs."


def output_parser(instance):
    pred_boxes = instance.pred_boxes if instance.has("pred_boxes") else None
    scores = instance.scores if instance.has("scores") else None
    pred_classes = instance.pred_classes if instance.has("pred_classes") else None
    classes = [c.numpy().tolist() for c in pred_classes]
    boxes = [pred_boxes[i].tensor.squeeze(0).numpy().tolist() for i in range(len(pred_boxes))]
    scores = [s.numpy().tolist() for s in scores] # suitable for only one class
    return {"classes": classes, "boxes": boxes, "scores": scores}