import os, time, cv2
from detectron2.engine import DefaultPredictor
from matplotlib import pyplot as plt
import numpy as np
import pySPM
from .cv2utils import plot_clearly
from .config import config

def predict(img_dir, out_dir, img):
    # delete earlier files
    upload_dir = os.listdir(img_dir)
    for pic in upload_dir:
        if (time.time() - os.path.getctime(os.path.join(img_dir, pic))) > 60:
            os.remove(os.path.join(img_dir, pic))
    result_dir = os.listdir(out_dir)
    for pic in result_dir:
        if (time.time() - os.path.getctime(os.path.join(out_dir, pic))) > 60:
            os.remove(os.path.join(out_dir, pic))
    #  predict and save
    predictor = DefaultPredictor(config())
    try:
        size = None
        file = os.path.join(img_dir, img)
        if file.endswith(".sxm"):
            size = sxm_converter(file)
            file = file + ".png"
            img = img + ".png"
        im = cv2.imread(file)
        gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img[..., np.newaxis]
        outputs = predictor(gray_img)  
        plot_clearly(file, outputs["instances"].to("cpu"))
        plt.savefig(os.path.join(out_dir, img), bbox_inches = 'tight', pad_inches = 0)
        return img, output_parser(outputs["instances"].to("cpu"), size) 
    except Exception as e:
        print(e.__str__())
        return "error", "error"

# prepare source json data
def output_parser(instance, size):
    pred_boxes = instance.pred_boxes if instance.has("pred_boxes") else None
    scores = instance.scores if instance.has("scores") else None
    pred_classes = instance.pred_classes if instance.has("pred_classes") else None
    classes = [c.numpy().tolist() for c in pred_classes]
    boxes = [pred_boxes[i].tensor.squeeze(0).numpy().tolist() for i in range(len(pred_boxes))]
    scores = [s.numpy().tolist() for s in scores] # suitable for only one class

    # deal with the size of CO
    if size is not None:
        print('in process')
    return {"classes": classes, "boxes": boxes, "scores": scores}


def sxm_converter(file):
    sxm_pic = pySPM.SXM(file).get_channel('Current')
    sxm_pic.show(cmap='viridis')
    plt.title(None)
    plt.axis('off')
    plt.savefig(file+".png", bbox_inches = 'tight', pad_inches = 0)
    return sxm_pic.size['real']
