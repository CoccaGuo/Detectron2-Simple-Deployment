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
        plt.savefig(os.path.join(out_dir, img))
        return outputs["instances"].to("cpu")
    except Exception as e:
        return "Error occurs."