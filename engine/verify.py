import os, cv2
from detectron2.engine import DefaultPredictor

from .cv2utils import show_clearly
from .config import config

# target_dir = "D:\\data\\tip_data_of_CO\\png_Au"
target_dir = "/media/cocca/cocca/data/png_good"
predictor = DefaultPredictor(config())

for _, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".png"):
            file = os.path.join(target_dir, file)
            im = cv2.imread(file)
            outputs = predictor(im)  
            show_clearly(file, outputs["instances"].to("cpu"))


# dataset_dicts = DatasetCatalog.get("co_val")
# for d in dataset_dicts:  
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     show_clearly(d["file_name"], outputs["instances"].to("cpu"))

