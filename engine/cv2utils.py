from detectron2.data.catalog import MetadataCatalog
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import matplotlib.patches as patches
import cv2

def cv2_imshow(image_path:str):
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.show()


def show_clearly(filepath ,predictions, metadata=None):
    plot_clearly(filepath ,predictions, metadata=None)
    plt.show()


def plot_clearly(filepath ,predictions, metadata=None):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
    labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))
    plt.imshow(imgplt.imread(filepath))
    currentAxis=plt.gca()
    _draw_boxes(currentAxis, boxes, labels)


def _draw_boxes(axis, boxes, labels):
    for i in range(len(boxes)):
        box = boxes[i]
        pos = box.tensor.squeeze(0)
        rect=patches.Rectangle((pos[0], pos[1]), pos[2]-pos[0], pos[3]-pos[1], linewidth=1,edgecolor='r',facecolor='none')
        axis.add_patch(rect)
        plt.text(pos[0], pos[1]-2, labels[i], color='mediumvioletred', fontsize=8)
        


def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels