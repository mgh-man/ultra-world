from ultralytics import YOLO
from ultralytics import YOLOWorld
# Create a YOLO-World model
# model = YOLOWorld("/mnt/code/manfenglin/ultralytics/runs/ISDD/new/Tra-trainval(isdd-noValText)+Te-Test(isdd)/ourpretrain-xv2-device5-mixup1.0-HAFB(2,3,4)-psconv-p2-pka7-3-5500-batch8-albumentions/weights/best.pt")  # or select yolov8m/l-world.pt for different sizes
model = YOLOWorld("/mnt/code/manfenglin/ultralytics/runs/SII/new/Tra-trainval(sii-noValText)+Te-Test(sii)/ourpretrain(last)-xv2-device7-mixup0.1-HAFB(2,3,4)-psconv-p2-pka7-3-5500-batch8-albumentions/weights/best.pt")  # or select yolov8m/l-world.pt for different sizes
# Conduct model validation on the COCO8 example dataset
metrics = model.val(data="data_SII.yaml",name='image/sii', batch=8, device='2')