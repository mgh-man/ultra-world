from ultralytics import YOLO

# Create a YOLO-World model
model = YOLO("/mnt/code/manfenglin/ultralytics/runs/ISDD/new/Tra-trainval(isdd-noValText)+Te-Test(isdd)/mv2-psconv2/weights/best.pt")  # or select yolov8m/l-world.pt for different sizes

# Conduct model validation on the COCO8 example dataset
metrics = model.val(data="ISDD+SII.yaml",name='ISDD/mv2-psconv2', batch=8, device='1')