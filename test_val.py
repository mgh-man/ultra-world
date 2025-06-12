from ultralytics import YOLO

# Create a YOLO-World model
model = YOLO("/mnt/code/manfenglin/ultralytics/runs/notext-test2text/xv2-isdd/weights/best.pt")  # or select yolov8m/l-world.pt for different sizes

# Conduct model validation on the COCO8 example dataset
metrics = model.val(data="data_isdd.yaml")