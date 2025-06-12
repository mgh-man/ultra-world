from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8m-worldv2.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data_isddPLUS.yaml", epochs=150, imgsz=640, batch=8, 
                      project='runs/ISDD/train:(train+val)-val:test-test:test', name='mv2', device='2')