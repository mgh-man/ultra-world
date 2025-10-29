# from ultralytics import YOLOWorld
# from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

# # data = dict(
# #     train=dict(
# #         yolo_data=["data_isddPLUS.yaml"]
# #     ),
# #     val=dict(yolo_data=["data_isddPLUS.yaml"]),
# # )
# model = YOLOWorld("yolov8m-worldv2.yaml")
# model.train(data="all2.yaml", batch=8, epochs=150, project='runs/all2/train:(train+test)-val:val-test:val', name='mv2', device='3')
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8x.pt")  # or "yolov8l.pt" for a pre-trained model

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data_SII.yaml", batch=16, epochs=150, project='runs/SII/new/Tra-trainval(sii)+Te-Test(sii)', name='pretrain-notext-xv2', device='2')