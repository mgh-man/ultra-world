from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
data = dict(
    train=dict(
        # yolo_data=["ISDD+SII.yaml","ISDD+SII2.yaml"],
        yolo_data=["data_SII.yaml","ISDD+SII.yaml","ISDD+SII2.yaml"],
        # grounding_data=[
        #     # dict(
        #     #     img_path="/mnt/datasets/manfenglin/ISDD/images/val",
        #     #     # json_file="/mnt/code/manfenglin/ultralytics/datasets/flickr30k/final_flickr_separateGT_test.json",
        #     #     json_file="/mnt/code/manfenglin/ultralytics/datasets/isdd/val_converted.json",
        #     # ),
        #     dict(
        #         img_path="/mnt/datasets/manfenglin/dataset_SII/images/train",
        #         json_file="/mnt/code/manfenglin/ultralytics/datasets/sii/train_converted.json",
        #     ),
        #     # dict(
        #     #     img_path="/mnt/datasets/manfenglin/dataset_SII/images/val",
        #     #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sii/val_converted.json",
        #     # ),
        #     dict(
        #         img_path="/mnt/datasets/manfenglin/ISDD/images/train",
        #         json_file="/mnt/code/manfenglin/ultralytics/datasets/isdd/train.json",
        #     ),
        #     # dict(
        #     #     img_path="/mnt/datasets/manfenglin/sfisd/images/train",
        #     #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sfisd/train_converted.json",
        #     # ),
        # ],
    ),
    # val=dict(yolo_data=["ISDD+SII.yaml"]),
    val=dict(yolo_data=["data_SII.yaml"]),
)
model = YOLOWorld("yolov8x-worldv2.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="ISDD+SII.yaml", epochs=150, imgsz=640, batch=8, 
                      project='runs/ISDD+SII/new/Tra-trainval(noValText)+Te-Test', name='xv2-notext-trainval', device='2')