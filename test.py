from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

data = dict(
    train=dict(
        # yolo_data=["ISDD+SII.yaml","ISDD+SII2.yaml"],
        YOLO_data=["data_isddPLUS.yaml"],
        grounding_data=[
            dict(
                img_path="/mnt/datasets/manfenglin/ISDD/images/val",
                # json_file="/mnt/code/manfenglin/ultralytics/datasets/flickr30k/final_flickr_separateGT_test.json",
                json_file="/mnt/code/manfenglin/ultralytics/datasets/isdd/val_converted.json",
            ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/dataset_SII/images/train",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sii/train_converted.json",
            # ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/dataset_SII/images/val",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sii/val_converted.json",
            # ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/ISDD/images/train",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/isdd/train.json",
            # ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/sfisd/images/train",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sfisd/train_converted.json",
            # ),
        ],
    ),
    val=dict(yolo_data=["data_isddPLUS.yaml"]),
)
model = YOLOWorld("yolov8l-worldv2.yaml")
# model = YOLOWorld("yolov8l-worldv2.pt")
model.train(data=data, batch=8, epochs=150, trainer=WorldTrainerFromScratch, 
            project='runs/ISDD/train:(train+val_isdd+val_text)-val:isdd_test-test:isdd_test', name='lv2-pkaloss7-3', device='2')