from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("/mnt/code/manfenglin/ultralytics/runs/ISDD/new/Tra-trainval(isdd-noValText)+Te-Test(isdd)/ourpretrain-xv2-device5-mixup1.0-HAFB(2,3,4)-psconv-p2-pka7-3-5500-batch8-albumentions/weights/best.pt")
# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict(source='/mnt/datasets/manfenglin/ISDD/images/test',
                #   imgsz=640,
                  project='runs/detect/predict/',
                  name='isdd',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )

# Show results
# results[0].show()