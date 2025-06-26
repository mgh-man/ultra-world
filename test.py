from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
import torch
import warnings

# 禁用确定性算法以避免CUDA操作警告
torch.use_deterministic_algorithms(False)

# 或者如果需要保持部分确定性，可以只禁用警告
# torch.use_deterministic_algorithms(True, warn_only=True)
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
data = dict(
    train=dict(
        yolo_data=["ISDD+SII.yaml","ISDD+SII2.yaml"],
        # yolo_data=["data_isddPLUS.yaml"],
        grounding_data=[
            # dict(
            #     img_path="/mnt/datasets/manfenglin/ISDD/images/val",
            #     # json_file="/mnt/code/manfenglin/ultralytics/datasets/flickr30k/final_flickr_separateGT_test.json",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/isdd/val_converted.json",
            # ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/dataset_SII/images/train",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sii/train_converted.json",
            # ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/dataset_SII/images/val",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sii/val_converted.json",
            # ),
            dict(
                img_path="/mnt/datasets/manfenglin/ISDD/images/train",
                json_file="/mnt/code/manfenglin/ultralytics/datasets/isdd/train.json",
            ),
            # dict(
            #     img_path="/mnt/datasets/manfenglin/sfisd/images/train",
            #     json_file="/mnt/code/manfenglin/ultralytics/datasets/sfisd/train_converted.json",
            # ),
        ],
    ),
    val=dict(yolo_data=["ISDD+SII.yaml"]),
)
model = YOLOWorld("yolov8x-worldv2-psconv2-wavepool.yaml")
# model = YOLOWorld("yolov8l-worldv2.pt")

# result = model.model.load_state_dict(torch.load("yolov8x-worldv2.pt")['model'].state_dict(), strict=False)

# print("="*80)
# print("模型加载结果详细解析:")
# print("="*80)

# # 获取预训练模型的状态字典
# pretrained_state_dict = torch.load("yolov8s-worldv2.pt")['model'].state_dict()
# current_state_dict = model.model.state_dict()

# print(f"预训练模型层数: {len(pretrained_state_dict)}")
# print(f"当前模型层数: {len(current_state_dict)}")

# # 分析加载结果
# missing_keys = result.missing_keys if hasattr(result, 'missing_keys') else []
# unexpected_keys = result.unexpected_keys if hasattr(result, 'unexpected_keys') else []

# print("\n" + "="*80)
# print("成功加载的层:")
# print("="*80)
# loaded_count = 0
# for key in pretrained_state_dict.keys():
#     if key in current_state_dict and key not in missing_keys:
#         print(f"✓ {key}")
#         loaded_count += 1

# print(f"\n成功加载层数: {loaded_count}")

# if missing_keys:
#     print("\n" + "="*80)
#     print("缺失的层 (当前模型中存在但预训练模型中不存在):")
#     print("="*80)
#     for key in missing_keys:
#         print(f"✗ {key}")
#     print(f"\n缺失层数: {len(missing_keys)}")

# if unexpected_keys:
#     print("\n" + "="*80)
#     print("未预期的层 (预训练模型中存在但当前模型中不存在):")
#     print("="*80)
#     for key in unexpected_keys:
#         print(f"! {key}")
#     print(f"\n未预期层数: {len(unexpected_keys)}")

# # 检查形状不匹配的层
# print("\n" + "="*80)
# print("形状匹配检查:")
# print("="*80)
# shape_mismatch_count = 0
# for key in pretrained_state_dict.keys():
#     if key in current_state_dict:
#         pretrained_shape = pretrained_state_dict[key].shape
#         current_shape = current_state_dict[key].shape
#         if pretrained_shape != current_shape:
#             print(f"⚠ {key}: 预训练{pretrained_shape} -> 当前{current_shape}")
#             shape_mismatch_count += 1

# if shape_mismatch_count == 0:
#     print("✓ 所有匹配的层形状都一致")
# else:
#     print(f"\n形状不匹配层数: {shape_mismatch_count}")

# print("\n" + "="*80)
# print("加载总结:")
# print("="*80)
# print(f"总成功率: {loaded_count}/{len(pretrained_state_dict)} ({loaded_count/len(pretrained_state_dict)*100:.1f}%)")

model.train(data=data, batch=8, epochs=150, trainer=WorldTrainerFromScratch, 
            project='runs/ISDD/new/Tra-trainval(isdd-noValText)+Te-Test(isdd)', name='xv2-pscov-pka7-3-wavepool', device='1')