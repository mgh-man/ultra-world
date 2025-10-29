from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
import torch
import warnings
import os
import wandb
import traceback
from pathlib import Path
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
torch.use_deterministic_algorithms(True, warn_only=True)
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

def smart_transfer_weights():
    """智能权重迁移：只迁移兼容的层"""
    
    # 创建自定义模块（使用方案1的代码）
    # ... (此处省略模块创建代码)
    
    print("🚀 加载自定义模型...")
    model = YOLOWorld("/home/manfenglin/code/ultralytics/ultralytics/cfg/models/v8/yolov8x-worldv2-p2-mod.yaml")
    # model = YOLOWorld("yolov8x-worldv2.pt")
    print("📥 加载预训练权重...")
    pretrained_dict = torch.load("yolov8x-worldv2.pt")['model'].state_dict()
    model_dict = model.model.state_dict()
    
    # 只保留兼容的权重
    compatible_dict = {}
    incompatible_layers = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                compatible_dict[k] = v
                print(f"✓ 迁移: {k}")
            else:
                incompatible_layers.append(k)
                print(f"✗ 跳过: {k} - 形状不匹配 {v.shape} -> {model_dict[k].shape}")
        else:
            print(f"? 未找到: {k}")
    
    # 加载兼容的权重
    model.model.load_state_dict(compatible_dict, strict=False)
    
    print(f"📊 成功迁移: {len(compatible_dict)} / {len(pretrained_dict)} 层")
    print(f"📊 不兼容层数: {len(incompatible_layers)}")
    
    return model

# data = dict(
#     train=dict(
#         yolo_data=["ultralytics/cfg/datasets/365.yaml"],
#         # yolo_data=["ultralytics/cfg/datasets/minival.yaml"],
#         grounding_data=[
#             dict(
#                 img_path="/mnt/datasets/manfenglin/flickr30k/images",
#                 json_file="/mnt/code/manfenglin/ultralytics/datasets/flickr30k/final_flickr_separateGT_train.json",
#             ),
#             dict(
#                 img_path="/mnt/datasets/manfenglin/GQA/images",
#                 json_file="/mnt/code/manfenglin/ultralytics/datasets/GQA/final_mixed_train_no_coco.json",
#             ),
#         ],
#     ),
#     val=dict(yolo_data=["ultralytics/cfg/datasets/minival.yaml"]),
# )

current_dir = Path(__file__).parent.absolute()

data = dict(
    train=dict(
        yolo_data=[str(current_dir / "ultralytics/cfg/datasets/365.yaml")],  # 绝对路径
        grounding_data=[
            dict(
                img_path="/mnt/datasets/manfenglin/flickr30k/images",
                json_file="/mnt/code/manfenglin/ultralytics/datasets/flickr30k/final_flickr_separateGT_train.json",
            ),
            dict(
                img_path="/mnt/datasets/manfenglin/GQA/images",
                json_file="/mnt/code/manfenglin/ultralytics/datasets/GQA/final_mixed_train_no_coco.json",
            ),
        ],
    ),
    val=dict(yolo_data=[str(current_dir / "ultralytics/cfg/datasets/minival.yaml")]),  # 绝对路径
)
# model = YOLOWorld("/home/manfenglin/code/ultralytics/ultralytics/cfg/models/v8/yolov8x-worldv2-p2-mod.yaml")
# model = YOLOWorld("yolov8x-worldv2.pt")
# model = YOLOWorld("/mnt/code/manfenglin/ultralytics/runs/pretrain/xv2-device7-mixup-HAFB(2,3,4)-psconv-p2-pka7-3-5500-batch8-albumentions/weights/last.pt")

# model = smart_transfer_weights()

# print("🎯 开始训练...")
# model.train(data=data, batch=40, epochs=30, trainer=WorldTrainerFromScratch, 
#             project='runs/pretrain', name='usecoco-xv2-device7-mixup-HAFB(2,3,4)-psconv(1)-p2-pka7-3-5500-batch8-albumentions',device='4,5,6,7')
# model.train(data=data, batch=48, epochs=30, trainer=WorldTrainerFromScratch, resume="/mnt/code/manfenglin/ultralytics/runs/pretrain/xv2-device7-mixup-HAFB(2,3,4)-psconv-p2-pka7-3-5500-batch8-albumentions/weights/last.pt")

try:
    model = smart_transfer_weights()
    # model = YOLOWorld("/mnt/code/manfenglin/ultralytics/runs/pretrain/use(offiX)-xv2-device7-mixup-HAFB(2,3,4)-psconv(1)-p2-nopka-5500-batch8-albumentions/weights/best.pt")

    # 强制所有梯度为 contiguous，避免 DDP 警告
    # import torch
    # try:
    #     for p in model.model.parameters():
    #         if p.requires_grad:
    #             p.register_hook(lambda g: g.contiguous())
    # except Exception:
    #     pass

    print("🎯 开始训练...")
    model.train(
        data=data, 
        batch=32,  # 减少批次大小
        epochs=30, 
        trainer=WorldTrainerFromScratch, 
        project='runs/pretrain', 
        name='resume-test-val',
        device='4,5,6,7',
        workers=4,  # 减少worker数量
        patience=10,
        verbose=True,
    )
    
except Exception as e:
    print(f"❌ 训练失败: {e}")
    print("完整错误信息:")
    traceback.print_exc()