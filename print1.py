from ultralytics import YOLO
import torch

# Load a.pt model without strict mode
model = YOLO("yolov8l.pt", task='detect')
model.model.load_state_dict(torch.load("yolov8l.pt")['model'].state_dict(), strict=False)

print(model)
print("\n" + "="*80)
print("模型层详细信息:")
print("="*80)

# 输出每一个层的名称和类型
for i, (name, module) in enumerate(model.model.named_modules()):
    if name:  # 跳过根模块（空名称）∫
        print(f"{i:3d}. {name:<40} -> {module.__class__.__name__}")

print("\n" + "="*80)
print("模型参数统计:")
print("="*80)

# 输出每个层的参数数量
total_params = 0
for name, module in model.model.named_modules():
    if name and hasattr(module, 'parameters'):
        layer_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if layer_params > 0:
            print(f"{name:<40} -> {layer_params:>10,} 参数")
            total_params += layer_params

print(f"\n总参数数量: {total_params:,}")

