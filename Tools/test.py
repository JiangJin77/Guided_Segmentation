import torch
print(torch.cuda.is_available()) # 输出 True 表示 GPU 可用
print(torch.cuda.get_device_name(0)) # 输出 GPU 名称