import torch
from torch2trt import torch2trt

model = torch.load("new.pt").eval().cuda().half()

# create example data
x = torch.ones((1, 3, 480, 640)).cuda().half()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x],fp16_mode=True)
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
torch.save(model_trt.state_dict(), 'bisenet_fp16.pth')
