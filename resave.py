import torch

model = torch.load("bisenet.pth")

torch.save(model.module,"new.pt")
