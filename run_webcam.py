from main import detect
import torch

torch.manual_seed(1234)

net  = torch.load('./model.pt')
detect(net)