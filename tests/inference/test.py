import torch

a = 2
a_ = torch.tensor(a).repeat(3)

print(a_.shape)