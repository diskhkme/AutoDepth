import torch

A = torch.range(1,12).view(2,3,2) # 2 x 3 x 2 tensor
Index = torch.tensor([[0,1,0],[1,0,1]]) # 2 x 3 tensor
Index.unsqueeze(2)

A.gather(2,Index.unsqueeze(2))