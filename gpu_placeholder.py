import torch

print("gpu_placeholder")
a6 = torch.rand(2500, 2500).cuda(6)
a0 = torch.rand(2500, 2500).cuda(0)
a1 = torch.rand(2500, 2500).cuda(1)
a2 = torch.rand(2500, 2500).cuda(2)
a3 = torch.rand(2500, 2500).cuda(3)
a4 = torch.rand(2500, 2500).cuda(4)
a5 = torch.rand(2500, 2500).cuda(5)
a7 = torch.rand(2500, 2500).cuda(7)
while True:
    b = a0 * a0
    b = a1 * a1
    b = a2 * a2
    b = a3 * a3
    b = a4 * a4
    b = a5 * a5
    b = a6 * a6
    b = a7 * a7
