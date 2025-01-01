import torch
a = torch.randn(100, 256, device='cuda')
b = torch.randn(256, 100, device='cuda')    
b2 = torch.randn(256, 100, device='cuda')

s= torch.cuda.Stream(-1)
with torch.cuda.stream(s):
    c = torch.matmul(a, b)
c2 = torch.matmul(a, b2)

torch.cuda.current_stream().wait_stream(s)
