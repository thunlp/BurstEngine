import torch

a = torch.cuda.Event()

s = torch.cuda.Stream() 

with torch.cuda.stream(s):
    c = torch.randn((10,10),dtype=torch.half,device="cuda")
    c *= 2
curr = torch.cuda.current_stream()
curr.wait_event(a)
print("event1 wait")
c = torch.randn((10,10),dtype=torch.half,device="cuda")
curr.wait_event(a)
print("event2 wait2")

