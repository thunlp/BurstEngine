import torch
import torch.distributed as dist


group = dist.init_process_group(backend='nccl', init_method='env://')
print('Rank:', dist.get_rank())
print('World size:', dist.get_world_size())
torch.cuda.set_device(dist.get_rank())
t = torch.ones(100).cuda()
res = torch.zeros(dist.get_world_size(), 100).cuda()
dist.all_gather_into_tensor(res, t, group=group)
print(res)

