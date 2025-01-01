from .. import nccl
from .shape import SHAPES
from ..global_var import config, world_size
from ..utils import print_rank
from .utils import format_size
import torch
def send_recv(send_recv_type="ring"):
    current_stream = torch.cuda.current_stream()
    if send_recv_type == "ring":
        src_rank = [i for i in range(config['world_size']) if i %2 == 0]
        des_rank = [i for i in range(config['world_size']) if i %2 == 1]
    else:
        src_rank = [i for i in range(config['world_size']) if i < config['world_size'] //2 ]
        des_rank = [i for i in range(config['world_size']) if i >= config['world_size'] //2 ]
    for i in range(5):
        for shape in [SHAPES[-1]]:
            send_size = shape

            send_buffer = torch.empty( send_size // 2, dtype=torch.half, device="cuda" )
            recv_buffer = torch.empty( send_size // 2, dtype=torch.half, device="cuda" )
            send_buffer2 = torch.empty( send_size // 2, dtype=torch.half, device="cuda" )
            recv_buffer2 = torch.empty( send_size // 2, dtype=torch.half, device="cuda" )
            
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            current_stream.record_event(start_evt)
            des = des_rank[src_rank.index(config['rank'])] if config['rank'] in src_rank else None
            src = src_rank[des_rank.index(config['rank'])] if config['rank'] in des_rank else None
            if send_recv_type == "ring":
                wsz = world_size()
                src = (config['rank'] - 1 + wsz) % wsz
                des = (config['rank'] + 1) % wsz
                stream = torch.cuda.current_stream()
                stream2  = torch.cuda.Stream()
                # if config['rank'] ==0:
                nccl.groupStart()
                nccl.send(send_buffer.storage(), des, config['comm'])
                # nccl.send(send_buffer2.storage(), src, config['comm'])
                # nccl.recv(recv_buffer2.storage(), src, config['comm'])
                nccl.recv(recv_buffer.storage(), src, config['comm'])
                nccl.groupEnd()
                # else:
                #     nccl.groupStart()
                #     nccl.recv(recv_buffer.storage(), src, config['comm'])
                #     nccl.send(send_buffer2.storage(), src, config['comm'])
                #     nccl.groupEnd()
                # with torch.cuda.stream(stream2):
                #     nccl.groupStart()
                #     if config['rank'] % 2 == 0:
                #         nccl.send(recv_buffer2.storage(), src, config['comm2'])
                #         nccl.recv(send_buffer2.storage(), des, config['comm2'])
                #     else:
                #         nccl.recv(send_buffer2.storage(), des, config['comm2'])
                #         nccl.send(recv_buffer2.storage(), src, config['comm2'])
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.current_stream().wait_stream(stream2)
            else:
                inter_rank=nccl.commRank(config['local_idx_comm'])
                inter_size=4
                inter_des = (inter_rank + 1) % inter_size
                inter_src = (inter_rank - 1 + inter_size) % inter_size
                nccl.send(send_buffer.storage(), inter_src, config['local_idx_comm'])
                nccl.recv(recv_buffer.storage(), inter_des, config['local_idx_comm'])
                # nccl.send(send_buffer.storage(), inter_des, config['local_idx_comm'])
                # nccl.recv(recv_buffer.storage(), inter_src, config['local_idx_comm'])
            current_stream.record_event(end_evt)
            current_stream.synchronize()
            time_usage = start_evt.elapsed_time(end_evt)

            bw = shape / 1024 / 1024 / 1024 * 1000 / time_usage
            print_rank("Send Recv:\tsize {}\ttime: {:4.3f}\tbw: {:2.6f} GB/s".format(format_size(send_size), time_usage, bw))

