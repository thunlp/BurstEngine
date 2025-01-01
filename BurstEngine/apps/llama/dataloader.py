import torch
import bmtrain as bmt
import json
import numpy as np

class DataLoader:
        def __init__(self, dataloader, args) -> None:

            self.input_length_sum = 0
            self.data_count = 0
            self.target_num = 0
            self.dataloader = iter(dataloader)
            self.args = args
            self.rank = bmt.config["local_rank"]
            self.tp_idx = 0
            self._idx = 0
            self._status = [0 for i in range(bmt.config["local_size"])]

        def write_shm_mem(self, data):
            data.pop('raw_data')
            d = {}
            rank = self.rank
            tp_idx = self.tp_idx
            with open(f"/dev/shm/BMT_{rank}_{tp_idx}_{self._status[rank]}.bin", "wb") as fb:
                for k in data:
                    if isinstance(data[k], np.ndarray):
                        bs = data[k].tobytes()
                        fb.write(bs)
                        d[k] = ["NUMPY", str(data[k].dtype), len(bs)] + list(data[k].shape)
                    else:
                        d[k] = data[k]
            with open(f"/dev/shm/BMT_{rank}_{tp_idx}_{self._status[rank]}.json", "w") as f:
                json.dump(d, f)
        
        def read_shm_mem(self, _idx):
            tp_idx = self.tp_idx
            with open(f"/dev/shm/BMT_{_idx}_{tp_idx}_{self._status[_idx]}.json", "r") as f:
                data = json.load(f)
            with open(f"/dev/shm/BMT_{_idx}_{tp_idx}_{self._status[_idx]}.bin", "rb") as fb:
                bs = fb.read()
                offset = 0
            for k in data:
                if isinstance(data[k], list) and len(data[k])>1 and data[k][0] == "NUMPY":
                    nw_offset = offset + data[k][2]
                    data[k] = np.frombuffer(bs[offset: nw_offset], dtype=data[k][1]).reshape(data[k][3:])
                    offset = nw_offset             
            self._status[_idx] = (self._status[_idx] + 1) % 2
            if _idx == self.rank:
                next_data = next(self.dataloader)
                self.write_shm_mem(next_data)
            return data

        def get_batch(self):
            if self._idx == 0:
                data = next(self.dataloader)
                self.write_shm_mem(data)
                bmt.synchronize()
            data = self.read_shm_mem(self._idx % bmt.config["local_size"])
            self._idx += 1
             
            return data

        def __iter__(self):
            while True:
                args = self.args
                data = self.get_batch()
                yield data
            # input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            # input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            # targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            # task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            # task_names = data["task_names"]
            # if args.flash == "cuda":
            #     cu_seqlens = torch.from_numpy(data["cu_seqlens"]).cuda().to(torch.int32)
            #     max_seqlen = data["max_seqlen"]
            #     position_ids = torch.from_numpy(data["position_ids"]).cuda().to(torch.int32)
            # else:
            #     input_context = torch.zeros_like(input_ids).cuda().bool()
            #     input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)
            # self.input_length_sum += input_length.float().mean()

        
