import torch
import time
import torch
def allocate_tensor(size_in_mib):
    num_elements = (size_in_mib * 1024 * 1024) // 4
    res = []
    for i in range(8):
        tensor = torch.empty(num_elements, dtype=torch.float32, device=f'cuda:{i}')
        res.append(tensor)
    return res

# Example usage
while True:
    try:
        mib = input("How many MiB to allocate? ")
        tensor = allocate_tensor(int(mib))
    except KeyboardInterrupt:
        break


