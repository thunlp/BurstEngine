import pynvml
import pynvml

# 初始化NVML库
pynvml.nvmlInit()

# 获取GPU数量
device_count = pynvml.nvmlDeviceGetCount()

# 遍历所有GPU，检查利用率
for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        print(f"GPU {i} has 0% utilization.")

            # 关闭NVML库
pynvml.nvmlShutdown()
