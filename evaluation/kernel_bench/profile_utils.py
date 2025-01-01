import torch

def get_summary(profiler):
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
    )
    if torch.distributed.get_rank() == 0:
        print(prof_lines)
    prof_lines = prof_lines.split("\n")
    kernel_names = ["flash", "SendRecv"]
    kernel_times = {kernel: [] for kernel in kernel_names}
    units = {"ms": 1, "us": 1e3}
    for kernel_name in kernel_names:
        for line in prof_lines:
            if kernel_name in line:
                ratio = line.split()[-4]
                time_str = line.split()[-3]
                for unit, scale in units.items():
                    if unit in time_str:
                        _t = float(time_str.replace(unit, "")) / scale
                        _r = float(ratio.replace("%", "")) / 100
                        kernel_times[kernel_name].append((_t, _r))
                        break
    # kernel_times = {kernel: np.sum(times[0]) for kernel, times in kernel_times.items()}
    summary = dict.fromkeys(kernel_names, 0)
    for _key in kernel_times:
        for _r in kernel_times[_key]:
            summary[_key] += _r[1]
    return summary, kernel_times
