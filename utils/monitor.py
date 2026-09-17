# cuda 12.2 -> torch2.6.0
import datetime
import subprocess
import re
import time

TOTAL_LEN = 20
def get_gpu_usage():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            encoding="utf-8"
        )

        gpus = []
        for line in result.strip().split("\n"):
            gpu_util, mem_used, mem_total = map(int, re.findall(r"\d+", line))
            mem_util = mem_used / mem_total
            gpus.append({
                "gpu_utilization": gpu_util / 100,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_utilization": mem_util
            })

        return gpus

    except FileNotFoundError:
        return "nvidia-smi 未找到，请确保已安装 NVIDIA 驱动"
    except Exception as e:
        return f"获取 GPU 信息失败: {e}"

if __name__ == "__main__":
    while True:
        gpu_info = get_gpu_usage()
        if isinstance(gpu_info, str):
            print(gpu_info)
            break
        else:
            for idx, gpu in enumerate(gpu_info):
                if gpu['gpu_utilization'] < 0.01 and gpu['memory_utilization'] < 0.01:
                    continue
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GPU {idx}: {'#' * int(TOTAL_LEN*gpu['gpu_utilization']) + '.' * (TOTAL_LEN - int(TOTAL_LEN*gpu['gpu_utilization']))}, {gpu['gpu_utilization']*100:.2f}%,  Mem: {'#' * int(TOTAL_LEN*gpu['memory_utilization']) + '.' * (TOTAL_LEN - int(TOTAL_LEN*gpu['memory_utilization']))}, {gpu['memory_utilization']*100:.2f}%")
        time.sleep(1)
