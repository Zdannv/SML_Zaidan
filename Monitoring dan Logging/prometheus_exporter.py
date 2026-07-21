
from prometheus_client import Gauge, start_http_server
import psutil,time

cpu=Gauge("cpu_usage_percent","CPU")
mem=Gauge("memory_usage_percent","Memory")

if __name__=="__main__":
    start_http_server(8001)
    while True:
        cpu.set(psutil.cpu_percent())
        mem.set(psutil.virtual_memory().percent)
        time.sleep(5)
