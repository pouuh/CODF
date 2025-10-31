#!/usr/bin/env python3
"""
Dask on BU SCC (GridEngineCluster) — Lightweight scheduler + multi-node workers
Author: Chishan Zhang
"""

from dask_jobqueue import SGECluster
from dask.distributed import Client, performance_report
import socket, os, time

import dask_jobqueue, dask
print(dask.__version__, dask_jobqueue.__version__)

# =======================
# 1. Scheduler configuration (主节点)
# =======================
cluster = SGECluster(
    cores=8,                     # 每个 worker 节点 8 核
    memory="128GB",               # 每个 worker 节点 内存
    queue="geo",
    project="modislc",
    resource_spec="h_rt=12:00:00,mem_free=100G",
    job_extra_directives=[
        "-l mem_per_core=16G",
        "-pe omp 8",
        "-V",
    ],
    walltime="12:00:00",
    # interface="ib0",              # 使用 InfiniBand 网络
    local_directory=os.environ.get("TMPDIR", "/projectnb/modislc/users/chishan/tmp"),
    log_directory="/projectnb/modislc/users/chishan/logs",
    env_extra=[
        "export OMP_NUM_THREADS=16",
        "export MKL_NUM_THREADS=16",
        "export OPENBLAS_NUM_THREADS=16",
        "export DASK_TEMPORARY_DIRECTORY=/projectnb/modislc/users/chishan/dask_tmp",
        "export GDAL_CACHEMAX=1024",
    ],
    # dashboard_address=":8787",
    # ---- ✅ 新版本语法 ----
    scheduler_options={"dashboard_address": ":8787"},
)

# =======================
# 2. Scale up workers
# =======================
cluster.scale(jobs=4)  # 启动 4 个 worker 节点 (共 64 核 / 400 GB)

client = Client(cluster)
print(f"\n🟢 Dask Dashboard : {cluster.dashboard_link}")
print(f"🟢 Scheduler Node : {socket.gethostname()}\n")

# 等待 workers 启动
for _ in range(60):
    n = len(client.scheduler_info()["workers"])
    if n >= 4:
        break
    time.sleep(5)
    print(f"Waiting for workers ({n}/4)...")
print("✅ Workers ready!")

# 打印所有 worker 节点
info = client.scheduler_info()["workers"]
print("Active workers:")
for w in info.keys():
    print("    ", w)

# =======================
# 3. Example workload
# =======================
import dask.array as da
x = da.random.random((40000, 40000), chunks=(2000, 2000))

with performance_report(filename="dask_report.html"):
    mean_val = x.mean().compute()
    print("Mean :", mean_val)

client.close()