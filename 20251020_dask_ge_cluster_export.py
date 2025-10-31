#!/usr/bin/env python3
"""
glance_mb_sge_final.py

GLANCE vs MapBiomas Forest Comparison (ODC + Dask on BU SCC via SGECluster)
- Auto-detects a usable IPv4 and passes it to scheduler (robust across SCC nodes)
- Starts SGECluster with multi-node workers (jobs x cores)
- Uses per-process single-thread math libs to avoid oversubscription
- Loads MapBiomas COG and GLANCE via odc.stac.load aligned to the MapBiomas geobox
- Computes TN/FP/FN/TP via dask boolean reductions (scales to large AOI)
- Writes a Dask performance report to /usr2/postdoc/chishan/logs/
"""

import os
import time
import socket
import xarray as xr
import rioxarray
import numpy as np
import dask.array as da
from odc.stac import load as odc_load

def pick_ipv4_address():
    """Return (iface_name, ipv4) or (None, None) if not found."""
    import socket
    try:
        import psutil
    except Exception:
        psutil = None

    preferred = ("ib0", "p3p1", "p3p2", "em1", "em2", "eno1", "ens1f0", "eth0")
    if psutil:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        for name in preferred:
            if name in stats and stats[name].isup and name in addrs:
                for a in addrs[name]:
                    if a.family == socket.AF_INET and not a.address.startswith("127."):
                        return name, a.address
    # fallback: derive outbound IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return None, ip
    except Exception:
        return None, None


def start_cluster(jobs=4, cores=8, memory="100GB"):
    """
    Try Dask auto-bind first; if that fails, detect IPv4 and retry with explicit host.
    """
    from dask_jobqueue import SGECluster
    from dask.distributed import Client
    import time

    # common env_extra (single-thread math libs etc.)
    env_extra = [
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_MAX_THREADS=1",
        "export DASK_TEMPORARY_DIRECTORY=/projectnb/modislc/users/chishan/dask_tmp",
        "export GDAL_CACHEMAX=256",
    ]

    # 1) First try: let Dask auto-pick (no 'host' in scheduler_options)
    try:
        sched_opts = {"dashboard_address": ":8787"}
        cluster = SGECluster(
            project="modislc",
            queue="geo",
            cores=cores,
            processes=cores,
            memory=memory,
            walltime="12:00:00",
            resource_spec=f"h_rt=12:00:00",
            job_extra_directives=[f"-pe omp {cores}", "-l mem_per_core=12G", "-V"],
            local_directory=os.environ.get("TMPDIR", "/projectnb/modislc/users/chishan/tmp"),
            log_directory="/projectnb/modislc/users/chishan/logs",
            env_extra=env_extra,
            scheduler_options=sched_opts,
        )
        cluster.scale(jobs=jobs)
        client = Client(cluster)
        print("✅ Cluster started with Dask auto-binding (no explicit host).")
    except Exception as e_auto:
        # 2) If auto-bind fails (common on nodes with strange NIC config), try explicit IPv4
        print("⚠️ Dask auto-bind failed: will attempt explicit IPv4 bind. Error:", e_auto)

    # common post-start prints & waiting
    import socket as _s
    print(f"\n🟢 Dask Dashboard : {cluster.dashboard_link}")
    print(f"🟢 Scheduler Node : {_s.gethostname()}\n")

    for i in range(60):
        n = len(client.scheduler_info().get("workers", {}))
        if n >= jobs:
            break
        time.sleep(5)
        print(f"   {i*5:>3d}s: {n}/{jobs} workers connected")
    print("✅ Workers connected!")
    for w in client.scheduler_info()["workers"].keys():
        print("   ", w)

    return client

# -----------------------
# GLANCE loader (local STAC)
# -----------------------
def load_glance_local(stac_path: str, year: int, geobox, chunk_size=2048):
    """
    Load GLANCE items for a given year and align to the provided geobox.
    Returns a single xarray.DataArray (squeezed time dim).
    """
    import pystac
    cat = pystac.Catalog.from_file(stac_path)
    items = [it for it in cat.get_items(recursive=True)
             if getattr(it, "datetime", None) and it.datetime.year == year]
    if not items:
        raise RuntimeError(f"No GLANCE items found for year {year}")

    ds = odc_load(
        items,
        geobox=geobox,
        chunks={"x": chunk_size, "y": chunk_size},
        resampling="nearest",
        fail_on_error=False,
    )

    # choose first datavar unless you know exact name
    var = next(iter(ds.data_vars))
    gl = ds[var].squeeze(drop=True)

    # normalize dim names
    rename = {}
    if "latitude" in gl.dims:
        rename["latitude"] = "y"
    if "longitude" in gl.dims:
        rename["longitude"] = "x"
    if rename:
        gl = gl.rename(rename)

    return gl

# -----------------------
# Main workflow
# -----------------------
def main():
    year = 2018
    chunk_size = 2048 #for both x and y

    # cluster sizing: override by environment if desired
    jobs   = int(os.environ.get("DASK_JOBS", "4"))          # number of worker jobs (nodes)
    cores  = int(os.environ.get("DASK_WORKER_CORES", "8")) # cores per worker job
    memory = os.environ.get("DASK_WORKER_MEM", "100GB")   # per-worker job memory

    env_extra = [
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_MAX_THREADS=1",
        "export DASK_TEMPORARY_DIRECTORY=/projectnb/modislc/users/chishan/dask_tmp",
        "export GDAL_CACHEMAX=512",  # 可以适当调高
        "export CPL_VSICURL_CACHE=TRUE",
        "export CPL_VSICURL_CACHE_SIZE=33554432",
    ]
    from dask_jobqueue import SGECluster
    from dask.distributed import Client

    cluster = SGECluster(
        project="modislc",
        queue="geo",
        cores=cores,                 # 8
        processes=1,                 # <= 1 worker per job (give all memory to single worker)
        memory=memory,               # worker gets the whole job memory
        walltime="12:00:00",
        resource_spec="h_rt=12:00:00",
        job_extra_directives=[
            f"-pe omp {cores}",
            "-l mem_per_core=12G",   # keep this consistent with job config
            "-V",
        ],
        local_directory=os.environ.get("TMPDIR", "/projectnb/modislc/users/chishan/tmp"),
        log_directory="/projectnb/modislc/users/chishan/logs",
        env_extra=env_extra,
        scheduler_options={"dashboard_address": ":8787"},
    )

    cluster.scale(jobs=jobs)
    client = Client(cluster)
    # start Dask cluster on SCC
    # client = start_cluster(jobs=jobs, cores=cores, memory=memory)

    # paths
    stac_path = "/projectnb/modislc/users/chishan/stac_glance_SA_fixed_m/catalog.json"
    mb_path   = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"

    # load MapBiomas COG (chunks aligne)
    mb = rioxarray.open_rasterio(mb_path, chunks={"y": chunk_size, "x": chunk_size}).squeeze(drop=True)
    if "latitude" in mb.dims or "longitude" in mb.dims:
        mb = mb.rename({"latitude": "y", "longitude": "x"})
    mb_nodata = mb.rio.nodata if mb.rio.nodata is not None else 0
    print(f"✓ MapBiomas loaded: shape={mb.shape}, dims={mb.dims}, CRS={mb.rio.crs}, nodata={mb_nodata}")
    print(f"  Geobox: width={mb.odc.geobox.width}, height={mb.odc.geobox.height}, res={mb.odc.geobox.resolution}")

    # load GLANCE aligned to MapBiomas grid
    gl = load_glance_local(stac_path, year, geobox=mb.odc.geobox, chunk_size=chunk_size)
    gl_nodata = gl.rio.nodata if hasattr(gl, "rio") and (gl.rio.nodata is not None) else 255
    print(f"✓ GLANCE loaded: shape={gl.shape}, dims={gl.dims}, nodata={gl_nodata}")

    # sanity check dims/shapes
    if gl.dims != mb.dims or gl.shape != mb.shape:
        raise AssertionError(f"Dimension/shape mismatch: GLANCE {gl.dims}/{gl.shape} vs MB {mb.dims}/{mb.shape}")
    print("✅ Dimensions aligned.")

    # reclassify to binary forest maps while preserving nodata
    # MapBiomas forest classes: [1,2,9]; GLANCE forest class: 5 (adjust if needed)
    mb_bin = xr.where(mb != mb_nodata, mb.isin([1,2,9]).astype("uint8"), np.uint8(255))
    gl_bin = xr.where(gl != gl_nodata, (gl == 5).astype("uint8"), np.uint8(255))

    # ensure consistent chunking
    mb_bin = mb_bin.chunk({"y": chunk_size, "x": chunk_size})
    gl_bin = gl_bin.chunk(mb_bin.chunksizes)

    # valid mask (both not nodata)
    valid = (mb_bin != 255) & (gl_bin != 255)

    code = (mb_bin.astype("int8") * 2 + gl_bin.astype("int8")).where(valid, other=4).astype("int8")
    # dask_code = code.data.ravel()  # 1-D dask array

    # performance report path
    job_id = os.environ.get("JOB_ID", os.environ.get("SGE_TASK_ID", "manual"))
    report_path = f"/usr2/postdoc/chishan/logs/dask_report_{job_id}.html"

    from dask.distributed import performance_report
    from odc.geo.xr import write_cog

    # with performance_report(filename=report_path):
        # hist_da = da.bincount(dask_code, minlength=5)  # dask array (length >=5)
        # hist = hist_da.compute()  # now a numpy array
    code = code.astype("uint8").persist()
    # code = code.compute()
    from dask.distributed import wait
    wait(code)

    out_path = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/glance_mb_overlap_{year}.tif"
    write_cog(
        code,
        out_path,
        nodata=4,
        overwrite=True,
        blocksize=512,
        compress="deflate",
        overview_levels=(2, 4, 8, 16, 32, 64),
        overview_resampling="nearest",
        use_windowed_writes=True
    )
    
    print(f"✓ Parallel COG export: {out_path}")


    # # robust unpacking (ensure length >= 5)
    # if hist.size < 5:
    #     # pad with zeros if some bins are missing
    #     hist = np.pad(hist, (0, 5 - hist.size), constant_values=0)
        
    # tn = int(hist[0])
    # fp = int(hist[1])
    # fn = int(hist[2])
    # tp = int(hist[3])
    # nodata_count = int(hist[4])  # optional

    # # metrics
    # precision = tp/(tp+fp) if (tp+fp) else 0.0
    # recall    = tp/(tp+fn) if (tp+fn) else 0.0
    # f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0

    # print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    # print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    # print("Dask report:", report_path)

    # cleanup
    client.close()
    print("✓ Dask cluster closed")

if __name__ == "__main__":
    # ensure single-threaded BLAS/OpenMP inside processes
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    main()