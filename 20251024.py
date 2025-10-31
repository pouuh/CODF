from pystac_client import Client
import planetary_computer as pc
import odc.stac
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

lon, lat = -60.138576, -16.079505
bbox = [lon-0.02, lat-0.02, lon+0.02, lat+0.02]

# ---- 连接 STAC ----
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
collections = ["landsat-7-c2-l2", "landsat-8-c2-l2", "landsat-9-c2-l2"]

def load_year(year):
    items = list(
        catalog.search(
            collections=collections,
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 60}},
        ).get_items()
    )
    if not items:
        return None
    signed = [pc.sign(i) for i in items]
    ds = odc.stac.load(
        signed,
        bands=["red", "nir08"],
        bbox=bbox,
        resolution=30,
        groupby="solar_day",
        fail_on_missing=False,
        skip_broken_datasets=True,
    )
    ds = ds.rename({"nir08": "nir"})
    # 旱季 6–10 月
    dry = ds.sel(time=ds.time.dt.month.isin([6,7,8,9,10]))
    ndvi = (dry["nir"] - dry["red"]) / (dry["nir"] + dry["red"])
    ndvi_median = ndvi.median(dim="time")
    return float(ndvi_median.sel(x=lon, y=lat, method="nearest").compute())

years = np.arange(2016, 2021)
ndvi_vals = []
for y in years:
    v = load_year(y)
    ndvi_vals.append(v)
    print(f"{y}: {v}")

# ---- 绘图 ----
plt.figure(figsize=(7,4))
plt.plot(years, ndvi_vals, "ko-", lw=1.8)
plt.axvline(2018, color="blue", ls="--", label="2018")
plt.title("NDVI (dry-season median, 2016–2020)")
plt.xlabel("Year"); plt.ylabel("NDVI")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("ndvi_2016_2020_point.png", dpi=200)
plt.show()