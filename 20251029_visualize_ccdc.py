#!/usr/bin/env python3
"""
CCDC point inspector and plotting helper.

Workflow:
  1. pip install earthengine-api pandas matplotlib
  2. ee.Authenticate()  # run once to store your credentials
  3. Adjust the CONFIG values below (asset path, point, bands, etc.)
  4. python ccdc_point_viewer.py
"""

import math
import calendar
import datetime as dt
from typing import Dict, List, Tuple, Optional

import ee
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# ------------------------- USER CONFIGURATION -------------------------
# ----------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Earth Engine asset path that stores the CCDC output
    "ASSET_PATH": "projects/CCDC/v3",
    # One of: "Image", "ImageCollection"
    "PATH_TYPE": "ImageCollection",
    # Optional prefix for filtering image collections by system:index
    "FILTER_PREFIX": "",
    # Number of CCDC segments expected in the results (use 10 for full output)
    "N_SEGMENTS": 10,
    # Bands you want to inspect; synthetic prediction will use TARGET_BAND
    "BANDS": ["SWIR1", "NIR", "RED"],
    "TARGET_BAND": "SWIR1",
    # Longitude / latitude of the pixel of interest
    "POINT_LON": -60.7440,
    "POINT_LAT": -8.8886,
    # Time window for plotting (ISO-8601); converted to CCDC units automatically
    "ANALYSIS_START": "2000-01-01",
    "ANALYSIS_END": "2020-01-01",
    # Months between synthetic evaluations (e.g., 3 = quarterly)
    "STEP_MONTHS": 3,
    # Sampling scale (meters). 30 for Landsat-based CCDC.
    "SAMPLE_SCALE": 30,
    # Optional: Earth Engine project. Leave None to use default credentials.
    "EE_PROJECT": None,
}
# ----------------------------------------------------------------------
# ------------------------- UTILITY HELPERS ----------------------------
# ----------------------------------------------------------------------

GLOBAL_SUBCOEFS = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]
GLOBAL_EXTRA_COEFS = ["PHASE", "AMPLITUDE", "PHASE2", "AMPLITUDE2", "PHASE3", "AMPLITUDE3"]
GLOBAL_SEGS = [f"S{i}" for i in range(1, 11)]

def datetime_to_fractional_year(timestamp: dt.datetime) -> float:
    start = dt.datetime(timestamp.year, 1, 1, tzinfo=timestamp.tzinfo)
    next_year = dt.datetime(timestamp.year + 1, 1, 1, tzinfo=timestamp.tzinfo)
    return timestamp.year + (timestamp - start).total_seconds() / (next_year - start).total_seconds()

def fractional_year_to_datetime(value: float) -> dt.datetime:
    year = int(math.floor(value))
    remainder = value - year
    start = dt.datetime(year, 1, 1)
    next_year = dt.datetime(year + 1, 1, 1)
    return start + dt.timedelta(seconds=remainder * (next_year - start).total_seconds())

def convert_datetime_to_ccdc_units(timestamp: dt.datetime, date_format: int) -> float:
    if date_format == 1:  # fractional year
        return datetime_to_fractional_year(timestamp)
    if date_format == 0:  # YYYYDDD (Julian day)
        return timestamp.year * 1000 + timestamp.timetuple().tm_yday
    if date_format == 2:  # Unix time (ms)
        return int(timestamp.timestamp() * 1000.0)
    raise ValueError(f"Unsupported dateFormat code: {date_format}")

def decode_ccdc_date(value: Optional[float], date_format: int) -> Optional[dt.datetime]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # 0 and negatives are padding -> no valid date
    if v <= 0:
        return None

    if date_format == 1:  # fractional year like 2010.5
        if v < 1500 or v > 3000:
            return None
        return fractional_year_to_datetime(v)

    if date_format == 0:  # YYYYDDD
        intval = int(round(v))
        year = intval // 1000
        doy = intval % 1000
        if year < 1500 or year > 3000 or doy < 1 or doy > 366:
            return None
        return dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)

    if date_format == 2:  # Unix ms
        if v <= 0:
            return None
        return dt.datetime.utcfromtimestamp(v / 1000.0)

    return None

    
    # if value is None:
    #     return None
    # if date_format == 1:
    #     return fractional_year_to_datetime(float(value))
    # if date_format == 0:
    #     intval = int(round(value))
    #     year = intval // 1000
    #     doy = intval % 1000
    #     return dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)
    # if date_format == 2:
    #     return dt.datetime.utcfromtimestamp(float(value) / 1000.0)
    # raise ValueError(f"Unsupported dateFormat code: {date_format}")

def add_months(timestamp: dt.datetime, months: int) -> dt.datetime:
    month = timestamp.month - 1 + months
    year = timestamp.year + month // 12
    month = month % 12 + 1
    day = min(timestamp.day, calendar.monthrange(year, month)[1])
    return dt.datetime(year, month, day, hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

def generate_ccdc_date_series(start_iso: str, end_iso: str, date_format: int, step_months: int) -> Tuple[List[float], List[dt.datetime]]:
    start = dt.datetime.fromisoformat(start_iso)
    end = dt.datetime.fromisoformat(end_iso)
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current)
        current = add_months(current, step_months)
    if timestamps[-1] != end:
        timestamps.append(end)
    unique = sorted(set(timestamps))
    return [convert_datetime_to_ccdc_units(t, date_format) for t in unique], unique

def stack_images(images: List[ee.Image]) -> ee.Image:
    if not images:
        raise ValueError("Cannot stack an empty list of images.")
    result = ee.Image(images[0])
    for img in images[1:]:
        result = result.addBands(ee.Image(img))
    return result

# ----------------------------------------------------------------------
# --------------------- CCDC UTILS (PYTHON PORT) -----------------------
# ----------------------------------------------------------------------

def build_segment_tag(n_segments: int) -> ee.List:
    return ee.List.sequence(1, n_segments).map(lambda i: ee.String("S").cat(ee.Number(i).format("%d")))

def build_magnitude(fit: ee.Image, n_segments: int, band_list: List[str]) -> ee.Image:
    segment_tag = build_segment_tag(n_segments)
    zeros = ee.Image(ee.Array(ee.List.repeat(0, n_segments)))
    images = []
    for band in band_list:
        band_const = ee.String(band)
        mag_img = (
            fit.select(band_const.cat("_magnitude"))
            .arrayCat(zeros, 0)
            .float()
            .arraySlice(0, 0, n_segments)
        )
        tags = segment_tag.map(lambda s: ee.String(s).cat("_").cat(band_const).cat("_MAG"))
        images.append(mag_img.arrayFlatten([ee.List(tags)]))
    return stack_images(images)

def build_rmse(fit: ee.Image, n_segments: int, band_list: List[str]) -> ee.Image:
    segment_tag = build_segment_tag(n_segments)
    zeros = ee.Image(ee.Array(ee.List.repeat(0, n_segments)))
    images = []
    for band in band_list:
        band_const = ee.String(band)
        rmse_img = (
            fit.select(band_const.cat("_rmse"))
            .arrayCat(zeros, 0)
            .float()
            .arraySlice(0, 0, n_segments)
        )
        tags = segment_tag.map(lambda s: ee.String(s).cat("_").cat(band_const).cat("_RMSE"))
        images.append(rmse_img.arrayFlatten([ee.List(tags)]))
    return stack_images(images)

def build_coefs(fit: ee.Image, n_segments: int, band_list: List[str]) -> ee.Image:
    segment_tag = build_segment_tag(n_segments)
    harmonic_tag = ee.List(GLOBAL_SUBCOEFS)
    zeros = ee.Image(ee.Array([ee.List.repeat(0, harmonic_tag.length())])).arrayRepeat(0, n_segments)
    images = []
    for band in band_list:
        band_const = ee.String(band)
        coef_img = (
            fit.select(band_const.cat("_coefs"))
            .arrayCat(zeros, 0)
            .float()
            .arraySlice(0, 0, n_segments)
        )
        tags = segment_tag.map(lambda s: ee.String(s).cat("_").cat(band_const).cat("_coef"))
        images.append(coef_img.arrayFlatten([ee.List(tags), harmonic_tag]))
    return stack_images(images)

def build_start_end_break_prob(fit: ee.Image, n_segments: int, tag: str) -> ee.Image:
    segment_tag = build_segment_tag(n_segments).map(lambda s: ee.String(s).cat("_" + tag))
    zeros = ee.Image(ee.Array(ee.List.repeat(0, n_segments)))
    values = (
        fit.select(tag)
        .arrayCat(zeros, 0)
        .float()
        .arraySlice(0, 0, n_segments)
    )
    return values.arrayFlatten([ee.List(segment_tag)])

def build_ccd_image(fit: ee.Image, n_segments: int, band_list: List[str]) -> ee.Image:
    image = build_coefs(fit, n_segments, band_list)
    image = image.addBands(build_rmse(fit, n_segments, band_list))
    image = image.addBands(build_magnitude(fit, n_segments, band_list))
    image = image.addBands(build_start_end_break_prob(fit, n_segments, "tStart"))
    image = image.addBands(build_start_end_break_prob(fit, n_segments, "tEnd"))
    image = image.addBands(build_start_end_break_prob(fit, n_segments, "tBreak"))
    image = image.addBands(build_start_end_break_prob(fit, n_segments, "changeProb"))
    image = image.addBands(build_start_end_break_prob(fit, n_segments, "numObs"))
    return image

def filter_coefs(ccd_results: ee.Image, date: float, band: str, coef: str, seg_names: List[str], behavior: str) -> ee.Image:
    seg_names_ee = ee.List(seg_names)
    start_bands = ccd_results.select(".*_tStart").rename(seg_names_ee)
    end_bands = ccd_results.select(".*_tEnd").rename(seg_names_ee)
    sel_str = ".*{}.*{}".format(band, coef)
    coef_bands = ccd_results.select(sel_str)
    date_num = ee.Number(date)

    if behavior == "normal":
        segment_match = start_bands.lte(date_num).And(end_bands.gte(date_num))
        out = coef_bands.updateMask(segment_match).reduce(ee.Reducer.firstNonNull())
    elif behavior == "after":
        segment_match = end_bands.gt(date_num)
        out = coef_bands.updateMask(segment_match).reduce(ee.Reducer.firstNonNull())
    else:  # before
        segment_match = start_bands.selfMask().lt(date_num).selfMask()
        out = coef_bands.updateMask(segment_match).reduce(ee.Reducer.lastNonNull())
    return out

def normalize_intercept(intercept: ee.Image, start: ee.Image, end: ee.Image, slope: ee.Image) -> ee.Image:
    middle = ee.Image(start).add(end).divide(2)
    slope_term = ee.Image(slope).multiply(middle)
    return ee.Image(intercept).add(slope_term)

def get_coef(ccd_results: ee.Image, date: float, band_list: List[str], coef: str, seg_names: List[str], behavior: str) -> ee.Image:
    images = []
    for band in band_list:
        result = filter_coefs(ccd_results, date, band, coef, seg_names, behavior)
        images.append(result.rename("{}_{}".format(band, coef)))
    return stack_images(images)

def apply_norm(band_coefs: ee.Image, seg_start: ee.Image, seg_end: ee.Image) -> ee.Image:
    intercepts = band_coefs.select(".*INTP")
    slopes = band_coefs.select(".*SLP")
    normalized = normalize_intercept(intercepts, seg_start, seg_end, slopes)
    return band_coefs.addBands(normalized, None, True)

def get_multi_coefs(ccd_results: ee.Image, date: float, band_list: List[str], coef_list: List[str], cond: bool, seg_names: List[str], behavior: str) -> ee.Image:
    if behavior == "auto":
        after_imgs = [get_coef(ccd_results, date, band_list, coef, seg_names, "after") for coef in coef_list]
        before_imgs = [get_coef(ccd_results, date, band_list, coef, seg_names, "before") for coef in coef_list]
        coefs_after = stack_images(after_imgs)
        coefs_before = stack_images(before_imgs)

        seg_start_after = filter_coefs(ccd_results, date, "", "tStart", seg_names, "after")
        seg_end_after = filter_coefs(ccd_results, date, "", "tEnd", seg_names, "after")
        norm_after = apply_norm(coefs_after, seg_start_after, seg_end_after)

        seg_start_before = filter_coefs(ccd_results, date, "", "tStart", seg_names, "before")
        seg_end_before = filter_coefs(ccd_results, date, "", "tEnd", seg_names, "before")
        norm_before = apply_norm(coefs_before, seg_start_before, seg_end_before)

        out_regular = ee.ImageCollection.fromImages([coefs_before, coefs_after]).mosaic()
        out_normalized = ee.ImageCollection.fromImages([norm_before, norm_after]).mosaic()
        return ee.Image(ee.Algorithms.If(cond, out_normalized, out_regular))

    images = [get_coef(ccd_results, date, band_list, coef, seg_names, behavior) for coef in coef_list]
    coefs = stack_images(images)
    seg_start = filter_coefs(ccd_results, date, "", "tStart", seg_names, behavior)
    seg_end = filter_coefs(ccd_results, date, "", "tEnd", seg_names, behavior)
    norm = apply_norm(coefs, seg_start, seg_end)
    return ee.Image(ee.Algorithms.If(cond, norm, coefs))

def get_changes(ccd_results: ee.Image, start_date: float, end_date: float, seg_names: List[str]) -> ee.Image:
    seg_names_ee = ee.List(seg_names)
    break_bands = ccd_results.select(".*_tBreak").rename(seg_names_ee)
    return break_bands.gte(start_date).And(break_bands.lt(end_date))

def filter_mag(ccd_results: ee.Image, start_date: float, end_date: float, band: str, seg_names: List[str]) -> ee.Image:
    seg_mask = get_changes(ccd_results, start_date, end_date, seg_names)
    sel_str = ".*{}.*MAG".format(band)
    feat_bands = ccd_results.select(sel_str)
    filtered_mag = feat_bands.updateMask(seg_mask)
    filtered_abs = filtered_mag.abs()
    max_abs = filtered_abs.reduce(ee.Reducer.max())
    matched_mag_mask = filtered_abs.eq(max_abs)
    selected_mag = filtered_mag.updateMask(matched_mag_mask).reduce(ee.Reducer.firstNonNull())
    filtered_tbreak = (
        ccd_results.select(".*tBreak")
        .updateMask(matched_mag_mask)
        .reduce(ee.Reducer.firstNonNull())
    )
    num_tbreak = (
        ccd_results.select(".*tBreak")
        .updateMask(seg_mask)
        .reduce(ee.Reducer.count())
    )
    return selected_mag.addBands([filtered_tbreak, num_tbreak]).rename(["MAG", "tBreak", "numTbreak"])

def new_phase_amplitude(img: ee.Image, sin_expr: str, cos_expr: str) -> ee.Image:
    sin = img.select(sin_expr)
    cos = img.select(cos_expr)
    phase = sin.atan2(cos).unitScale(-math.pi, math.pi).multiply(365)
    amplitude = sin.hypot(cos)
    phase_names = phase.bandNames().map(lambda x: ee.String(x).replace("_SIN", "_PHASE"))
    amplitude_names = amplitude.bandNames().map(lambda x: ee.String(x).replace("_SIN", "_AMPLITUDE"))
    return phase.rename(phase_names).addBands(amplitude.rename(amplitude_names))

def get_synthetic_for_year(image: ee.Image, date: float, date_format: int, band: str, segs: List[str]) -> ee.Image:
    omegas = [
        2.0 * math.pi / 365.25,
        2.0 * math.pi,
        2.0 * math.pi / (1000.0 * 60.0 * 60.0 * 24.0 * 365.25),
    ]
    omega = ee.Number(omegas[date_format])
    date_num = ee.Number(date)
    factors = ee.Image.constant([
        1.0,
        date_num,
        date_num.multiply(omega).cos(),
        date_num.multiply(omega).sin(),
        date_num.multiply(omega.multiply(2)).cos(),
        date_num.multiply(omega.multiply(2)).sin(),
        date_num.multiply(omega.multiply(3)).cos(),
        date_num.multiply(omega.multiply(3)).sin(),
    ]).float()
    coef_img = get_multi_coefs(image, date, [band], GLOBAL_SUBCOEFS, False, segs, "auto").float()
    return factors.multiply(coef_img).reduce('sum').rename(band)

def get_multi_synthetic(image: ee.Image, date: float, date_format: int, band_list: List[str], segs: List[str]) -> ee.Image:
    images = [get_synthetic_for_year(image, date, date_format, band, segs) for band in band_list]
    return stack_images(images)

# ----------------------------------------------------------------------
# ---------------------- TABLE / PLOT HELPERS --------------------------
# ----------------------------------------------------------------------

def extract_segment_times(sample_props: Dict[str, float], n_segments: int, date_format: int) -> pd.DataFrame:
    records = []
    for idx in range(1, n_segments + 1):
        segment = f"S{idx}"
        row = {"segment": segment}
        for metric in ["tStart", "tEnd", "tBreak", "changeProb", "numObs"]:
            key = f"{segment}_{metric}"
            value = sample_props.get(key)
            row[metric] = value
            if metric in {"tStart", "tEnd", "tBreak"}:
                row[f"{metric}_dt"] = decode_ccdc_date(value, date_format) if value is not None else None
        records.append(row)
    df = pd.DataFrame(records)
    return df.dropna(how="all", subset=["tStart", "tEnd", "tBreak", "changeProb", "numObs"])

def extract_coefficients(sample_props: Dict[str, float], band_list: List[str], n_segments: int) -> pd.DataFrame:
    records = []
    for idx in range(1, n_segments + 1):
        segment = f"S{idx}"
        for band in band_list:
            for coef in GLOBAL_SUBCOEFS + GLOBAL_EXTRA_COEFS:
                key = f"{segment}_{band}_coef_{coef}"
                if key in sample_props and sample_props[key] is not None:
                    records.append({"segment": segment, "band": band, "metric_type": "coef", "metric": coef, "value": sample_props[key]})
            rmse_key = f"{segment}_{band}_RMSE"
            if rmse_key in sample_props and sample_props[rmse_key] is not None:
                records.append({"segment": segment, "band": band, "metric_type": "RMSE", "metric": "RMSE", "value": sample_props[rmse_key]})
            mag_key = f"{segment}_{band}_MAG"
            if mag_key in sample_props and sample_props[mag_key] is not None:
                records.append({"segment": segment, "band": band, "metric_type": "MAG", "metric": "MAG", "value": sample_props[mag_key]})
    return pd.DataFrame(records)

def load_ccdc_asset(config: Dict[str, object]) -> Tuple[ee.Image, ee.Image]:
    path = config["ASSET_PATH"]
    path_type = config["PATH_TYPE"]
    if path_type == "Image":
        image = ee.Image(path)
        return image, image
    if path_type != "ImageCollection":
        raise ValueError("PATH_TYPE must be 'Image' or 'ImageCollection'.")

    collection = ee.ImageCollection(path)
    prefix = config.get("FILTER_PREFIX")
    if prefix:
        collection = collection.filterMetadata("system:index", "starts_with", prefix)
    if collection.size().getInfo() == 0:
        raise ValueError("ImageCollection is empty after applying the filter.")
    return collection.mosaic(), ee.Image(collection.first())

def gather_metadata(img: ee.Image) -> Dict[str, Optional[float]]:
    meta = img.toDictionary(["dateFormat", "startDate", "endDate"]).getInfo()
    return {
        "dateFormat": meta.get("dateFormat"),
        "startDate": meta.get("startDate"),
        "endDate": meta.get("endDate"),
    }

# ----------------------------------------------------------------------
# --------------------------- MAIN ROUTINE -----------------------------
# ----------------------------------------------------------------------

def main():
    project = CONFIG.get("EE_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        print("Failed to initialize Earth Engine:", exc)
        print("Hint: run `ee.Authenticate()` once before executing this script.")
        return

    n_segments = int(CONFIG["N_SEGMENTS"])
    band_list = list(CONFIG["BANDS"])
    target_band = CONFIG["TARGET_BAND"]
    seg_names = [f"S{i}" for i in range(1, n_segments + 1)]

    ccd_raw, metadata_image = load_ccdc_asset(CONFIG)
    metadata = gather_metadata(metadata_image)
    date_format = metadata.get("dateFormat")
    if date_format is None:
        print("Warning: dateFormat metadata missing; assuming fractional years (code 1).")
        date_format = 1
    date_format = int(date_format)

    ccd_long = build_ccd_image(ccd_raw, n_segments, band_list)
    phase_amp = new_phase_amplitude(ccd_long.select(".*coef.*"), ".*SIN.*", ".*COS.*")
    ccd_with_phase = ccd_long.addBands(phase_amp)

    point = ee.Geometry.Point([CONFIG["POINT_LON"], CONFIG["POINT_LAT"]])
    sample_fc = ccd_with_phase.sample(point, CONFIG["SAMPLE_SCALE"], numPixels=1)
    sample_feat = sample_fc.first()

    if sample_feat is None or sample_feat.getInfo() is None:
        print("No CCDC data found at this point.")
        return

    sample_props = sample_feat.toDictionary().getInfo()

    segment_times_df = extract_segment_times(sample_props, n_segments, date_format)
    coef_df = extract_coefficients(sample_props, band_list, n_segments)

    print("\n=== Segment metadata ===")
    display_fields = ["segment", "tStart_dt", "tEnd_dt", "tBreak_dt", "changeProb", "numObs"]
    print(segment_times_df[display_fields])

    if not coef_df.empty:
        print("\n=== Coefficients/RMSE/Magnitude (first rows) ===")
        print(coef_df.head())

    # Synthetic predictions
    date_values, datetime_values = generate_ccdc_date_series(
        CONFIG["ANALYSIS_START"], CONFIG["ANALYSIS_END"], date_format, CONFIG["STEP_MONTHS"]
    )
    synthetic_records = []
    for val, dt_obj in zip(date_values, datetime_values):
        synthetic_img = get_multi_synthetic(ccd_long, val, date_format, [target_band], seg_names)
        syn_feat = synthetic_img.sample(point, CONFIG["SAMPLE_SCALE"], numPixels=1).first()
        if syn_feat is None:
            continue
        syn_dict = syn_feat.toDictionary().getInfo()
        synthetic_records.append({"date": dt_obj, target_band: syn_dict.get(target_band)})

    synthetic_df = pd.DataFrame(synthetic_records)
    if synthetic_df.empty:
        print("\nNo synthetic values could be extracted for the chosen window.")
        return

    # Plotting
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(synthetic_df["date"], synthetic_df[target_band], marker="o", label=f"Synthetic {target_band}")

    # Mark breakpoints
    for _, row in segment_times_df.iterrows():
        if pd.notnull(row.get("tBreak_dt")):
            ax.axvline(row["tBreak_dt"], color="crimson", linestyle="--", alpha=0.35)

    ax.set_title(f"{target_band} synthetic CCDC time series at point ({CONFIG['POINT_LON']:.4f}, {CONFIG['POINT_LAT']:.4f})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target_band} value")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Optional: inspect intercept/slope table for the target band
    coef_subset = coef_df[(coef_df["band"] == target_band) & (coef_df["metric_type"] == "coef")]
    if not coef_subset.empty:
        pivot = coef_subset.pivot(index="segment", columns="metric", values="value")
        print("\n=== Target-band coefficients ===")
        print(pivot.loc[:, [c for c in ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3", "PHASE", "AMPLITUDE", "PHASE2", "AMPLITUDE2", "PHASE3", "AMPLITUDE3"] if c in pivot.columns]])

if __name__ == "__main__":
    main()
