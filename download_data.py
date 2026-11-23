#!/usr/bin/env python3
"""Download the raw data for the ICON-D2 day-ahead PV model.

Each source is saved as its own parquet file in data/raw/:

  icon_d2_hist.parquet   ICON-D2 from the Open-Meteo Historical Forecast API:
                         the first hours of every run stitched into one
                         continuous series. One row per valid_time and one
                         column per variable and point, e.g.
                         shortwave_radiation_p01. Updated in place: only the
                         days and columns the file lacks are downloaded.
  pv_de.parquet          Realised PV generation in Germany, SMARD filter 4068,
                         hourly, MW.
  capacity_de.parquet    Installed PV capacity in Germany, Energy-Charts,
                         GW (AC and DC). The API only has monthly values, so
                         each month's value is repeated for every hour.
                         Whole timeline.

Timestamps are UTC and kept as delivered. Open-Meteo labels radiation with the
END of the hour it averages; SMARD labels each hour with its START.
train_model.py aligns them (build_table), not here.

Usage:
  python download_data.py
"""
import time
from pathlib import Path

import pandas as pd
import requests

# random points inside Germany as (name, lat, lon)
N_POINTS = 100
POINTS = list(pd.read_csv(Path(__file__).parent / "points.csv")
                .head(N_POINTS).itertuples(index=False, name=None))

HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SMARD = "https://www.smard.de/app/chart_data/4068/DE/{file}.json"
ENERGY_CHARTS = "https://api.energy-charts.info/installed_power"

VARIABLES = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
             "temperature_2m", "cloud_cover", "wind_speed_10m"]
START = "2024-01-01"   # first day of the data we download
END = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date().isoformat()  # yesterday
OUT = Path(__file__).parent / "data" / "raw"
WEATHER_FILE = OUT / "icon_d2_hist.parquet"


def get_json(url, params=None):
    """GET with a wait-and-retry on HTTP 429 (Open-Meteo's per-minute limit)."""
    for _ in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 429:
            break
        print("  rate limited, waiting 60 s")
        time.sleep(60)
    r.raise_for_status()
    return r.json()


def months(start, end):
    """(first_day, last_day) ISO-date pairs covering start..end, one per calendar month."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    edges = [s, *pd.date_range(s + pd.offsets.MonthBegin(1), e, freq="MS"),
             e + pd.Timedelta(days=1)]
    return [(a.date().isoformat(), (b - pd.Timedelta(days=1)).date().isoformat())
            for a, b in zip(edges, edges[1:])]


def download_icon_d2(points, start, end, variables=VARIABLES):
    """One request per month for all points and variables at once, which keeps
    each request well under Open-Meteo's per-minute limit."""
    names, lats, lons = zip(*points)
    frames = []
    for first, last in months(start, end):
        print(f"  {first} .. {last}")
        data = get_json(HISTORICAL_FORECAST, {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "start_date": first,
            "end_date": last,
            "hourly": ",".join(variables),
            "models": "icon_d2",
            "timezone": "GMT",
        })
        if isinstance(data, dict):        # a single point comes back as one object
            data = [data]
        df = pd.DataFrame({f"{v}_{name}": loc["hourly"][v]
                           for name, loc in zip(names, data)
                           for v in variables})
        df.insert(0, "valid_time", pd.to_datetime(data[0]["hourly"]["time"], utc=True))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def update_icon_d2(path, points, start, end):
    """Update the parquet file at path instead of downloading everything again.
    Columns the file lacks (a new variable or point) are downloaded for the days it
    already covers; days before its first or after its last valid_time are downloaded
    for all points and variables. Without a file, everything is downloaded."""
    if not path.exists():
        return download_icon_d2(points, start, end)
    old = pd.read_parquet(path).set_index("valid_time")
    first, last = old.index.min().normalize(), old.index.max().normalize()
    day = pd.Timedelta(days=1)

    # missing columns, grouped by the points that lack them: a new variable is one
    # download for all points, a new point one download for all variables
    missing = {}
    for v in VARIABLES:
        lacking = tuple(p for p in points if f"{v}_{p[0]}" not in old.columns)
        if lacking:
            missing.setdefault(lacking, []).append(v)
    for lacking, variables in missing.items():
        print(f"  new columns: {', '.join(variables)} for {len(lacking)} points")
        new = download_icon_d2(lacking, first.date().isoformat(), last.date().isoformat(),
                               variables)
        old = old.join(new.set_index("valid_time"))

    # days outside the file
    parts = [old.reset_index()]
    if pd.Timestamp(start, tz="UTC") < first:
        print("  days before the file")
        parts.insert(0, download_icon_d2(points, start, (first - day).date().isoformat()))
    if last < pd.Timestamp(end, tz="UTC"):
        print("  days after the file")
        parts.append(download_icon_d2(points, (last + day).date().isoformat(), end))
    return (pd.concat(parts, ignore_index=True)
              .drop_duplicates("valid_time")
              .sort_values("valid_time")
              .reset_index(drop=True))


def download_pv(start, end):
    """SMARD serves one file per week; fetch every week that overlaps start..end."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    weeks = [ts for ts in get_json(SMARD.format(file="index_hour"))["timestamps"]
             if s - pd.Timedelta(days=8) < pd.Timestamp(ts, unit="ms", tz="UTC") < e]
    print(f"  {len(weeks)} weekly files")
    rows = []
    for ts in weeks:
        rows += get_json(SMARD.format(file=f"4068_DE_hour_{ts}"))["series"]
    df = pd.DataFrame(rows, columns=["valid_time", "pv_mw"])
    df["valid_time"] = pd.to_datetime(df["valid_time"], unit="ms", utc=True)
    df["pv_mw"] = df["pv_mw"].astype(float)
    df = df[(df["valid_time"] >= s) & (df["valid_time"] < e)]
    return df.drop_duplicates("valid_time").sort_values("valid_time").reset_index(drop=True)


def download_capacity():
    j = get_json(ENERGY_CHARTS, {"country": "de", "time_step": "monthly"})
    series = {p["name"]: p["data"] for p in j["production_types"]}
    monthly = pd.DataFrame({
        "valid_time": pd.to_datetime(j["time"], format="%m.%Y", utc=True),
        "solar_ac_gw": series["Solar AC"],
        "solar_dc_gw": series["Solar DC"],
    }).set_index("valid_time")
    # the API has nothing finer than monthly: every hour gets the value of its month
    hours = pd.date_range(monthly.index[0], monthly.index[-1] + pd.offsets.MonthBegin(1),
                          freq="h", inclusive="left", name="valid_time")
    return monthly.reindex(hours, method="ffill").reset_index()


def save(df, path):
    df.to_parquet(path, index=False)
    t = df.iloc[:, 0]
    print(f"  wrote {path} ({len(df):,} rows, {t.min()} -> {t.max()})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ICON-D2 historical forecast, {len(POINTS)} points, {START} .. {END}")
    save(update_icon_d2(WEATHER_FILE, POINTS, START, END), WEATHER_FILE)
    print("SMARD 4068, PV generation")
    save(download_pv(START, END), OUT / "pv_de.parquet")
    print("Energy-Charts, installed PV capacity")
    save(download_capacity(), OUT / "capacity_de.parquet")


if __name__ == "__main__":
    main()
