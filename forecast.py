#!/usr/bin/env python3
"""Forecast Germany's PV output from the newest ICON-D2 run, keep the forecast
and plot it against the real output.

  1. Finds the newest ICON-D2 run on Open-Meteo and downloads it for the
     forecast points. The weather is not saved.
  2. Feeds it to the model from train_model.py. The predicted fraction times
     the installed capacity from Energy-Charts gives the output in MW.
  3. Appends to forecast/pv_forecast.parquet the three hours that end at each
     horizon: run + 22..24 h and run + 46..48 h. ICON-D2 runs every 3 hours,
     so the windows of consecutive runs join into one timeline per horizon.
     Rows already in the file are never changed.
  4. Plots each timeline from three days ago to its last hour, with the real
     output from Energy-Charts for comparison, to forecast/pv_forecast_24h.png
     and forecast/pv_forecast_48h.png.

ICON-D2 forecasts reach 48 h, so that is the longest horizon. A run that is
already in the file is not downloaded again, so the script can run every hour:
that way it gets every run, even when a start is late. A run it misses leaves
a 3-hour gap in each timeline.

Timestamps are UTC and label the END of each hour, as in the training table.

Usage:
  python forecast.py
"""
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from download_data import POINTS, VARIABLES, download_capacity, get_json
from train_model import MODEL_FILE, add_sun

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
ICON_D2_META = "https://api.open-meteo.com/data/dwd_icon_d2/static/meta.json"
REAL_PV = "https://api.energy-charts.info/public_power"

HORIZONS = [24, 48]             # hours after the run start; ICON-D2 ends at +48 h
WINDOW = 3                      # hours kept per run and horizon: a new run comes every 3 hours
PAST = pd.Timedelta(days=3)     # the plots start this long before now
HOUR = pd.Timedelta(hours=1)

# lead time (hours after the run start) -> horizon, for the hours kept from each
# run: the WINDOW hours that end at each horizon, e.g. 22, 23, 24 -> 24
KEPT = {lead: h for h in HORIZONS for lead in range(h - WINDOW + 1, h + 1)}

# in the repository, not in data/: the forecasts of past runs cannot be downloaded again
OUT = Path(__file__).parent / "forecast"
FORECAST_FILE = OUT / "pv_forecast.parquet"


def newest_run():
    """Start time of the newest ICON-D2 run that Open-Meteo serves."""
    meta = get_json(ICON_D2_META)
    return pd.Timestamp(meta["last_run_initialisation_time"], unit="s", tz="UTC")


def download_run(run):
    """The newest run's weather from the first kept hour to the last, one row per
    valid_time and one column per variable and point, as in icon_d2_hist.parquet."""
    names, lats, lons = zip(*POINTS)
    data = get_json(FORECAST_API, {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "hourly": ",".join(VARIABLES),
        "models": "icon_d2",
        "timezone": "GMT",
        "start_hour": f"{run + min(KEPT) * HOUR:%Y-%m-%dT%H:%M}",
        "end_hour": f"{run + max(KEPT) * HOUR:%Y-%m-%dT%H:%M}",
    })
    if isinstance(data, dict):        # a single point comes back as one object
        data = [data]
    df = pd.DataFrame({f"{v}_{name}": loc["hourly"][v]
                       for name, loc in zip(names, data)
                       for v in VARIABLES})
    df.insert(0, "valid_time", pd.to_datetime(data[0]["hourly"]["time"], utc=True))
    return df


def capacity_mw(run):
    """Installed PV capacity in the month of the run, in MW, or in the newest
    month Energy-Charts has if that month is not published yet."""
    capacity = download_capacity().dropna(subset=["solar_ac_gw"])
    return capacity.loc[capacity["valid_time"] <= run, "solar_ac_gw"].iloc[-1] * 1000


def predict(run, weather, capacity):
    """The model output for the kept hours of the run: one row per horizon and hour."""
    kept = weather[((weather["valid_time"] - run) // HOUR).isin(list(KEPT))].reset_index(drop=True)
    if kept.isna().any().any():
        raise RuntimeError(f"the ICON-D2 run of {run:%Y-%m-%d %H} UTC has missing values")
    saved = joblib.load(MODEL_FILE)             # the model and the columns it expects
    fraction = saved["model"].predict(add_sun(kept)[saved["features"]])
    lead_h = (kept["valid_time"] - run) // HOUR
    return pd.DataFrame({"horizon_h": lead_h.map(KEPT), "run": run, "lead_h": lead_h,
                         "valid_time": kept["valid_time"], "pv_fraction": fraction,
                         "capacity_mw": capacity, "pv_mw": fraction * capacity})


def append(new, path):
    """Add the new rows to the parquet file at path and return everything in it.
    An hour the file already has for a horizon keeps its old value. The file is
    written to a temporary file first, so an interrupted run cannot damage it."""
    table = pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
    table = (table.drop_duplicates(["horizon_h", "valid_time"])
                  .sort_values(["horizon_h", "valid_time"])
                  .reset_index(drop=True))
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return table


def add_newest_run(forecast):
    """Add the newest run to the forecast file, unless it is already in it.
    Returns the whole forecast table, or None if there is none yet."""
    run = newest_run()
    if forecast is not None and (forecast["run"] == run).any():
        print(f"ICON-D2 run {run:%Y-%m-%d %H} UTC is already in {FORECAST_FILE.name}")
        return forecast
    print(f"ICON-D2 run {run:%Y-%m-%d %H} UTC, {len(POINTS)} points")
    weather = download_run(run)
    if newest_run() != run:
        # the download may mix the two runs: leave it, the next start gets the new run
        print("  a new run came out during the download, nothing added")
        return forecast
    new = predict(run, weather, capacity_mw(run))
    print(f"  capacity {new['capacity_mw'].iloc[0]:,.0f} MW; added:")
    print(new[["horizon_h", "lead_h", "valid_time", "pv_mw"]].to_string(index=False))
    return append(new, FORECAST_FILE)


def download_real(start, end):
    """Real PV output in Germany from Energy-Charts, hourly, MW. The API gives
    quarter hours labelled with their start, and their hourly means are SMARD's
    values, which the model was trained on. Hours without all four quarters are
    left out, and each hour is labelled with its end, like the forecast."""
    j = get_json(REAL_PV, {"country": "de", "start": f"{start:%Y-%m-%dT%H:%MZ}",
                           "end": f"{end:%Y-%m-%dT%H:%MZ}"})
    solar = next(p["data"] for p in j["production_types"] if p["name"] == "Solar")
    quarters = pd.Series(solar, index=pd.to_datetime(j["unix_seconds"], unit="s", utc=True),
                         dtype=float)
    hours = quarters.resample("h").agg(["mean", "count"])
    real = hours.loc[hours["count"] == 4, "mean"]
    real.index = real.index + HOUR
    return real


def plot(forecast, real, now):
    """One timeline per horizon, from PAST before now to its last forecast hour."""
    start = (now - PAST).floor("h")
    for h in HORIZONS:
        rows = forecast[forecast["horizon_h"] == h]
        end = rows["valid_time"].max()
        # on the full hourly range a missing hour breaks the line instead of being bridged
        hours = pd.date_range(start, end, freq="h")
        predicted = rows.set_index("valid_time")["pv_mw"].reindex(hours)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(hours, real.reindex(hours), lw=1.5, label="real (Energy-Charts)")
        ax.plot(hours, predicted, lw=1.5, label=f"predicted {h} h ahead")
        ax.axvline(now, color="gray", ls="--", lw=1, label="now")
        ax.set(title=f"PV output in Germany, ICON-D2 forecast {h} h ahead "
                     f"(newest run {rows['run'].max():%d %b %Y, %H} UTC)",
               xlabel="UTC", ylabel="PV [MW]", xlim=(start, end))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / f"pv_forecast_{h}h.png", dpi=150)
        plt.close(fig)
    print(f"plots written to {OUT}")


def main():
    OUT.mkdir(exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    forecast = add_newest_run(pd.read_parquet(FORECAST_FILE) if FORECAST_FILE.exists() else None)
    if forecast is not None:
        plot(forecast, download_real(now - PAST - HOUR, now), now)


if __name__ == "__main__":
    main()
