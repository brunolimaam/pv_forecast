#!/usr/bin/env python3
"""Train the PV model on all the data and save it to models/pv_model.joblib.

Reads the three files download_data.py writes to data/raw/, adds two variables
per point computed with pvlib, and fits a gradient boosting model. The model
predicts the fraction of the installed capacity that is producing: it does not
grow with the fleet, so the trees never have to predict beyond what they saw
in training. The fraction times the capacity is the output in MW (predict_mw).

verify_model.py imports the table and the functions from here, to test the
same model on data it was not trained on.

Usage:
  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pvlib.location import Location
from sklearn.ensemble import HistGradientBoostingRegressor

from download_data import POINTS

# there are three files with data for the training model:
RAW = Path(__file__).parent / "data" / "raw"
WEATHER_DATA = RAW / "icon_d2_hist.parquet"
PV_OUTPUT_DATA = RAW / "pv_de.parquet"
PV_CAPACITY_DATA = RAW / "capacity_de.parquet"

# the model is saved together with its feature list: it needs these columns, in this order
MODEL_FILE = Path(__file__).parent / "models" / "pv_model.joblib"

# Chosing which features will be used to train the model.
FEATURES = [f"{variable}_{name}"
            for name, lat, lon in POINTS
            for variable in ('shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
                             'temperature_2m', 'cloud_cover', 'GHI', 'is_day', 'wind_speed_10m')]
TARGET = 'pv_mw'                                    # what the errors are measured on
FRACTION = 'pv_fraction'                            # what the models are trained on
CAPACITY_FACTOR = 'capacity_mw'


def add_sun(df):
    """Add two variables per point, computed with pvlib for the middle of each
    hour (Open-Meteo labels each hour with its end):
      1. clear-sky GHI [W/m2]
      2. day or night (1/0)"""
    times = pd.DatetimeIndex(df["valid_time"]) - pd.Timedelta(minutes=30)
    ghi = pd.DataFrame({f"GHI_{name}": Location(lat, lon).get_clearsky(times)["ghi"].to_numpy()
                        for name, lat, lon in POINTS}, index=df.index)
    is_day = pd.DataFrame({f"is_day_{name}":
                           (Location(lat, lon).get_solarposition(times)["apparent_elevation"] > 0)
                           .astype(int).to_numpy()
                           for name, lat, lon in POINTS}, index=df.index)
    return pd.concat([df, ghi, is_day], axis=1)


def build_table():
    """Weather, PV output and capacity in one table, one row per hour."""
    pv = pd.read_parquet(PV_OUTPUT_DATA)
    pv["valid_time"] += pd.Timedelta(hours=1)   # SMARD labels each hour with its start
    data = (pd.read_parquet(WEATHER_DATA)
              .merge(pv, on="valid_time", how="left")
              .merge(pd.read_parquet(PV_CAPACITY_DATA), on="valid_time", how="left"))
    data = add_sun(data)
    data["capacity_mw"] = data["solar_ac_gw"] * 1000
    data["pv_fraction"] = data["pv_mw"] / data["capacity_mw"]
    return data.dropna(subset=['pv_mw'])       # hours without a SMARD value can't be learned


def fit(train_rows, features, target):
    """Fit a model on the training data"""
    return HistGradientBoostingRegressor(early_stopping=False).fit(
        train_rows[features],                       # what column(s) will be used as feature(s)
        train_rows[target])                         # what column(s) will be used as target(s)


def predict_mw(model, data_to_predict_from, capacity_mw):
    """The predicted fraction times the installed capacity gives the output in MW."""
    return model.predict(data_to_predict_from) * np.asarray(capacity_mw)


def main():
    data = build_table()
    model = fit(data, FEATURES, FRACTION)       # all the data: this is the forecast model
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
    print(f"trained on {len(data):,} hours, {data['valid_time'].min():%Y-%m-%d} .. "
          f"{data['valid_time'].max():%Y-%m-%d}; saved the model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
