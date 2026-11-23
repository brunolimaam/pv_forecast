# pv_forecast

A machine learning project that predicts Germany's solar (PV) power output
24 hours in advance. It learns from ICON-D2 weather forecasts fetched from
Open-Meteo, the realised PV generation published on SMARD, and the installed
PV capacity from the Energy-Charts API.

The forecast is shown, with an explanation of the model and its verification,
at [brunolima.de/projects/pv-icon-d2/](https://brunolima.de/projects/pv-icon-d2/).

## Data

| Source | What | Resolution |
| --- | --- | --- |
| [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) | ICON-D2, the German Weather Service's (DWD) high-resolution model for Central Europe: shortwave, direct and diffuse radiation, 2 m temperature, cloud cover and 10 m wind speed at 100 points across Germany | hourly |
| [SMARD](https://www.smard.de/) (Bundesnetzagentur) | Realised PV generation in Germany, in MW | hourly |
| [Energy-Charts API](https://api.energy-charts.info/) (Fraunhofer ISE) | Installed PV capacity in Germany, in GW | monthly |

The data starts on 1 January 2024. All timestamps are UTC.

## How the model works

- **Points.** The weather is taken at random points inside Germany, listed in
  `points.csv`. The file holds 1000 points; the first `N_POINTS` (100) in
  `download_data.py` are used, so the set can grow later.
- **Features.** For every point: the six ICON-D2 variables, plus two computed
  with [pvlib](https://pvlib-python.readthedocs.io/): the clear-sky irradiance
  (GHI) and whether the sun is up. With 100 points that is 800 features.
- **Model.** A gradient boosting regressor
  (scikit-learn's `HistGradientBoostingRegressor`). The forecast model is
  trained on all the data. For the verification, the same model is trained on
  the data up to 2026-01-19 and tested on the data from 2026-01-20 on.
- **Target.** The fraction of the installed capacity
  that is producing. This does not grow with the fleet, so the model never has
  to predict beyond what it saw in training (a difficulty from decision trees). The predicted fraction times the
  installed capacity gives the output in MW.

## Files

| File | What it does |
| --- | --- |
| `download_data.py` | Downloads the three data sources into `data/raw/`. The weather file is only updated if already exists |
| `train_model.py` | Builds the feature table, trains the model on all the data and saves it to `models/pv_model.joblib` |
| `verify_model.py` | Trains the same model on the earlier data, tests it on the rest, and plots the results to `data/plots/` |
| `forecast.py` | Runs the model on the newest ICON-D2 run, appends the hours 22–24 h and 46–48 h ahead to `forecast/pv_forecast.parquet`, and plots both timelines with the real output of the last three days |
| `forecast/` | The forecasts of all past runs and the forecast plots. Kept in the repository: past runs cannot be downloaded again |
| `points.csv` | The forecast points: name, latitude, longitude |
| `model.ipynb` | The explanation of the model and its verification, shown on the website |
| `models/` | The trained model, with the list of features it expects |

If the `data/` directory is not in the repository, running `download_data.py` creates it.

## Running it

This project uses Python 3.13 with multiple libraries. The exact version can be easily syncd with [uv](https://docs.astral.sh/uv/), which installs Python 3.13 and the
exact package versions documented in `uv.lock`. For that use:

```bash
uv sync
uv run python download_data.py   # data/raw/
uv run python train_model.py     # models/pv_model.joblib
uv run python verify_model.py    # data/plots/
uv run python forecast.py        # forecast/
```

`forecast.py` is meant to run every hour. ICON-D2 runs every 3 hours, and a
run that is already stored is skipped, so an hourly schedule still catches
every run when one start is late.

`verify_model.py` also trains a model for every number of points and for 1000
random sets of 10 points, so it takes a while. `train_model.py` trains only
the forecast model.
