#!/usr/bin/env python3
"""Verify the PV model: train it on the hours before TEST_START, test it on the
rest, and plot the results to data/plots/.

  - the test error of the model with all the points
  - the test error against the number of points it uses: 1, 2, ... all
  - N_SETS random sets of 10 points: which points make the error lower or higher

The model, its features and the table come from train_model.py, so this tests
the same model that is used for the forecast, only trained on less data.
It trains more than 1100 models, so it takes a while.

Usage:
  python verify_model.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from download_data import POINTS
from train_model import CAPACITY_FACTOR, FEATURES, FRACTION, TARGET, build_table, fit, predict_mw

# outline of Germany, drawn behind the points in the maps
world = gpd.read_file("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip")
germany = world[world["NAME"] == "Germany"].geometry

TEST_START = pd.Timestamp("2026-01-20", tz="UTC")   # first 2 years train, the rest test
N_SETS = 1000                                       # random sets of 10 points
MODEL_NAME = f"{len(POINTS)} points"                # the model with all the points, in titles
PLOTS = Path(__file__).parent / "data" / "plots"


def point_map(ax, values, title, signed=False):
    """Draw the points on a map of Germany, marker area proportional to values
    (one value per point, in POINTS order). By default negative values are noise,
    drawn as 0. With signed=True the area follows the size of the value and the
    colour its sign: red positive, blue negative."""
    values = np.array(values, dtype=float)
    if not signed:
        values = np.clip(values, 0, None)
    size = np.abs(values)
    germany.plot(ax=ax, color="whitesmoke", edgecolor="gray")
    ax.scatter([lon for _, _, lon in POINTS], [lat for _, lat, _ in POINTS],
               s=size / size.max() * 400 if size.max() > 0 else 0,
               color=np.where(values < 0, "tab:blue", "tab:red"), alpha=0.6)
    for name, lat, lon in POINTS:
        ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.set(title=title, xlabel="longitude", ylabel="latitude")


data = build_table()
TRAIN_ROWS = data[data["valid_time"] < TEST_START]
TEST_ROWS = data[data["valid_time"] >= TEST_START]

# the model with all the points
model = fit(TRAIN_ROWS, FEATURES, FRACTION)
predicted = predict_mw(model, TEST_ROWS[FEATURES], TEST_ROWS[CAPACITY_FACTOR])
residual = predicted - TEST_ROWS[TARGET]
mae = mean_absolute_error(TEST_ROWS[TARGET], predicted)
mse = mean_squared_error(TEST_ROWS[TARGET], predicted)
print(f"model {MODEL_NAME}: {len(TRAIN_ROWS):,} train rows, {len(TEST_ROWS):,} test rows, "
      f"test MAE {mae:,.0f} MW, MSE {mse:,.0f} MW²")


# test MAE of the whole-data model against the number of points it uses:
# the model is trained once with the first point, once with the first two, ...
mae_by_points, mse_by_points = {}, {}
for n in range(1, len(POINTS) + 1):
    features = [f for f in FEATURES if f.rsplit("_", 1)[-1] in [p[0] for p in POINTS[:n]]]
    set_model = fit(TRAIN_ROWS, features, FRACTION)
    set_pred = predict_mw(set_model, TEST_ROWS[features], TEST_ROWS[CAPACITY_FACTOR])
    mae_by_points[n] = mean_absolute_error(TEST_ROWS[TARGET], set_pred)
    mse_by_points[n] = mean_squared_error(TEST_ROWS[TARGET], set_pred)
    print(f"{n:2d} points: test MAE {mae_by_points[n]:,.0f} MW, MSE {mse_by_points[n]:,.0f} MW²")


# rng = np.random.default_rng(0)
# y = TEST_ROWS[TARGET]
# base_pred = predict_mw(model, TEST_ROWS[FEATURES], TEST_ROWS[CAPACITY_FACTOR])
# base_mae = mean_absolute_error(y, base_pred)
# base_mse = mean_squared_error(y, base_pred)

# groups = {v: [c for c in data_to_predict_from.columns if c.startswith(v + "_")]
#           for v in ('shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
#                     'temperature_2m', 'cloud_cover', 'GHI', 'is_day', 'wind_speed_10m')}
# groups["lead_h"] = ["lead_h"]

# importance_mae, importance_mse = {}, {}
# for name, cols in groups.items():
#     shuffled = data_to_predict_from.copy()
#     shuffled[cols] = shuffled[cols].to_numpy()[rng.permutation(len(data_to_predict_from))]
#     shuffled_pred = predict_mw(model, shuffled, TEST_ROWS["capacity_mw"])
#     importance_mae[name] = mean_absolute_error(y, shuffled_pred) - mae
#     importance_mse[name] = mean_squared_error(y, shuffled_pred) - mse
#     print(f"{name:20s} +{importance_mae[name]:,.0f} MW   +{importance_mse[name]:,.0f} MW²")

# # bar charts of the importance, largest at the top: MAE on the left, MSE on the right
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# for ax, importance, base, label, unit in [(axes[0], importance_mae, base_mae, "MAE", "MW"),
#                                           (axes[1], importance_mse, base_mse, "MSE", "MW²")]:
#     ranked = pd.Series(importance).sort_values()
#     ax.barh(ranked.index, ranked.to_numpy())
#     ax.set(title=f"Permutation importance, model all (test {label} {base:,.0f} {unit})",
#            xlabel=f"increase in test {label} when the group is shuffled [{unit}]")
# fig.tight_layout()
# fig.savefig(PLOTS / "importance_all.pdf")
# plt.close(fig)

# # permutation importance per point: shuffle all variables of one point at once
# importance_points_mae, importance_points_mse = {}, {}
# for name, lat, lon in POINTS:
#     cols = [c for c in data_to_predict_from.columns if c.endswith("_" + name)]
#     shuffled = data_to_predict_from.copy()
#     shuffled[cols] = shuffled[cols].to_numpy()[rng.permutation(len(data_to_predict_from))]
#     shuffled_pred = predict_mw(model, shuffled, TEST_ROWS["capacity_mw"])
#     importance_points_mae[name] = mean_absolute_error(y, shuffled_pred) - base_mae
#     importance_points_mse[name] = mean_squared_error(y, shuffled_pred) - base_mse
#     print(f"{name:5s} +{importance_points_mae[name]:,.0f} MW   +{importance_points_mse[name]:,.0f} MW²")

# # maps side by side: MAE on the left, MSE on the right
# fig, axes = plt.subplots(1, 2, figsize=(12, 7))
# for ax, importance, label, unit in [(axes[0], importance_points_mae, "MAE", "MW"),
#                                     (axes[1], importance_points_mse, "MSE", "MW²")]:
#     values = [importance[name] for name, lat, lon in POINTS]
#     point_map(ax, values, f"Permutation importance per point, model all, {label}\n"
#                           f"(marker area proportional, largest = +{max(values):,.0f} {unit})")
# fig.tight_layout()
# fig.savefig(PLOTS / "importance_points_map.pdf")
# plt.close(fig)


# N_SETS random sets of 10 points: train the whole-data model on each set and keep its
# test MAE and MSE
set_rng = np.random.default_rng(1)
set_results = []        # one dict per set: the names of its points, test MAE and test MSE
for i in range(N_SETS):
    chosen = [POINTS[j][0] for j in sorted(set_rng.choice(len(POINTS), 10, replace=False))]
    features = [f for f in FEATURES if f.rsplit("_", 1)[-1] in chosen]
    set_model = fit(TRAIN_ROWS, features, FRACTION)
    set_pred = predict_mw(set_model, TEST_ROWS[features], TEST_ROWS[CAPACITY_FACTOR])
    set_mae = mean_absolute_error(TEST_ROWS[TARGET], set_pred)
    set_mse = mean_squared_error(TEST_ROWS[TARGET], set_pred)
    set_results.append({"points": chosen, "mae": set_mae, "mse": set_mse})
    print(f"set {i + 1:3d}: test MAE {set_mae:,.0f} MW, MSE {set_mse:,.0f} MW²")

# mean test error of each point: the mean over all the sets the point appears in
mean_error = {"mae": {}, "mse": {}}
for key in mean_error:
    for name, lat, lon in POINTS:
        in_sets = [r[key] for r in set_results if name in r["points"]]
        if in_sets:
            mean_error[key][name] = np.mean(in_sets)


PLOTS.mkdir(parents=True, exist_ok=True)

# map of Germany with the points
fig, ax = plt.subplots(figsize=(6, 7))
germany.plot(ax=ax, color="whitesmoke", edgecolor="gray")
for name, lat, lon in POINTS:
    ax.plot(lon, lat, "o", color="tab:red", ms=5)
    ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set(title="Forecast points", xlabel="longitude", ylabel="latitude")
fig.tight_layout()
fig.savefig(PLOTS / "points_map.pdf")
plt.close(fig)


# residuals: predicted minus real, against the predicted value
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(predicted, residual, s=3, alpha=0.3)
ax.axhline(0, color="k", lw=1)
ax.set(title=f"Residuals, model {MODEL_NAME} (test MAE {mae:,.0f} MW, MSE {mse:,.0f} MW²)",
       xlabel="predicted PV [MW]", ylabel="predicted - real [MW]")
fig.tight_layout()
fig.savefig(PLOTS / "residuals.pdf")
plt.close(fig)

# timeline of the test period: real and predicted PV output as daily means,
# so each is one line (hourly values swing to zero every night)
daily = (pd.DataFrame({"real": TEST_ROWS[TARGET].to_numpy(), "predicted": predicted},
                      index=pd.DatetimeIndex(TEST_ROWS["valid_time"]))
         .resample("D").mean())
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily.index, daily["real"], lw=1.5, label="real")
ax.plot(daily.index, daily["predicted"], lw=1.5, label="predicted")
ax.set(title=f"Test period, model {MODEL_NAME} (daily mean)", xlabel="", ylabel="PV [MW]")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "timeline.pdf")
plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(mae_by_points), list(mae_by_points.values()), "o-", color="tab:blue")
ax.set(title="Whole-data model: test MAE and MSE against number of points",
       xlabel="number of points", xticks=list(mae_by_points))
ax.set_ylabel("test MAE [MW]", color="tab:blue")
ax2 = ax.twinx()        # the MSE has other units, so it gets its own axis on the right
ax2.plot(list(mse_by_points), list(mse_by_points.values()), "s-", color="tab:orange")
ax2.set_ylabel("test MSE [MW²]", color="tab:orange")
ax.legend(ax.lines + ax2.lines, ["MAE", "MSE"])
fig.tight_layout()
fig.savefig(PLOTS / "mae_by_points.pdf")
plt.close(fig)


# anomaly maps side by side, MAE on the left and MSE on the right: each point's mean
# minus the mean over all the sets, so red points are worse than average, blue better
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
for ax, key, label, unit in [(axes[0], "mae", "MAE", "MW"), (axes[1], "mse", "MSE", "MW²")]:
    overall = np.mean([r[key] for r in set_results])
    anomaly = [mean_error[key].get(name, overall) - overall for name, lat, lon in POINTS]
    point_map(ax, anomaly, f"Test {label} anomaly of the random sets each point appears in\n"
                           f"(mean of all sets {overall:,.0f} {unit}; red above, blue below; "
                           f"largest {np.abs(anomaly).max():,.0f} {unit})", signed=True)
fig.tight_layout()
fig.savefig(PLOTS / "random_sets_map.pdf")
plt.close(fig)
