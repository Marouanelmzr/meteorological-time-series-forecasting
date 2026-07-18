# Time Series Forecasting vs. NWP Post-Processing

Two problems often confused because they both touch meteorology, but they don't
share the same input, nor the same suitable models.

## 1. Pure Time Series Forecasting

**Question asked:** given the history of a variable, what will its future value be?

```
y(t+1) = f( y(t), y(t-1), y(t-2), ... )
```

- **Input**: only the past values of the variable itself (and possibly other
  correlated variables).
- **No physical model upstream** — the model has to learn the temporal dynamics
  from scratch, purely from observations.
- **Typical output**: a continuous value (regression).
- **Suitable models**: ARIMA, SARIMA, Holt-Winters, LSTM/Encoder-Decoder, temporal
  Transformers.
- **Typical use case**: predicting tomorrow's temperature from its own history,
  with no simulated weather data as input.


## 2. NWP Post-Processing (Model Output Statistics)

**Question asked:** a physical model (NWP) has already simulated the future state
of the atmosphere — how do we correct or interpret that simulation?

```
y_observed = g( physical_model_output_at_this_instant, derived_features )
```

- **Input**: the already-computed output of a physical model (wind, shear, TKE,
  PBLH, temperature...) at a given `(run_time, valid_time, lead_time, station)`.
- **The temporal dynamics are already encoded** in the physical model's output —
  there is no need to relearn them.
- **Typical output**: can be continuous (bias correction) or **binary/rare**
  (event occurrence).
- **Suitable models**: logistic regression, Random Forest, Gradient Boosting
  (XGBoost/LightGBM), tabular neural networks — treated as a **tabular** problem,
  not a sequential one.
- **Typical use case**: from AROME variables (wind, TKE, shear...) at a given
  instant, predicting whether a wind gust will occur.


## 3. How to inject temporal signal into case 2 (without doing real forecasting)

Even in a post-processing problem, temporal information stays useful — but it is
added as a **feature**, not as a sequential architecture:

- **Chronological** train/val/test split (never random) to prevent leakage from
  the future into the past.
- Cyclical encodings: `sin/cos` of hour, day of year.
- `lead_time` as an explicit feature (the physical model's skill varies with the
  forecast horizon).
- Trend/lag features: delta between two consecutive `lead_time` values, rolling
  average over the last steps of the same `run_time`.


## 4. Summary Table

| | Time Series Forecasting | NWP Post-Processing |
|---|---|---|
| Input | Raw history of the variable | Already-simulated output of a physical model |
| Temporal dynamics | Learned by the model | Already encoded upstream |
| Nature of the problem | Sequential | Tabular (one row = one state) |
| Common target | Continuous value | Continuous or binary/rare |
| Typical models | ARIMA, SARIMA, LSTM | Logistic Regression, Random Forest, GBM |
| Role of time | Main input variable | Split constraint + auxiliary feature |


## 5. How to tell which case I'm in

- If my dataset contains **only the history of the variable I want to predict**
  (and nothing else) → time series forecasting.
- If my dataset contains **the output of a physical model already computed for
  the instant I want to predict** → post-processing / MOS, to be treated as a
  tabular problem.