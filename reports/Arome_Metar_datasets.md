# Feature Selection and the Prevention of Temporal Leakage

## 1. Motivation

A model intended for operational deployment is only as trustworthy as the assumption that its inputs at inference time match its inputs at training time. In wind gust correction, this assumption is easy to violate silently: several variables that are technically present in a merged AROME–METAR dataset would, in a real forecasting setting, simply not exist yet at the moment a forecast is issued. A model trained on such variables can appear highly accurate offline while being unusable — or actively misleading — in production. This chapter formalises the criterion used throughout this study to decide which variables are admissible as predictors, distinguishes it from a more naive but insufficient notion of "using only past values," and outlines how METAR observations, excluded from the present feature set on these grounds, are nonetheless expected to be a productive direction for future work.

## 2. Two Distinct Notions of Time

Two timestamps must be kept conceptually separate when reasoning about causality in this pipeline:

- **$t_0$ — the issue time.** The instant at which the AROME model is initialised and run. This is the only moment at which any correction system, statistical or otherwise, can actually be executed in an operational setting.
- **$T$ — the valid time.** The instant the forecast describes, defined as $T = t_0 + h$, where $h$ is the forecast lead time (`forecast_hour`), ranging up to 96 h for station-level AROME forecasts in this network.

A predictor is admissible if, and only if, it is knowable at $t_0$ for every lead time $h$ used in the study. This is a stronger requirement than simply excluding the target variable itself: a feature can be entirely distinct from the target and still violate this condition if it is only observable near $T$ rather than at $t_0$.

## 3. Why "Only Past Values" Is an Insufficient Criterion

An initial, intuitive rule — "only use METAR observations from before the prediction is made" — is necessary but not sufficient, and applying it naively produces a leakage pattern that is easy to miss because it does not manifest as a single incorrect variable, but as an inconsistency that varies with lead time.

Consider a feature defined as "the most recent METAR observation prior to $T$." For a short lead time ($h = 1\,\text{h}$), this observation is drawn from close to $t_0$ and is a legitimate persistence signal. For a long lead time ($h = 96\,\text{h}$), the same construction rule would, if it were possible to compute today, draw on an observation close to $T$ — information that plainly does not exist at $t_0$, when the forecast must actually be produced. Constructing the feature relative to $T$ rather than $t_0$ therefore injects an amount of leakage that grows with lead time, silently inflating short-range skill scores relative to what is achievable operationally while leaving long-range scores comparatively — and misleadingly — more honest. Because the leakage is graded rather than binary, it is considerably harder to detect through standard train/test evaluation than a straightforward duplicate-of-the-target error.

The correct formulation replaces "before $T$" with "at or before $t_0$," and — critically — fixes this reference point across all lead times within a given AROME run, so that a single value of any persistence feature is shared by every row corresponding to that run, regardless of $h$.

## 4. Resulting Feature Set

Under this criterion, the full set of AROME output variables — `u10`, `v10`, `t2m`, `rh2m`, the 850/950 hPa wind components, `psurf`, `u_gust60`, `v_gust60`, `tke20m`, `edr20m`, and `pblh` — is admissible without qualification: every one of these is itself a forecast quantity, produced at $t_0$ and valid for $T$, and is therefore available by construction at the moment a correction would need to be applied.

METAR-derived variables, by contrast, are observations made *at* $T$ and are excluded from the baseline predictor set in their entirety — not only `gust` and `peak_wind_gust` but also concurrent state variables such as `tmpf`, `relh`, `mslp`, `vsby`, and the sky-condition and present-weather codes. None of these are known at $t_0$ for lead times beyond the very shortest, and including them without lead-time-aware handling would reproduce, in a more diffuse form, the leakage already identified and corrected earlier in this pipeline.

## 5. Future Work: METAR as a Persistence Signal

Excluding METAR from the present feature set is a methodological decision, not a judgement on its informativeness. The station's own recent observational history is a natural and physically motivated source of additional skill, and is deliberately left for a subsequent phase of this work rather than incorporated prematurely.

The proposed extension is a **persistence feature**, defined strictly with respect to the issue time:

$$
\text{metar\_persist}(s, t_0) = \text{most recent METAR observation at station } s \text{ available at or before } t_0
$$

evaluated once per station per run and broadcast identically to every lead time $h$ associated with that run. This construction is causally sound by design: it cannot vary with $T$, and therefore cannot introduce the lead-time-dependent leakage described in Section 3.

Framed this way, the extension also becomes a well-posed empirical question rather than an assumption: does conditioning the correction model on the atmospheric state actually observed at issue time improve on AROME-only predictors, and if so, at which lead times? The expected pattern — meaningful gain at short lead times, decaying toward negligible as $h$ grows and the persistence signal becomes uninformative about conditions tens of hours later — is itself a testable hypothesis. Evaluating it requires no change to the modelling framework established here, only the addition of the persistence feature and a lead-time-stratified comparison  between the AROME-only baseline and its METAR-augmented counterpart. This is proposed as a natural next step once the baseline established in this chapter has been validated.