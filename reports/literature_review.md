# Long-term / Short-term forcasting
Long term and short term forcasting are two different types of TSF.

**Long-term**: spans longer periods, including months, years, or even longer durations, and addresses challenges related to long-term trends and seasonal variations that can significantly impact prediction accuracy.

**Short-term**: involves shorter time spans, often ranging from hours to weeks, emphasizing high prediction accuracy and is suitable for tasks demanding precision.

But in our specific task, Wind gust is a short acceleration of speed that doesn't usually last more than a minute. And is also a rare phenomenon that doesn't usually occur. Taking that into consideration, we cannot use one type or another, Long-term forcasting is necessary to know what day the wind gust can occur, but also short-term forcasting can also be used for more precision withing that same day.

# Univariate and Multivariate forecasting

**Multivariate**: includes multiple variables.

**Univariate**: Only one variable is considered during the forecasting process.

These two types of forecasting affect the dimentionality of the inputs, and can be both discussed and explored to compare both performance and the accuracy of the forecasting methods.

- We can first see if the Univariate approch does not give enough signal to the models used.
- And if its not the case then we can explore either the Multivariate approch helps the model with more signal, or affects the model negatively due to the noise of useless informations.
- Last, we can also compare the performance of both approches, knowing that more variables mean more weights and higher training time.

# Understanding the Dataset

- The dataset contains observations from **26 meteorological stations**. If we train a model on the entire dataset as a single continuous time series, would this introduce noise? Since the records are chronologically ordered, consecutive observations may belong to different stations. Would these artificial transitions negatively affect the model?

  > Yes. Treating all stations as a single continuous sequence would introduce a significant source of noise. The model would attempt to learn temporal dependencies between observations originating from different geographical locations, even though no real temporal relationship exists between them. These artificial station-to-station transitions do not reflect the underlying physical process and may degrade the model's ability to learn meaningful patterns.
  >
  > To address this issue, each station must be treated as an **independent time series**. Sliding windows are therefore generated separately for every station, ensuring that each input sequence contains observations from only one location. This is why the **station identifier** is an essential feature during data preprocessing.

- Since this is a time series dataset spanning more than one year, how should temporal information be represented? Using only the day or month would incorrectly suggest that **31 December** and **1 January** are far apart, even though they are separated by only one day.

  > This issue is addressed through **cyclical time encoding**. Temporal variables such as the day of the year or hour of the day are transformed using sine and cosine functions, allowing the model to capture their periodic nature. Consequently, the encoded representation correctly reflects that the end and the beginning of the year are temporally adjacent, enabling the model to learn seasonal patterns more effectively.

- The primary variable of interest in this project is **wind**. Since wind is inherently a vector quantity, it is characterized by both **magnitude** and **direction**. Rather than storing wind speed and direction directly, the dataset represents wind using two orthogonal components: **U** and **V**, which together fully describe the wind vector.
  
  A key question is the distinction between **VENT.ZONAL** and **U.RAF60M**. Although both represent wind, what differentiates them?

  > Understanding this distinction is fundamental to the project.
  >
  > **VENT.ZONAL** and **VENT.MERIDIEN** correspond to the **mean horizontal wind components** measured during the observation period. Together, they describe the average wind vector at the surface.
  >
  > In contrast, **U.RAF60M** and **V.RAF60M** represent the horizontal components of the **maximum wind gust forecast** produced by the AROME numerical weather prediction model over the corresponding time interval.
  >
  > Both variables are necessary because wind gusts are short-duration events that typically last only a few seconds or minutes. As a result, they have little influence on the average wind speed and would not be adequately represented by the mean wind components alone. Incorporating both the average wind and the predicted gust information enables the model to capture both sustained winds and extreme wind events.

- What is the difference between **VENT.MERIDIEN** and variables such as **P85000VENT_MERID**? Why does **VENT.MERIDIEN** not specify an altitude?

  > **VENT.MERIDIEN** represents the meridional wind component measured at the standard meteorological reference height of **10 meters above ground level**, which explains why no altitude is explicitly indicated.
  >
  > Variables such as **P85000VENT_MERID** represent the meridional wind component at a specific **pressure level** rather than at a fixed geometric altitude. In this case, **850 hPa** corresponds to an atmospheric pressure level typically located around **1.5 km above sea level**, although its exact altitude varies with atmospheric conditions.
  >
  > Pressure levels are commonly used in meteorology because they provide a more consistent representation of atmospheric circulation than fixed altitudes.

- A notable limitation of this dataset is that the upper-atmosphere variables are available only for the **meridional (V) component** of the wind. The corresponding **zonal (U) component** is missing.

  > This represents a significant limitation because a complete wind vector requires both the **U** and **V** components. Without the zonal component, it is impossible to reconstruct the true wind magnitude or direction at these pressure levels. Consequently, any information derived from these variables provides only a partial description of the atmospheric wind field, potentially limiting the model's ability to exploit upper-air dynamics.

  - What is **AROME**, what is the difference between a **Numerical Weather Prediction (NWP)** model and a **Machine Learning (ML)** model, and why are AROME forecasts used as input features?

  > **AROME** is a high-resolution **Numerical Weather Prediction (NWP)** model developed by Météo-France. Unlike Machine Learning models, which learn statistical relationships from historical data, an NWP model predicts future atmospheric conditions by numerically solving the physical equations governing the atmosphere, including the conservation of momentum, mass, and energy.
  >
  > The two approaches therefore differ fundamentally. NWP models are **physics-based** and rely on mathematical simulations of the atmosphere, whereas ML models are **data-driven** and learn patterns directly from observations without explicitly modeling the underlying physical processes.
  >
  > In this project, AROME forecasts are used as input features because they provide valuable prior knowledge about the expected atmospheric state. Variables such as predicted wind, temperature, humidity, and wind gusts summarize complex physical interactions that would otherwise be difficult for the ML model to infer solely from historical observations.
  >
  > Rather than replacing the NWP model, the Machine Learning model complements it by learning the systematic biases and local effects that the numerical model may not fully capture. This hybrid approach often produces more accurate predictions than using either method independently.