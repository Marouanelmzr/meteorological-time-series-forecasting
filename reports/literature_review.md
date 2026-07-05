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

