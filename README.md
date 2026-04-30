# Econometric analysis of the impact of oil prices on inflation
# Project objective – to determine the impact of oil price changes on inflation in Lithuania,
# identify long-run cointegration between variables, and estimate the speed of adjustment after a shock.
# Programming language used – Python.
# Econometric models applied – OLS and VECM.
# The models use two variables – average monthly oil price and average annual inflation
# calculated based on the Consumer Price Index (CPI).
# Time series frequency – monthly.
# Examined period: 2020 01 – 2025 12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygments.styles.dracula import green

# Load data
df_Infliacija = pd.read_csv('Vidutine metine infliacija.csv')
df_Nafta = pd.read_csv('Crude Oil WTI Futures Historical Data.csv')

# Convert dates
df_Infliacija['Laikotarpis'] = pd.to_datetime(df_Infliacija['Laikotarpis'], format='%YM%m')
df_Nafta['Date'] = pd.to_datetime(df_Nafta['Date'])

# Merge datasets
df_combined = pd.merge(
    df_Infliacija[['Laikotarpis', 'Reikšmė']],
    df_Nafta[['Date', 'Price']],
    left_on='Laikotarpis',
    right_on='Date',
    how='inner'
)

# OLS Model
df_combined['Price'] = pd.to_numeric(df_combined['Price'].astype(str).str.replace(',', ''))
results = smf.ols('Reikšmė ~ np.log(Price)', data=df_combined).fit()
print(results.summary())

OLS Regression Results                            
==============================================================================
Dep. Variable:                Reikšmė   R-squared:                       0.195
Model:                            OLS   Adj. R-squared:                  0.184
Method:                 Least Squares   F-statistic:                     16.96
Date:                Thu, 30 Apr 2026   Prob (F-statistic):           0.000103
Time:                        11:35:56   Log-Likelihood:                -229.92
No. Observations:                  72   AIC:                             463.8
Df Residuals:                      70   BIC:                             468.4
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept       -31.1056      9.139     -3.404      0.001     -49.333     -12.879
np.log(Price)     8.9378      2.170      4.119      0.000       4.610      13.266
==============================================================================
Omnibus:                       10.365   Durbin-Watson:                   0.070
Prob(Omnibus):                  0.006   Jarque-Bera (JB):               11.666
Skew:                           0.975   Prob(JB):                      0.00293
Kurtosis:                       2.708   Cond. No.                         57.7
==============================================================================

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.regplot(x=np.log(df_combined['Price']), y=df_combined['Reikšmė'],
            ax=ax1, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
ax1.set_title('OLS: Inflation vs log(Oil price)')
ax1.set_xlabel('log(Price)')
ax1.set_ylabel('Inflation (%)')

ax2.plot(df_combined['Laikotarpis'], results.resid)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title(f'OLS Residuals (Durbin-Watson: {sm.stats.durbin_watson(results.resid):.2f})')
ax2.set_ylabel('Error')

plt.tight_layout()
plt.show()
<img width="1489" height="490" alt="OLS" src="https://github.com/user-attachments/assets/5fda8a55-d1d3-4259-8fb3-10ede9dc2b78" />


# Based on the linear regression model results, the impact of oil prices on inflation
# is statistically significant and explains 19.5% of inflation growth.
# According to the estimated coefficient, a 1% increase in oil prices
# leads to a 0.089% increase in inflation.
# However, the low Durbin-Watson statistic (0.07) indicates high autocorrelation,
# meaning past time series values influence current data.
# Therefore, it is suggested to include a one-month lag –
# oil price changes from the previous month may affect current inflation.

# Create log-transformed oil price
df_combined['log_Price'] = np.log(df_combined['Price'])

# Create 1-month lag
df_combined['log_Price_lag1'] = df_combined['log_Price'].shift(1)

# Model with lag
# Using .dropna() because the first row after shift(1) becomes NaN
results_lag = smf.ols('Reikšmė ~ log_Price_lag1', data=df_combined.dropna()).fit()
print(results_lag.summary())
OLS Regression Results                            
==============================================================================
Dep. Variable:                Reikšmė   R-squared:                       0.174
Model:                            OLS   Adj. R-squared:                  0.162
Method:                 Least Squares   F-statistic:                     14.56
Date:                Thu, 30 Apr 2026   Prob (F-statistic):           0.000292
Time:                        11:35:57   Log-Likelihood:                -228.05
No. Observations:                  71   AIC:                             460.1
Df Residuals:                      69   BIC:                             464.6
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept        -29.1588      9.362     -3.115      0.003     -47.835     -10.483
log_Price_lag1     8.4757      2.221      3.816      0.000       4.045      12.906
==============================================================================
Omnibus:                       10.140   Durbin-Watson:                   0.064
Prob(Omnibus):                  0.006   Jarque-Bera (JB):               11.319
Skew:                           0.961   Prob(JB):                      0.00348
Kurtosis:                       2.634   Cond. No.                         57.6
==============================================================================

# After including the lag, the Durbin-Watson statistic decreases even further (Durbin-Watson: 0.064),
# therefore it is suggested to test stationarity using the ADF test.
# If p-value > 0.05, the variable is non-stationary -> differencing is required

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_combined['Reikšmė'])
print(f'Inflation ADF p-value: {result[1]}')
Inflation ADF p-value: 0.5928259185351312

# The ADF test results indicate that the data are non-stationary,
# therefore they must be differenced.
# Create differenced variables
df_combined['d_infliacija'] = df_combined['Reikšmė'].diff()
df_combined['d_log_price'] = np.log(df_combined['Price']).diff()

# Create inflation lag
df_combined['d_infliacija_lag1'] = df_combined['d_infliacija'].shift(1)

# Create oil price lag
df_combined['d_log_price_lag1'] = df_combined['d_log_price'].shift(1)

# OLS Model: Inflation change depends on its own past and oil price past
model_final = smf.ols('d_infliacija ~ d_infliacija_lag1 + d_log_price_lag1',
                        data=df_combined.dropna()).fit()

print(model_final.summary())
OLS Regression Results                            
==============================================================================
Dep. Variable:           d_infliacija   R-squared:                       0.962
Model:                            OLS   Adj. R-squared:                  0.961
Method:                 Least Squares   F-statistic:                     851.6
Date:                Thu, 30 Apr 2026   Prob (F-statistic):           2.32e-48
Time:                        11:35:57   Log-Likelihood:                 27.464
No. Observations:                  70   AIC:                            -48.93
Df Residuals:                      67   BIC:                            -42.18
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept            -0.0007      0.020     -0.034      0.973      -0.041       0.039
d_infliacija_lag1     0.9804      0.024     41.238      0.000       0.933       1.028
d_log_price_lag1     -0.0883      0.136     -0.649      0.518      -0.360       0.183
==============================================================================
Omnibus:                        6.572   Durbin-Watson:                   0.808
Prob(Omnibus):                  0.037   Jarque-Bera (JB):                6.613
Skew:                          -0.751   Prob(JB):                       0.0366
Kurtosis:                       2.901   Cond. No.                         6.81
==============================================================================

# Plot
plt.figure(figsize=(12, 5))

# Plot actual differenced inflation (starting from 2nd row)
plt.plot(df_combined['Laikotarpis'].iloc[2:],
         df_combined['d_infliacija'].iloc[2:],
         label='Δ Inflation (Actual)', color='blue')

# Plot model prediction
plt.plot(df_combined['Laikotarpis'].iloc[2:],
         model_final.fittedvalues,
         label='Model prediction', color='orange', linestyle='--')

plt.title('Modeling Differenced Inflation (After dropping lost observations)')
plt.xlabel('Period')
plt.ylabel('Change')
plt.legend()
plt.show()
<img width="1025" height="470" alt="Diferencijuota" src="https://github.com/user-attachments/assets/8b195bff-ebf0-4e35-bf04-c4b965ad9a0a" />

# A significant improvement in the Durbin-Watson statistic was achieved.
# However, a broader econometric analysis is required,
# since even after differencing and adding autoregressive terms,
# The problems of non-stationarity and autocorrelation were not fully resolved.
# It is suggested to analyze the data using VECM
# to observe how quickly oil prices and inflation return after a shock
# (most likely the data are non-stationary due to oil price shocks in 2020 and 2022
# and the significant inflation spike in 2023 –
# such shocks imply that the mean of the data is not constant over time).
# We know that both variables are non-stationary,
# but before estimating VECM we must test for cointegration using the Johansen test.

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
jres = coint_johansen(df_combined[['Reikšmė', 'log_Price']], det_order=0, k_ar_diff=1)

print(jres.lr1) # Trace statistic
print(jres.cvt) # Critical values (trace must be higher than critical value to confirm cointegration)
[67.14474089  4.19114574]
[[13.4294 15.4943 19.9349]
 [ 2.7055  3.8415  6.6349]]

# Johansen test results indicate that oil prices and inflation
# have a statistically significant long-run relationship,
# therefore, we can proceed to estimate the VECM model.
# k_ar_diff indicates how many lags to use in the differenced part of the model
# coint_rank=1 indicates the number of cointegration relationships
# VECM model

model_vecm = VECM(df_combined[['Reikšmė', 'log_Price']], k_ar_diff=1, coint_rank=1, deterministic='ci')
vecm_res = model_vecm.fit()

print(vecm_res.summary())
Det. terms outside the coint. relation & lagged endog. parameters for equation Reikšmė
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
L1.Reikšmė       1.0094      0.016     64.687      0.000       0.979       1.040
L1.log_Price    -0.1675      0.088     -1.905      0.057      -0.340       0.005
Det. terms outside the coint. relation & lagged endog. parameters for equation log_Price
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
L1.Reikšmė      -0.0083      0.021     -0.402      0.688      -0.049       0.032
L1.log_Price     0.1454      0.117      1.245      0.213      -0.083       0.374
              Loading coefficients (alpha) for equation Reikšmė               
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ec1           -0.0207      0.002     -9.567      0.000      -0.025      -0.016
             Loading coefficients (alpha) for equation log_Price              
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ec1            0.0057      0.003      1.969      0.049    2.73e-05       0.011
          Cointegration relations for loading-coefficients-column 1           
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
beta.1         1.0000          0          0      0.000       1.000       1.000
beta.2       -12.5348      1.883     -6.658      0.000     -16.225      -8.845
const         46.1896      7.940      5.818      0.000      30.628      61.751
==============================================================================

# Impulse response function plot
irf = vecm_res.irf(periods=12)
irf.plot(impulse='log_Price', response='Reikšmė', orth=True)
plt.suptitle('Impulse response function: +1% oil price shock impact on inflation', fontsize=12)
plt.show()
<img width="989" height="593" alt="VECMimpulse" src="https://github.com/user-attachments/assets/9f0006a8-ba12-42c1-a4b1-b4c75d7fcfec" />
<img width="951" height="973" alt="VECM" src="https://github.com/user-attachments/assets/88fcdf73-9384-4c80-b904-c1a6b9dab5eb" />


# VECM results indicate that there is a stable long-run relationship
# between oil prices and inflation.
# The cointegration coefficient (beta) is 12.53 –
# a 1% increase in oil prices increases inflation by 0.125%.
# The speed of adjustment (alpha) is 2% per month,
# indicating slow but steady convergence after a shock.
# The model also confirms strong inflation inertia (AR1 coefficient > 1).

# Project conclusion –
# The study, using two variables (average monthly oil price and average annual CPI-based inflation)
# and applying two econometric models (OLS and VECM),
# reveals that a 1% increase in oil prices increases inflation in Lithuania by 0.125%.
# The analysis also shows that the data are non-stationary
# and the mean is not constant over time.
# Oil price and inflation shocks return slowly (alpha 2% per month)
# but steadily to equilibrium.
# For a more precise estimation of oil price impact on inflation,
# future research should include additional inflation determinants
# (interest rates, wage levels, other energy prices)
# and dummy variables for crisis periods.
