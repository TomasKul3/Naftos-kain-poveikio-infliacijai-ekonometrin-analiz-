# Naftos kainų poveikio infliacijai ekonometrinė analizė
# Projekto tikslas - nustatyti naftos kainų pokyčių poveikį infliacijai Lietuvoje, identifikuoti ilgalaikę kointegraciją tarp kintamųjų ir apskaičiuoti grįžimo po šoko greitį. Naudojama programavimo kalba - Python. Naudoti ekonometriniai modeliai - OLS ir VECM. Modeliams sudaryti naudojami du kintamieji - vidutinė mėnesio naftos kaina ir vidutinė metinė infliacija apskaučiuota pagal vartotojų kainų indeksą. Laiko eilutės - mėnesinės. Tiriamas Laikotarpis 2020 01 - 2025 12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygments.styles.dracula import green

# Užkrauname duomenis
df_Infliacija = pd.read_csv('Vidutine metine infliacija.csv')
df_Nafta = pd.read_csv('Crude Oil WTI Futures Historical Data.csv')

# Konvertuojame datas
df_Infliacija['Laikotarpis'] = pd.to_datetime(df_Infliacija['Laikotarpis'], format='%YM%m')
df_Nafta['Date'] = pd.to_datetime(df_Nafta['Date'])

# Sujungiame duomenis
df_combined = pd.merge(
    df_Infliacija[['Laikotarpis', 'Reikšmė']],
    df_Nafta[['Date', 'Price']],
    left_on='Laikotarpis',
    right_on='Date',
    how='inner'
)

#  OLS Model
df_combined['Price'] = pd.to_numeric(df_combined['Price'].astype(str).str.replace(',', ''))
results = smf.ols('Reikšmė ~ np.log(Price)', data=df_combined).fit()
print(results.summary())

# Grafikas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


sns.regplot(x=np.log(df_combined['Price']), y=df_combined['Reikšmė'],
            ax=ax1, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
ax1.set_title('OLS: Infliacija vs log(Naftos kaina)')
ax1.set_xlabel('log(Price)')
ax1.set_ylabel('Infliacija (%)')


ax2.plot(df_combined['Laikotarpis'], results.resid)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title(f'OLS Liekanos (Durbin-Watson: {sm.stats.durbin_watson(results.resid):.2f})')
ax2.set_ylabel('Paklaida')

plt.tight_layout()
plt.show()
<img width="1489" height="490" alt="OLS" src="https://github.com/user-attachments/assets/8ef9cdba-c789-4f61-8a21-45dfcca9d387" />

#%%
# Sprendžiant iš atlikto linijinės regresijos modelio rezultatų, naftos kainos įtaka infliacijai yra statistiškai reikšminga ir įtakoja 19,5% Infliacijos augimo. Atsižvelgiant į koeficientą, Naftos kainos pokytis 1% įtakoja Infliacijos pokytį 0,089%. Tačiau žemas (0,07) Durbin - Watson koeficientas nurodo aukšta autokoreliaciją, jog laiko eilutės (iš praeities) įtakoja duomenis, todėl siūlau įtraukti vieno mėnesio pavėlavimą - naftos kainos pokytis iš praeito mėnesio įtakoja einamojo mėnesio infliaciją.

#  Sukuriame logaritmuotą naftos kainą
df_combined['log_Price'] = np.log(df_combined['Price'])

#  Sukuriame 1 mėnesio vėlinimą (lag)
df_combined['log_Price_lag1'] = df_combined['log_Price'].shift(1)

#  Modelis su vėlinimu
# Naudojame .dropna(), nes pirmoji eilutė po shift(1) tampa NaN
results_lag = smf.ols('Reikšmė ~ log_Price_lag1', data=df_combined.dropna()).fit()
print(results_lag.summary())

#%%
# Įtraukus pavėlavimą, Durbin - Watson koeficientas tik sumažėja (Durbin-Watson: 0.064), siūlau patikrinti su ADF testu duomenų stacionarumą.
# Jei p-value > 0.05, kintamasis nestacionarus -> reikia diferencijuoti
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_combined['Reikšmė'])
print(f'Infliacijos ADF p-value: {result[1]}')
Infliacijos ADF p-value: 0.5928259185351312
# ADF testo rezultatai parodė, kad duomenys nestacionarūs, todėl juos reikia diferencijuoti.

#%%

# Sukuriame diferencijuotus kintamuosius
df_combined['d_infliacija'] = df_combined['Reikšmė'].diff()
df_combined['d_log_price'] = np.log(df_combined['Price']).diff()

# Sukuriame infliacijos lag'ą
df_combined['d_infliacija_lag1'] = df_combined['d_infliacija'].shift(1)

# Sukuriame naftos kainos lag'ą
df_combined['d_log_price_lag1'] = df_combined['d_log_price'].shift(1)

# OLS Modelis: Infliacijos pokytis priklauso nuo savo praeities ir naftos praeities
model_final = smf.ols('d_infliacija ~ d_infliacija_lag1 + d_log_price_lag1',
                        data=df_combined.dropna()).fit()

print(model_final.summary())


# Grafikas
plt.figure(figsize=(12, 5))

# Braižome tikrąją diferencijuotą infliaciją (nuo 2-os eilutės)
plt.plot(df_combined['Laikotarpis'].iloc[2:],
         df_combined['d_infliacija'].iloc[2:],
         label='Δ Infliacija (Tikroji)', color='blue')

# Braižome prognozę (ji natūraliai yra trumpesnė, todėl X ašis turi atitikti)
plt.plot(df_combined['Laikotarpis'].iloc[2:],
         model_final.fittedvalues,
         label='Modelio prognozė', color='orange', linestyle='--')

plt.title('Diferencijuotos infliacijos modeliavimas (Atmetus prarastas eilutes)')
plt.xlabel('Laikotarpis')
plt.ylabel('Pokytis')
plt.legend()
plt.show()
<img width="1025" height="470" alt="Diferencijuota" src="https://github.com/user-attachments/assets/ac837b6c-f516-43f4-b1b7-ae2e574929e8" />

# Pasiektas reikšmingas Durbin - Watson pagerėjimas, tačiau reikėtų atlikti platesnę ekonometrinę analizę taikant kitus modelius, nes nors ir diferencijavus kintamuosius ir pridėjus autoregresinius dėmenis, nepavyko pilnai išspręsti nestacionarumo ir autokoreliacijos problemos. Siūlau išanalizuoti duomenis taikant VECM, tam kad pamatyti kaip greitai naftos kainos ir infliacija grįžta po šoko (greičiausiai duomenys nestacionarūs dėl naftos kainų šoko 2020 ir 2022 metais ir ženklaus infliacijos šuolio 2023 metais - tokie šuoliai reiškia, kad duomenų vidurkis nėra pastovus laike)
#%%

# Žinome kad abu kintamieji yra nestacionarūs, tačiau prieš atliekant VECM reikia patikrinti jų kointegraciją Johansen testu.
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
jres = coint_johansen(df_combined[['Reikšmė', 'log_Price']], det_order=0, k_ar_diff=1)

print(jres.lr1) # Trace statistika
print(jres.cvt) # Kritinės reikšmės (turi būti didesnė už kritinę, kad būtų kointegracija)
[67.14474089  4.19114574]
[[13.4294 15.4943 19.9349]
 [ 2.7055  3.8415  6.6349]]

# Johansen testo rezultatas parodo, kad nafta ir infliacija turi statistiškai reikšminga ilgalaikį ryšį (4.19 žemiau už 6.63, bet aukščiau už 3.8, 67.14 aukščiausias) ir tai leidžia pereiti prie VECM modelio.
#%%

# k_ar_diff nurodo, kiek vėlinimų naudoti modelio diferencijuotoje dalyje
# coint_rank=1 nurodo, kiek yra kointegracinių ryšių
# VECM modelis
model_vecm = VECM(df_combined[['Reikšmė', 'log_Price']], k_ar_diff=1, coint_rank=1, deterministic='ci')
vecm_res = model_vecm.fit()

print(vecm_res.summary())

# Grafikas
irf = vecm_res.irf(periods=12)
irf.plot(impulse='log_Price', response='Reikšmė', orth=True)
plt.suptitle('Impulso atsako funkcija: Naftos kainos +1% šoko įtaka infliacijai', fontsize=12)
plt.show()

irf = vecm_res.irf(periods=24)
fig = irf.plot(impulse='log_Price', response='Reikšmė', orth=True)
fig.set_size_inches(10, 6)

for ax in fig.axes:
    ax.set_title('Naftos kainos šoko (+1%) ilgalaikis poveikis infliacijai', fontsize=12)
    ax.set_xlabel('Mėnesiai po šoko')
    ax.set_ylabel('Poveikis (proc. punktais)')
    # Pridedame liniją ties nuliu, kuri žymi pusiausvyrą
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()
<img width="951" height="973" alt="VECM" src="https://github.com/user-attachments/assets/601bd6d5-07b9-4789-9cff-090513ea3667" />
<img width="989" height="593" alt="VECMimpulse" src="https://github.com/user-attachments/assets/fb6a161f-bddd-4ef6-a02e-5aac6bedce8a" />


# VECM  rezultatas nurodo, kad tarp naftos kainų ir infliacijos egzistuoja stabilus ilgalaikis ryšys (beta kointegracijos koef. 12.53 - naftos kainai padidėjus 1%, infliacija padidėja 0,125%). Sistemos klaidų taisymo greitis (alpha) yra 2% per mėnesį, kas rodo lėtą, bet užtikrintą kainų reikšmių grįžimą po šoko. Taip pat modelis patvirtino stiprią infliacijos inerciją (AR1 koef. > 1).

# Projekto rezultatas - Atliktas tyrimas naudojant du kintamuosius: vidutinė mėnesio naftos kaina ir vidutinė metinė infliacija apskaučiuota pagal vartotojų kainų indeksa ir sukuriant du ekonometrinius modelius: OLS ir VECM, atskleidė, kad Naftos kainos pokytis 1% įtakoja Infliaciją Lietuvoje 0,125%. Taip pat analizė atskleidė, kad duomenys yra nestacionarūs ir duomenų vidurkis nėra pastovus laike. Tai paaiškina naftos kainų ir infliacijos šokai, kurie lėtai ( alpha 2% per mėnesį), bet užtikrintai grįžta į iki šoko būseną. Tam kad nustatyti tikslesnę naftos įtaką infliacijai siūlyčiau sekančiame tyrime įtraukti daugiau infliaciją įtakojančių kintamųjų (palūkanų normos, atlyginimo dydis, kitų energetinių šaltinių kainos) ir įtraukti pseudokintamuosius kriziniams laikotarpiams apibrėžti.
