#!/usr/bin/env python
# coding: utf-8

# # Renewable energy generation 1965-2018

# # Įvadas
# 
# Šis duomenų rinkinys pateikia informaciją apie Atsinaujinančių energijos šaltinių gamybą (TWh) visame pasaulyje. Duomenų rinkinį sudaro:
# 
# 1. 5036 eilutės
# 2. 7 stulpeliai:
#           - pasaulio valstybės
#           - valstybės kodas
#           - atsinaujinančių energijos šaltinių (AEŠ) tipai:
#                     * Vandens TWh (Hydropower (terawatt-hours));
#                     * Saulės TWh (Solar (terawatt-hours));
#                     * Vėjo TWh (Wind (terawatt-hours));
#                     * Kiti (biomasė, geoterminė energija) TWh (Other renewables (terawatt-hours))
# 
# 
# Analizės tikslas yra išanalizuoti pateiktus duomenis pagal jų pasiskirstymą pasaulyje, jų kasmetinį augimą ir sukurti tieisnės regresijos modelį.
# Pastaba: duomenų rinkinį sudaro šalys, regionai ir duomenys analizei naudojami tokie kokie yra, nes regionai apjungia šalis, kurios nėra atskirai išvardintos.
# 
# 
# P.S.: Teravatvalandė (TWh) - energijos matavimo vienetas, dažniausiai naudojamas elektros prietaisų suvartotai ar pagamintai elektros energijai skaičiuoti.

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # DataFrame apžvalga

# In[6]:


df = pd.read_csv("datasets_769175_1325857_modern-renewable-energy-consumption.csv")
df.head()
df


# In[7]:


df.describe()


# Paskaiciuotas PASAULIO atsinaujinačių energijos šaltinių gamybos vidurkis (TWh) pagal metus

# In[8]:


world_mean = df.groupby('Year')[['Hydropower (terawatt-hours)', 'Solar (terawatt-hours)', 
                                 'Wind (terawatt-hours)', 'Other renewables (terawatt-hours)']].mean().round(4)
world_mean 


# In[9]:


world_mean.plot()
plt.ylabel("TWh")


# In[206]:


world_all = df[['Hydropower (terawatt-hours)', 'Solar (terawatt-hours)', 
                                 'Wind (terawatt-hours)', 'Other renewables (terawatt-hours)']].mean()
world_all


# In[207]:


world_all.plot(kind = 'barh')


# Pasaulio šalių atsinaujinančių energijos šaltinių gamybos vidurkis pagal kategorijas

# In[12]:


world_countries = df.groupby('Entity')['Hydropower (terawatt-hours)', 'Solar (terawatt-hours)', 
                                 'Wind (terawatt-hours)', 'Other renewables (terawatt-hours)'].mean()
world_countries


# In[294]:


world_countries.plot(figsize = (8,5))
plt.ylabel('TWh')


# TOP 25 šalys, turinčios didžiausią VANDENS (Hydro) energijos gamybą

# In[238]:


hydro_25 = df.groupby('Entity')['Hydropower (terawatt-hours)'].sum().nlargest(25).round(4)
hydro_25


# In[239]:


hydro_25.plot(kind = 'bar')
plt.ylabel('TWh')


# TOP 25 šalys, turinčios didžiausią SAULĖS (Solar) energijos gamybą

# In[287]:


solar_25 = df.groupby('Entity')['Solar (terawatt-hours)'].sum().nlargest(15).round(4)
solar_25


# In[243]:


solar_25.plot(kind = 'bar')
plt.ylabel('TWh')


# TOP 25 šalys, turinčios didžiausią VĖJO (Wind) energijos gamybą

# In[245]:


wind_25 =df.groupby('Entity')['Wind (terawatt-hours)'].sum().nlargest(25).round(4)
wind_25


# In[246]:


wind_25.plot(kind = 'bar')
plt.ylabel('TWh')


# TOP 25 šalys, turinčios didžiausią KITI (Other) energijos gamybą

# In[247]:


other_25 = df.groupby('Entity')['Other renewables (terawatt-hours)'].sum().nlargest(25).round(4)
other_25


# In[248]:


other_25.plot(kind = 'bar')
plt.ylabel('TWh')


# LIETUVOS atsinaujinančių energijos šaltinių gamyba pagal metus

# In[22]:


LT = df[df['Entity'] == 'Lithuania']
LT.set_index("Year", inplace = True)
LT


# In[23]:


LT[['Hydropower (terawatt-hours)', 'Solar (terawatt-hours)',
        'Wind (terawatt-hours)', 'Other renewables (terawatt-hours)']].plot()
plt.ylabel('TWh')


# LIETUVOS atsinaujinančių energijos šaltinių vidurkis pagal visą periodą

# In[27]:


LT_mean = LT.mean()
LT_mean


# In[38]:


LT_mean.plot.barh()
plt.xlabel('TWh')


# Baltijos šalių (LT, LV, EST) AEŠ pasiskirstymas nuo 2000 m.

# In[29]:


baltic = df[((df['Entity'] == 'Lithuania') | (df['Entity'] == 'Latvia') | (df['Entity'] == 'Estonia')) & (df['Year'] >= 2010)]
baltic.set_index("Year", inplace = True)
baltic


# In[277]:


baltic.plot(kind = 'bar')
plt.ylabel('TWh')
plt.xlabel(['Estonia 2010-2018', 'Latvia 2010-2018', 'Lithuania 2010-2018'])


# Baltijos šalių AEŠ gamybos vidurkis pagal kategorijas

# In[31]:


baltic_mean = baltic.groupby('Entity')['Hydropower (terawatt-hours)', 'Solar (terawatt-hours)', 
                                 'Wind (terawatt-hours)', 'Other renewables (terawatt-hours)'].mean()
baltic_mean


# In[293]:


baltic_mean.plot.barh(figsize = (6,7))
plt.xlabel('TWh')


# # Duomenu vizualizavimas

# Koreliacijos įvertinimas tarp visų kintamųjų.
# 
# Galima pastebėti, kad pati geriausia koreliacija yra tarp kintamųjų:
# 
# - Wind (terawatt-hours) & Solar (terawatt-hours) ~ 87%
# - Wind (terawatt-hours) & Other renewables (terawatt-hours) ~ 79,7%
# 
# Gera:
# 
# 
# - Hydropower (terawatt-hours) & Other renewables (terawatt-hours) ~ 76,9 %

# In[41]:


df.corr()


# In[42]:


sns.heatmap(df.corr(), cmap = 'YlGnBu', annot = True)


# Koreliacija tarp visų kintamųjų

# In[161]:


sns.pairplot(df)


# # Prognozavimas

# In[176]:


X.isnull().sum()
df.fillna(0, inplace = True)
X.head()


# In[196]:


feature_cols = ['Solar (terawatt-hours)']


# In[197]:


X = df[feature_cols] 
y = df['Wind (terawatt-hours)']


# In[198]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Regresijos funkcija

# In[199]:


linear = LinearRegression()
linear.fit(X_train, y_train)


# Tikslumas (R2)

# In[200]:


accuracy = linear.score(X_test, y_test)
print(accuracy)


# In[201]:


print(linear.intercept_)
print(linear.coef_)


# Prognozė

# In[202]:


predictions = linear.predict(X_test)
predictions


# Tiesinė regresija ant duomenų, kurių nematė

# In[203]:


plt.scatter(X_test, y_test)
plt.plot(X_test, predictions, color = 'green')
plt.xlabel("Solar (terawatt-hours)")
plt.ylabel("Wind (terawatt-hours)")


# Tiesinė regresija ant train duomenų

# In[204]:



plt.scatter(X_train, y_train)
plt.plot(X_test, predictions, color = 'green')
plt.xlabel("Solar (terawatt-hours)")
plt.ylabel("Wind (terawatt-hours)")


# # Išvados

# Pagal atliktą anazlizę (1965-2018) galima padaryti tokias išvadas:
#     1. pagal bendrą viso pasaulio energijos gamybos vidurkį (1965-2018) Vandens energijos šaltiniai užima pirmą 
#     ir reikšmingą vietą tarp visų kitų AEŠ, antrą vietą - kiti energijos šaltiniai(biomasė, geaterminė energija), trečią - vėjo, 
#     paskutinę - saulės energijos gamyba.
#     2. žiūrint visų šalių vidurkius pagal kiekvienus metus, pirmauja ir nuolat auga Vandens energijos šaltiniai, taip pat 
#     gerą augimą parodė vėjo energija.
#     3.bendras viso pasaulio šalių AEŠ vidurkis per visą laikotarpį (1965 -2018) - pirmauja vandens ir kiti energijos šaltiniai.
#     4. Lietuvoje pagal energijos gamybos vidurkį pirmauja vandens ir vėjo energija su kitais (biomasė ir geotermine energija). 
#     Bet jei žiūrėti į AEŠ energijos gamybos augimą pagal metus, tai didelį potencialą ir prieuagį parodė vėjo ir biomasė&geoterminė 
#     energija, vandens energija tolygiai auga metų metus.
#     5. pagal Baltijos šalių analizę buvo nustatyta, kad LV pirmauja pagal vandens energiją, LT - pagal vėjo energiją,
#     EST- pagal saulės energiją.
#     6. Galima pastebėti, kad pati geriausia koreliacija yra tarp kintamųjų:
#        - Wind (terawatt-hours) & Solar (terawatt-hours) ~ 87%
#        - Wind (terawatt-hours) & Other renewables (terawatt-hours) ~ 79,7%
#     Išnagrinėjus kintamuosius, kurie turi didžiausią koreliaciją ir atlikus tiesinę regresiją, mes galime daryti išvadą, kad 
#     kaip mes matom AEŠ vidurkių grafike, tą patį atspintdi tiesinė regresija, būtent, kad saulės ir vejo energijos gamyba
#     auga kartu ir panasiais tempais.
