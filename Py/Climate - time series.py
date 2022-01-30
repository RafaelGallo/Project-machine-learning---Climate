#!/usr/bin/env python
# coding: utf-8

# # Modelo - Série temporal clima

# Modelo de série temporal - Clima previsão da temperatura  

# Base dados - [Dataset](https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data)

# # Importação das bibliotecas

# In[1]:


# Versão do python

from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[3]:


# Importação das bibliotecas 

# Pandas carregamento csv
import pandas as pd 

# Numpy para carregamento cálculos em arrays multidimensionais
import numpy as np 

# Visualização de dados
import seaborn as sns
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# In[5]:


# Configuração para os gráficos largura e layout dos graficos

plt.rcParams["figure.figsize"] = (25, 20)

plt.style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

m.rcParams['axes.labelsize'] = 25
m.rcParams['xtick.labelsize'] = 25
m.rcParams['ytick.labelsize'] = 25
m.rcParams['text.color'] = 'k'


# # Base de dados

# In[6]:


ds = pd.read_csv('Bases de dados\DailyDelhiClimateTrain.csv')
ds


# # Descrição dados
# 
# - Verificação de linhas colunas informaçãos dos dados e tipos de variáveis. Valores das colunas verficando dados nulos ou vazios.

# In[7]:


# Exibido 5 primeiros dados

ds.head()


# In[8]:


# Exibido 5 últimos dados 

ds.tail()


# In[9]:


# Número de linhas e colunas

ds.shape


# In[10]:


# Verificando informações das variaveis

ds.info()


# In[11]:


# Exibido tipos de dados

ds.dtypes


# In[12]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(ds.shape[0]))
print("Números de colunas: {}" .format(ds.shape[1]))


# In[13]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", ds.isnull().sum().values.sum())
print("\nUnique values :  \n",ds.nunique())


# In[14]:


# Sum() Retorna a soma dos valores sobre o eixo solicitado
# Isna() Detecta valores ausentes

ds.isna().sum()


# In[15]:


# Retorna a soma dos valores sobre o eixo solicitado
# Detecta valores não ausentes para um objeto semelhante a uma matriz.

ds.notnull().sum()


# In[16]:


# Total de número duplicados

ds.duplicated()


# # Estatística descritiva

# In[17]:


# Exibindo estatísticas descritivas visualizar alguns detalhes estatísticos básicos como percentil, média, padrão, etc. 
# De um quadro de dados ou uma série de valores numéricos.

ds.describe().T


# In[19]:


# Gráfico distribuição normal
plt.figure(figsize=(18.2, 8))

ax = sns.distplot(ds['humidity']);
plt.title("Distribuição normal", fontsize=20)
plt.xlabel("Umidade")
plt.ylabel("Total")
plt.axvline(ds['humidity'].mean(), color='b')
plt.axvline(ds['humidity'].median(), color='r')
plt.axvline(ds['humidity'].mode()[0], color='g');
plt.legend(["Media", "Mediana", "Moda"])
plt.show()


# In[20]:


# Matriz correlação de pares de colunas, excluindo NA / valores nulos.
ds.corr()


# In[22]:


# Gráfico da matriz de correlação

plt.figure(figsize=(20,11))
ax = sns.heatmap(ds.corr(), annot=True, cmap='YlGnBu');
plt.title("Matriz de correlação")


# In[34]:


# Matriz de correlação interativa 
fig = px.imshow(ds.iloc[:, 1:].corr())
fig.show()


# # Análise dados

# In[27]:


# Cálculo da média movel

media_humidity = ds[['date', 'humidity']].groupby('date').mean()
media_wind_speed = ds[["date", "wind_speed"]].groupby('date').mean()

print("Média de média Humidity", media_humidity)
print()
print("Média de média media wind speed", media_wind_speed)


# In[29]:


# Gráfico média movel - Humidity e Wind speed

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50.5, 25));
plt.rcParams['font.size'] = '25'

ax1.plot(media_humidity, marker='o', color = 'blue', markersize = 15);
ax1.set(title="Média móvel - Humidity", xlabel = "Date", ylabel = "Humidity")

ax2.plot(media_wind_speed, marker='o', color = 'blue', markersize = 15);
ax2.set(title="Média móvel - Wind speed", xlabel="Date", ylabel="Wind speed")


# In[30]:


# Gráfico da temperatura
fig = px.line(ds, x="date", y="meantemp", title="Temperatura")
fig.show()


# In[31]:


# Gráfico umidade
fig = px.line(ds, x="date", y="humidity", title="Umidade")
fig.show()


# In[32]:


# Gráfico velocidade do vento
fig = px.line(ds, x="date", y="wind_speed", title="Velocidade do vento")
fig.show()


# In[36]:


# Observando total de Temperatura

sns.histplot(ds["meantemp"])
plt.title("Temperatura Cº")
plt.xlabel("Temperatura")
plt.ylabel("Total")


# In[40]:


# Observando total de umidade

sns.histplot(ds["humidity"])
plt.title("Total da umidade")
plt.xlabel("Umidade")
plt.ylabel("Total")


# In[42]:


sns.histplot(ds["wind_speed"])
plt.title("Velocidade vento")
plt.xlabel("Velocidade do vento")
plt.ylabel("Total")


# # Análise de dados = Univariada

# In[43]:


# Fazendo um comparativo dos dados 

ds.hist(bins = 40, figsize=(50.2, 20))
plt.title("Gráfico de histograma")
plt.show()


# # Dataset - Climate

# In[46]:


data = pd.read_csv('Bases de dados\DailyDelhiClimateTrain.csv', index_col='date', parse_dates=True)
data


# In[47]:


data.info()


# In[48]:


data.dtypes


# In[50]:


# Plot total
data.plot(subplots=True, figsize=(20.5, 18))
plt.show()


# In[52]:


# Média dos dados
df = data['meantemp'].resample('MS').mean()

# Gráfico da média - Tempo médio
df.plot(figsize=(14,6))
plt.show()


# # Decomposição Sazonal

# In[53]:


# Importação da biblioteca decomposição sazonal
from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposição aditiva
sd = seasonal_decompose(df, freq = 12)
sd.plot()
plt.show()


# In[54]:


# Padrão de tendência extraído
dt = sd.trend

dt.plot(figsize=(10.5, 8))


# In[56]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(figsize=(20,8))
plot_acf(df, lags=25, zero=False, ax=ax)
plt.show()


# Para evitar a tendência da série temporal, você precisa subtraí-la da média móvel do período de mais do que o período de sazonalidade.

# In[58]:


# Média movel
media_movel = df - df.rolling(20).mean()
media_movel = media_movel.dropna()

# Gráfico - Autocorrelation
fig, ax1 = plt.subplots(figsize=(20.5, 8))
plot_acf(media_movel, lags = 20, zero = False, ax = ax1)
plt.show()


# - O ACF mostra que existe um componente sazonal e, portanto, incluí-lo melhorará suas previsões. Assim, podemos ajustar nossas séries temporais ao modelo sazonal ARIMA

# In[61]:


# SARIMA

# Gráfico 
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20.5, 18))
plot_acf(media_movel, lags=11, zero=False, ax=ax1)

# Gráfico PACF
plot_pacf(media_movel, lags=11, zero=False, ax=ax2)
plt.show()


# In[62]:


# ACF corta no lag 1. então, temos que usar o modelo MA.

media_movel = media_movel.diff(1).diff(12).dropna()
media_movel


# # Modelo - ARIMA

# O ACF não sazonal não mostra nenhum dos padrões usuais dos modelos MA, AR ou ARMA, então não escolhemos nenhum deles. O Seaosnal ACF e PACF parecem um modelo MA(1). Selecionamos o modelo que combina ambos.

# In[63]:


# Modelo ARIMA
from pmdarima.arima import auto_arima

modelo_arima_auto = auto_arima(df,easonal = True, 
                               m = 25, d = 0, D = 1, max_p = 2, max_q = 2,
                               trace = True, error_action ='ignore',
                               suppress_warnings = True)


# - Modelo menor AIC e um pouco diferente anterior a componente sazonal Deltra e 1 ao invés 2

# In[64]:


# Modelo - Auto ARIMA
modelo_arima_auto


# In[65]:


# Modelo aic - Maior que anterior modelo
modelo_arima_auto.aic()


# In[66]:


# Súmario do modelo
print(modelo_arima_auto.summary())


# In[118]:


# Previsão do modelo ARIMA
modelo_arima_auto_pred_1 = modelo_arima_auto.predict(n_periods=100)
modelo_arima_auto_pred_1


# In[123]:


# Data frame da previsão temperatura
pred_1 = pd.DataFrame(modelo_arima_auto_pred_1,columns=['Previsão'])
pred_1


# In[132]:


pd.concat([pred_1.Previsão],axis=1).plot(linewidth=1, figsize=(20,5))
plt.legend(["Previsão"])
plt.xlabel('Previsão da temperatura')
plt.title('Previsão',size=15)
plt.show();


# # Model SARIMA

# - SARIMA(2, 0, 2)x(2, 1, 0, 12) tem um desempenho melhor que outro modelo de ordens e tem baixo valor de AIC.
# - Divida o conjunto de trem e o conjunto de teste do conjunto de dados de trem e ajuste nosso modelo.

# # 6.7) Treino e Teste
# 
# - Treino e teste da base de dados da coluna temperatura

# In[102]:


x = df[:-13] # Variável para treino
y = df[-13:] # Variável para teste


# In[103]:


# Total de linhas e colunas dados variável x
x.shape


# In[104]:


# Total de linhas e colunas dados variável y
y.shape


# # Modelo SARIMA

# In[137]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Modelo SARIMAX
# Ajuste o modelo SARIMAX ao conjunto de treino
modelo_sarimax = SARIMAX(x, order = (2,0, 2), seasonal_order = (2, 1, 0, 12))
modelo_sarimax_fit = modelo_sarimax.fit()
print(modelo_sarimax_fit.summary())


# **Modelo SARIMAX**
# 
# - Prob(Q) é >0,05, então não rejeitamos a hipótese nula de que os resíduos não são correlacionados. Prob(JB) >0,05, então não rejeitamos a hipótese nula de que os resíduos não são normalmente distribuídos Assim, com base no resumo dado, os Resíduos não são correlacionados e normalmente distribuídos

# In[148]:


# 4 gráfico diagnóstico do modelo SARIMA
modelo_sarimax_fit.plot_diagnostics(figsize=(28.5, 25))
plt.show()


# **Standardized residul**
# 
# - O gráfico de resíduos padronizado informa que não há padrões óbvios nos resíduos A curva KDE é muito semelhante à distribuição normal. A maioria dos Datapoints está na linha reta. Além disso, correlações de 95% para atraso maior que um não são significativas Nosso modelo segue um comportamento padronizado. se não, temos que melhorar nosso modelo Prever os valores para o conjunto de teste

# In[151]:


# Prever os valores para o conjunto de teste

x_1 = len(x)
y_2 = len(x) + len(y) - 1

pred = modelo_sarimax_fit.predict(start = x_1, end = y_2)
pred


# In[152]:


# Previsão 

pred = modelo_sarimax_fit.predict(n_periods=100)
pred = pd.DataFrame(pred)
pred


# In[189]:


plt.plot(pred["predicted_mean"])
plt.title("Previsão modelo SARIMA - Temperatura")
plt.xlabel("Temperatura previsão")
predictions.plot(label='Previsão')


# # Métricas para o modelo

# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[182]:


from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(x, pred))
mae = mean_absolute_error(x, pred)
mape = mean_absolute_percentage_error(x, pred)
mse = mean_squared_error(x, pred)
r2 = r2_score(x, pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# # Modelo 02 - Dados test- Daily Delhi Climate Test

# In[241]:


data_test = pd.read_csv('Bases de dados\DailyDelhiClimatetest.csv', index_col='date', parse_dates=True)
data_test.head()


# In[242]:


data_test.tail()


# In[243]:


data_test.shape


# In[244]:


data_test.info()


# In[245]:


data_test_md = data_test['meantemp'].resample('MS').mean().dropna()
data_test_md


# In[246]:


# Modelo SARIMAX dados test

modelo_sarima_2 = SARIMAX(df, order=(2,0,2), seasonal_order=(2,1,0,12))
modelo_sarima_2_fit = modelo_sarima_2.fit()
print(modelo_sarima_2_fit.summary())


# In[247]:


# 4 gráfico diagnóstico do modelo SARIMA

modelo_sarima_2_fit.plot_diagnostics(figsize=(28.5, 25))
plt.show()


# In[248]:


# Exibindo os ultimos dados 
data_test_md.tail()


# In[249]:


# Index dos dados
data_test_md.index[0]


# In[250]:


x_1 = data_test_md.index[0]
y_1 = data_test_md.index[-1]

pred_1 = results_1.predict(start=x_1, end=y_1)
pred_1


# In[251]:


# Gráfico da previsão da temperatura
data_test_md.plot(legend=True,figsize=(25.5, 18))
pred_1.plot()
plt.title("Previsão da temperatura")
plt.legend(["Atual", "Previsão"])
plt.show()


# In[252]:


# Create SARIMA mean forecast
sarima_forecast = results_1.get_forecast(steps=20).predicted_mean
sarima_forecast


# In[253]:


sarima_forecast.plot(label='SARIMAX',figsize=(25.5, 18), legend=True)
test_monthly.plot()
plt.title("Previsão da temperatura - SARIMA Test")
plt.legend(["Previsão"])
plt.show()


# # Métricas do modelo 02 - Dados de test

# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[254]:


rmse = np.sqrt(mean_squared_error(test_monthly, predictions_1))
mae = mean_absolute_error(test_monthly, predictions_1)
mape = mean_absolute_percentage_error(test_monthly, predictions_1)
mse = mean_squared_error(test_monthly, predictions_1)
r2 = r2_score(test_monthly, predictions_1)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[255]:


# Salvando modelo

import pickle

with open('pred.pkl', 'wb') as file:
    pickle.dump(pred, file)
    
with open('# modelo_arima_auto_pred_1', 'wb') as file:
    pickle.dump(modelo_arima_auto_pred_1, file)
    
with open('pred.pkl', 'wb') as file:
    pickle.dump(pred, file)


# # Conclusão do modelo machine learning
# 
# Nesse modelo foi feito dois modelos ARIMA, SARIMA com os dados de treino, teste, modelo teve uma previsão das temperaturas podemos observar a previsão de 2013 ate 2017 nível de temperatura está subindo o nível de Co2. Pela análise a temperatura devido efeito estufa. A conclusão é a temperatura em 4 anos vai subir mais ainda.

# In[ ]:




