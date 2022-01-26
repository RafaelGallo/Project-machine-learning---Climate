# Série temporal Co2

# Carregando a bibliotecas
library(quantmod)
library(xts)
library(moments)
library(readxl) 
library(foreign)
library(dynlm) 
library(car) 
library(lmtest) 
library(sandwich)
library(fpp2) 
library(tseries) 
library(zoo)
library(xts)
library(forecast) 
library(ggplot2)

# Base de dados online
data <- load(url("https://userpage.fu-berlin.de/soga/300/30100_data_sets/Earth_Surface_Temperature.RData"))
str(t.global)

# Dataset
data <- t.global <- apply.yearly(t.global, mean)
data

# Temperatura Global puxando dados de 1800 até 2021
temp <- t.global["1800/2017", 'Monthly.Anomaly.Global']
temp

# Temperatura global puxando dados de 2001 até 2021
temp_test <- t.global["2001/2016", 'Monthly.Anomaly.Global']
temp_test

# Visualizando os 5 primeiros dados - temp
head(temp)

# Visaulizando os 5 últimos dados - temp
tail(temp)

# Visualizando os 5 primeiros dados - temp_test
head(temp_test)

# Visaulizando os 5 últimos dados - temp_test
tail(temp_test)


#### Série temporal - modelo

library(tseries)

# Stationarity
data <- kpss.test(temp.global)
data

train_temp <- temp.global.diff1 <- diff(temp.global)
train_temp

# Stationarity - test
data_test <- kpss.test(temp.global.diff1)
data_test

# Transformando em séries temporal
lab_data <- BoxCox.lambda(temp.global)
temp_train <- BoxCox(temp.global,lambda)
temp_train

################ Análise explorátoria - Série temporal ################

# Gráfico 1 - Co2 em 2001 até 2016
p_1 <- plot.zoo(cbind(temp,
               temp_test),
         plot.type = "single", 
         col = c("blue", "green"), 
         main = 'Co2', 
         ylab = 'Total', xlab = 'Co2')
legend('topleft', legend = c('Treinamento 1850-2000','Teste 2001-2016'), 
       col = c("black", "gray"),
       lty = 1, cex = 0.65)


# Gráfico 2 - Time series Co2
p_2 <- plot.zoo(cbind(temp.global,
               temp.global.BC),
         col = c("blue", "black"),
         main = "Time series Co2 1850 até 2000",
         ylab = "Total",
         xlab = "Time series Co2")

# Gráfico 3 - Têmperaturas de Jan 1851 até Jan 2001
g3 <- plot(temp.global.diff1)
g3

# Gráfico 4 - ACF
Acf(temp.global.diff1,main = "ACF")

# Gráfico 5 - Têmperaturas pôr ano

library(dygraphs)

dygraph(temp.global, main = " Co2") %>%
  dyAxis("x", drawGrid = TRUE) %>% dyEvent("2000-1-01", "2022", labelLoc = "bottom") %>% 
  dyEvent("2000-1-01", "1800", labelLoc = "bottom") %>% 
  dyEvent("2000-5-01", "2000", labelLoc = "bottom") %>% 
  dyEvent("2017-12-11","2017", labelLoc = "bottom") %>%
  dyOptions(drawPoints = TRUE, pointSize = 2)

# Gráficos com grid
p1 <- autoplot(Acf(temp.global.diff1, 
                   plot = F, 
                   lag.max = 15)) + ggtitle('ACF')

## PACF
p2 <- autoplot(Acf(temp.global.diff1, 
                   plot = F, 
                   lag.max = 15, 
                   type = 'partial')) + ggtitle('PACF')

grid.arrange(p1, p2, ncol = 2)


######### Modelo ARIMA ######### 

# Modelo ARIMA 1
model_arima_fit_1 <-Arima(temp.global, order = c(3, 1, 0))
model_arima_fit_1
summary(model_arima_fit_1)

checkresiduals(model_arima_fit_1)

# Modelo ARIMA 2
model_arima_fit_2 <- Arima(temp.global, order = c(3,1,1))
model_arima_fit_2      
summary(model_arima_fit_2)      

checkresiduals(model_arima_fit_2)

# Modelo ARIMA 3
model_arima_fit_3 <- Arima(temp.global, order = c(3,1,2))
model_arima_fit_3      
summary(model_arima_fit_3)         

checkresiduals(model_arima_fit_3)

# Modelo ARIMA 4
model_arima_fit_4 <- Arima(temp.global, order = c(2,1,2))
model_arima_fit_4      
summary(model_arima_fit_4)

checkresiduals(model_arima_fit_4)

# Modelo auto arima
model_arima_fit <- auto.arima(temp.global, seasonal = F)
model_arima_fit
summary(model_arima_fit)

checkresiduals(model_arima_fit)

# Residuos
g1 <- Acf(residuals(model_arima_fit))
g1

g1_test <- Box.test(residuals(model_arima_fit),
                    lag = 10,
                    fitdf = 6,
                    type = "L")
g1_test

# Previsão temperatura
model_predict <- predict(arima(temp.global, order = c(4,4,5)), n.ahead = 50)
model_predict

# Gráfico 1 - Previsão Co2
autoplot(forecast(model_arima_fit, h=50, title = "Revisão Co2",
                  xlab = "Total",
                  ylab = "Co2"))

# Gráfico 2 - Previsão Co2
plot(forecast(Arima(y = temp.global, order = c(1, 1, 2))))
plot(forecast(Arima(y = temp.global, order = c(3, 3, 4))))


# Gráfico 3 - Previsão das têmperaturas
pred.forecast <- forecast(model_arima_fit, h = 10)
plot(pred.forecast)
lines(ts(coredata(temp.global.test),
         start = start(temp.forecast$mean)[1],
         frequency = 1), col = 'magenta', main = "Co2")

# Salvando os dados em um arquivo .rds
saveRDS(data, file = "CO2.rds")
df = readRDS("CO2.rds")
dir()
head(df)
