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

load(url("https://userpage.fu-berlin.de/soga/300/30100_data_sets/Earth_Surface_Temperature.RData"))
str(t.global)

t.global  <- apply.yearly(t.global, mean)
t

temp.global  <- t.global["1850/2000", 'Monthly.Anomaly.Global']
temp.global.test  <- t.global["2001/2021", 'Monthly.Anomaly.Global']

plot.zoo(cbind(temp.global,
               temp.global.test),
         plot.type = "single", 
         col = c("black", "gray"), 
         main = 'Earth Surface Temperature Anomalies', 
         ylab = '', xlab = '')
legend('topleft', 
       legend = c('training set 1850-2000',
                  'test set 2001-2016'), 
       col = c("black", "gray"),
       lty = 1, cex = 0.65)


library(forecast)
lambda <- BoxCox.lambda(temp.global)
lambda

# transformed time series
temp.global.BC <- BoxCox(temp.global,lambda)

plot.zoo(cbind(temp.global,
               temp.global.BC),
         col = c("black", "gray"), 
         main = 'Original vs. Box-Cox transformed time series', 
         ylab = '', xlab = '')

library(tseries)
kpss.test(temp.global)

temp.global.diff1 <- diff(temp.global)
kpss.test(temp.global.diff1)

plot(temp.global.diff1)

Acf(temp.global.diff1,
    main = 'ACF for Differenced Series')

library(gridExtra)
## ACF
p1 <- autoplot(Acf(temp.global.diff1, 
                   plot = F, 
                   lag.max = 15)) + ggtitle('ACF')

## PACF
p2 <- autoplot(Acf(temp.global.diff1, 
                   plot = F, 
                   lag.max = 15, 
                   type = 'partial')) + ggtitle('PACF')

grid.arrange(p1, p2, ncol = 2)


# Modelo ARIMA
fit <- Arima(temp.global, order = c(3,1,0))
summary(fit)

print(paste('ARIMA(3,1,0) - AICc: ', round(fit$aicc,2)))

fit.test <- Arima(temp.global, order = c(3,1,1))
print(paste('ARIMA(3,1,1) - AICc: ', round(fit.test$aicc,2)))

fit.test <- Arima(temp.global, order = c(3,1,2))
print(paste('ARIMA(3,1,2) - AICc: ', round(fit.test$aicc,2)))

fit.test <- Arima(temp.global, order = c(2,1,2))
print(paste('ARIMA(2,1,2) - AICc: ', round(fit.test$aicc,2)))

fit.auto <- auto.arima(temp.global, seasonal = F)
summary(fit.auto)

Acf(residuals(fit.auto))

Btest <- Box.test(residuals(fit.auto), 
                  lag = 10, 
                  fitdf = 6, 
                  type = "Ljung")
Btest

# pot forecast
temp.forecast <- forecast(fit.auto, h = 16)
plot(temp.forecast)

# plot test time series of the period 2001-2016
lines(ts(coredata(temp.global.test),
         start = start(temp.forecast$mean)[1],
         frequency = 1), col = 'magenta')

