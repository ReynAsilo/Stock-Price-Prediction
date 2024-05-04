library(quantmod)
library(forecast)
library(tseries)
library(timeSeries)
library(dplyr)
library(tsfknn)
library(prophet)

# Extracting stock data for Amazon
getSymbols("AMZN", from= "2019-01-01", to = "2024-04-01")

head(AMZN)

tail(AMZN)

# Separating Closing Prices of stocks from data
AMZN_CP = AMZN[,4]

# Plotting graph of Amazon Stock Prices to observe the trend
plot(AMZN_CP)

# Plotting the ACF and PACF plot of data
par(mfrow=c(1,2))
Acf(AMZN_CP, main = 'ACF Plot')
Pacf(AMZN_CP, main = 'PACF Plot')

# Plotting Additive and Multiplicative Decomposition
AMZN.ts <- ts(AMZN_CP, start=c(2019,1,1), frequency = 365.25)
AMZN.add  <- decompose(AMZN.ts,type = "additive")
plot(AMZN.add)
AMZN.mult <- decompose(AMZN.ts,type = "multiplicative")
plot(AMZN.mult)

# ADF test on Closing Prices 
print(adf.test(AMZN_CP))

# Splitting into test and train data 
N = length(AMZN_CP)
n = 0.7*N
train = AMZN_CP[1:n, ]
test  = AMZN_CP[(n+1):N,  ]
predlen=length(test)

# Taking log of dataset 
logs=diff(log(AMZN_CP), lag =1)
logs = logs[!is.na(logs)]

# Log returns plot
plot(logs, type='l', main= 'Log Returns Plot')

# ADF test on log of Closing Prices
print(adf.test(logs))

# Histogram and Emperical Distribution
m=mean(logs);
s=sd(logs);
hist(logs, nclass=40, freq=FALSE, main='Closing Price Histogram');
curve(dnorm(x, mean=m,sd=s), from = -0.3, to = 0.2, add=TRUE, col="red")
plot(density(logs), main='Closing Price Empirical Distribution');
curve(dnorm(x, mean=m,sd=s), from = -0.3, to = 0.2, add=TRUE, col="red")

# ACF and PACF of log data 
Acf(logs, main = 'ACF of log data')
Pacf(logs, main = 'PACF of log data')

# Fitting the ARIMA model
# Auto ARIMA with seasonal = FALSE
fit1<-auto.arima(AMZN_CP, seasonal=FALSE)
tsdisplay(residuals(fit1), lag.max = 40, main='(1,1,1) Model Residuals')
fcast1<-forecast(fit1, h=30)
plot(fcast1)
accuracy(fcast1)

# Auto ARIMA with lambda = "auto"
fit2<-auto.arima(AMZN_CP, lambda = "auto")
tsdisplay(residuals(fit2), lag.max = 40, main='(2,1,2) Model Residuals')
fcast2<-forecast(fit2, h=30)
plot(fcast2)
accuracy(fcast2)

# ARIMA model with optimized p,d and q
fit3<-arima(AMZN_CP, order=c(8,2,8))
tsdisplay(residuals(fit3), lag.max = 40, main='(8,2,8) Model Residuals')
fcast3<-forecast(fit3, h=30)
plot(fcast3)
accuracy(fcast3)

# K Nearest Neighbours
df <- data.frame(ds = index(AMZN),
                 y = as.numeric(AMZN[,'AMZN.Close']))

predknn <- knn_forecasting(df$y, h = 30, lags = 1:30, k = 40, msas = "MIMO")
ro <- rolling_origin(predknn)
print(ro$global_accu)
plot(predknn, type="c")

# Prophet
df <- data.frame(ds = index(AMZN),
                 y = as.numeric(AMZN[,'AMZN.Close']))
prophet_model <- prophet(df)
future <- make_future_dataframe(prophet_model, periods = 30)
forecast_prophet <- predict(prophet_model, future)

# Calculating MAPE
# Extracting the last 30 days of actual closing prices
actual <- as.numeric(AMZN_CP[(length(AMZN_CP) - 29):length(AMZN_CP)])

# Extracting the last 30 days of predicted closing prices from the Prophet forecast
predicted <- forecast_prophet$yhat[(nrow(forecast_prophet) - 29):nrow(forecast_prophet)]

mape <- mean(abs((actual - predicted) / actual)) * 100
cat("MAPE for Prophet model:", mape, "%\n")

print(mape)

#Plotting
plot(
  prophetpred,
  forecastprophet,
  uncertainty = TRUE,
  plot_cap = TRUE,
  xlabel = "ds",
  ylabel = "y"
)
dataprediction <- data.frame(forecastprophet$ds,forecastprophet$yhat)
trainlen <- length(AMZN_CP)
dataprediction <- dataprediction[c(1:trainlen),]
prophet_plot_components(prophetpred,forecastprophet)

# Given MAPE values
ARIMA_MAPE <- 1.5486
KNN_MAPE <- 2.4425
Prophet_MAPE <- 2.1009

# Calculate accuracy for each method
ARIMA_accuracy <- 100 - ARIMA_MAPE
KNN_accuracy <- 100 - KNN_MAPE
Prophet_accuracy <- 100 - Prophet_MAPE

# Print accuracies
cat("ARIMA Accuracy:", ARIMA_accuracy, "%\n")
cat("KNN Accuracy:", KNN_accuracy, "%\n")
cat("Prophet Accuracy:", Prophet_accuracy, "%\n")
