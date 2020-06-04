# TIME SERIES FORECASTING

# Methods
1. Model Validation (all models) - rolling forecast sales based on historical data (+ external features) for validating result
2. Forecast Production (selected models) - forecast sales based on historical data (+ external features) and perform model selection based on historical forecast

# Models
- Single Exponential Smoothing (Simple Smoothing)
- Double Exponential Smoothing (Holt’s Method)
- Triple Exponential Smoothing (Holt-Winters’ Method)
- ARIMA (Autoregressive Integrated Moving Average)
- ARIMAX (Autoregressive Integrated Moving Average with Explanatory Variable)
- [Prophet by Facebook](https://facebook.github.io/prophet/)
- Linear Regression
- Random Forest
- XGBoost
- LSTM (Long Short-Term Memory)


# REST API
## Request headers
apikey: [AUTH_KEY]

## Request body
```
{
	"run": "[validate/forecast]",
	"path": "[PATH_TO_RUNNING_CONFIGURATION_FILE]"
}
```

# Deployment
1. Build Container
```
git clone https://github.com/ssiwapol/time-series-forecasting.git
cd time-series-forecasting
docker build -t [IMAGE_NAME] .
```

2. Prepare config.yaml and Cloud Authentication File (other directory)

config.yaml
```
PLATFORM: [local/gcp]
APIKEY: [API_KEY]
DEBUG: [TRUE/FALSE]
PORT: [PORT]
TIMEZONE: [TIMEZONE]
LOG_TAG: [LOG_TAG_IN_CLOUD]
CLOUD_AUTH: [CLOUD_AUTH_FILE]
```

3. Run Container
```
docker run --name [CONTAINER_NAME] -v $(pwd):/app/ext -d -p [PORT]:5000 [IMAGE_NAME]
```
