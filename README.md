# TIME SERIES FORECASTING

# Method
1. Model Validation (all models) - forecast sales based on historical data and compare with actual sales
2. Forecast Production (selected models) - forecast sales based on historical data and perform model selection based on historical forecast

# REST API
## Request headers
apikey: [AUTH_KEY]

## Request body
```
{
	"run": "[validate/forecast]",
	"path": "[PATH_TO_RUNNING_CONFIGURATION_FILE]",
	"gbqdest": "[GCP_PROJECT].[GBQ_DATASET].[GBQ_TABLE]"  #optional
}
```

# Models
- Single Exponential Smoothing (Simple Smoothing)
- Double Exponential Smoothing (Holt’s Method)
- Triple Exponential Smoothing (Holt-Winters’ Method)
- ARIMA (Autoregressive Integrated Moving Average)
- ARIMAX (Autoregressive Integrated Moving Average with Explanatory Variable)
- [Prophet by Facebook](https://facebook.github.io/prophet/)
- LSTM (Long Short-Term Memory)

# Deployment (Python Environment)
1. Requirement - [python>=3.7](https://www.python.org/), [pip3](https://docs.python.org/3/installing/index.html), [venv](https://docs.python.org/3/tutorial/venv.html)
2. Create config.yaml file
3. Download [deploy_python.sh](deploy/deploy_python.sh)
4. Make it executable
5. Run `./deploy_python.sh`

# Deployment (Docker)
1. Requirement - [docker](https://www.docker.com/)
2. Create config.yaml file
3. Download [deploy_docker.sh](deploy/deploy_docker.sh)
4. Make it executable
5. Run `./deploy_docker.sh`

# Configuration File
- config.yaml
```
PLATFORM: [local/gcp]
APIKEY: [API_KEY]
DEBUG: [TRUE/FALSE]
PORT: [PORT]
TIMEZONE: [TIMEZONE]
LOG_TAG: [LOG_TAG_IN_CLOUD]
CLOUD_AUTH: [CLOUD_AUTH_FILE]
```

- validate.yaml
```
ACT_PATH: [PATH_TO_ACTUAL_SALES_DATA].csv
OUTPUT_DIR: [OUTPUT_DIRECTORY]/
ACT_START: [YYYY-MM-DD]
ACT_END: [YYYY-MM-DD]
TEST_START: [YYYY-MM-DD]
TEST_END: [YYYY-MM-DD]
TEST_PERIOD: [N]
TEST_MODEL: [MODEL1, MODEL2, MODEL3, MODELN]
MTH_START: [N] #default 0
CHUNKSIZE: [N]
CPU: [NO_OF_RUNNING_PROCESSORS]
```

- forecast.yaml
```
ACT_PATH: [PATH_TO_ACTUAL_SALES_DATA].csv
FCST_PATH: [PATH_TO_FORECAST_LOG_DATA].csv
OUTPUT_DIR: [OUTPUT_DIRECTORY]/
ACT_START: [YYYY-MM-DD]
FCST_START: [YYYY-MM-DD]
FCST_MODEL:
  0: [MODEL1, MODEL2, MODEL3, MODELN]
  1: [MODEL1, MODEL2, MODEL3, MODELN]
  2: [MODEL1, MODEL2, MODEL3, MODELN]
  3: [MODEL1, MODEL2, MODEL3, MODELN]
  n: [MODEL1, MODEL2, MODEL3, MODELN]
TEST_BACK: [N]
MTH_START: [N] #default 0
CHUNKSIZE: [N]
CPU: [NO_OF_RUNNING_PROCESSORS]
```
