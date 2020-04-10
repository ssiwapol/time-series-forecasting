# TIME SERIES FORECASTING
---

## REST API

### Request URL
[please copy this url](/api)

### Request headers
apikey: [AUTH_KEY]

### Request body

```
{
	"run": "[RUN OPTION(validate/forecast)]",
	"path": "[PATH_TO_RUNNING_CONFIGURATION_FILE]",
	"gbqdest": "[GCP_PROJECT].[GBQ_DATASET].[GBQ_TABLE]"  #optional
}
```

## CONFIGURE RUN
- configuration: validate.yaml
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

- configuration: forecast.yaml
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

### Remarks
- file
```
local path: "[PATH_TO_FILE]/[FILE_NAME]"
gcp path: "gs://[BUCKET_NAME]/[PATH_TO_FILE]/[FILE_NAME]"
```
- directory
```
local path: "[PATH_TO_DIR]/"
gcp path: "gs://[BUCKET_NAME]/[PATH_TO_DIR]/"
```

## MODELS
- model.yaml
```
expo01:
  model: Exponential Smoothing
  name: Single Exponential Smoothing (Simple Smoothing)
  frequency: monthly
  ytype: nominal
  parameter: auto
expo02:
  model: Exponential Smoothing
  name: Double Exponential Smoothing (Holt’s Method)
  frequency: monthly
  ytype: nominal
  parameter: auto
expo03:
  model: Exponential Smoothing
  name: Triple Exponential Smoothing (Holt-Winters’ Method)
  frequency: monthly
  ytype: nominal
  parameter: auto
arima01:
  model: ARIMA / ARIMAX
  name: Auto ARIMA
  frequency: monthly
  ytype: nominal
  parameter: auto
prophet01:
  model: Prophet
  name: Prophet by Facebook
  frequency: daily
  ytype: nominal
  parameter: auto
```

## EXAMPLE FILES
- input: input_actual.csv
```
id,ds,y
a,2014-01-01,100
a,2014-01-02,200
a,2014-01-03,300
a,2014-01-04,400
a,2014-01-05,500
b,2014-01-01,100
b,2014-01-02,200
b,2014-01-03,300
b,2014-01-04,400
b,2014-01-05,500
```
- output: output_validate_0001-0100.csv
```
id,ds,mth,model,actual,forecast,error,time
a,2019-01-01,0,MODEL1,500,500,0,0.01
a,2019-02-01,1,MODEL1,600,600,0,0.01
a,2019-03-01,2,MODEL1,700,700,0,0.01
a,2019-01-01,0,MODEL2,800,800,0,0.01
a,2019-02-01,1,MODEL2,900,900,0,0.01
a,2019-03-01,2,MODEL2,1000,900,0.1,0.01
b,2019-01-01,0,MODEL1,500,500,0,0.01
b,2019-02-01,1,MODEL1,600,600,0,0.01
b,2019-03-01,2,MODEL1,700,700,0,0.01
b,2019-01-01,0,MODEL2,800,800,0,0.01
b,2019-02-01,1,MODEL2,900,900,0,0.01
b,2019-03-01,2,MODEL2,1000,800,0.2,0.01
```
- input: input_forecast.csv
```
id,ds,mth,model,forecast
a,2019-01-01,0,MODEL1,500
a,2019-02-01,1,MODEL1,600
a,2019-03-01,2,MODEL1,700
a,2019-01-01,0,MODEL2,800
a,2019-02-01,1,MODEL2,900
a,2019-03-01,2,MODEL2,900
b,2019-01-01,0,MODEL1,500
b,2019-02-01,1,MODEL1,600
b,2019-03-01,2,MODEL1,700
b,2019-01-01,0,MODEL2,800
b,2019-02-01,1,MODEL2,900
b,2019-03-01,2,MODEL2,800
```
- output: output_forecast_0001-0100.csv
```
id,ds,mth,model,forecast,time
a,2020-01-01,0,MODEL1,500,0.01
a,2020-02-01,1,MODEL1,600,0.01
a,2020-03-01,2,MODEL1,700,0.01
a,2020-01-01,0,MODEL2,800,0.01
a,2020-02-01,1,MODEL2,900,0.01
a,2020-03-01,2,MODEL2,900,0.01
b,2020-01-01,0,MODEL1,500,0.01
b,2020-02-01,1,MODEL1,600,0.01
b,2020-03-01,2,MODEL1,700,0.01
b,2020-01-01,0,MODEL2,800,0.01
b,2020-02-01,1,MODEL2,900,0.01
b,2020-03-01,2,MODEL2,800,0.01
```
- output: output_selection_0001-0100.csv
```
id,ds,mth,model,forecast,error,time
a,2020-01-01,0,MODEL1,500,0,0.01
a,2020-02-01,1,MODEL1,600,0,0.01
a,2020-03-01,2,MODEL1,700,0,0.01
b,2020-01-01,0,MODEL1,500,0,0.01
b,2020-02-01,1,MODEL1,600,0,0.01
b,2020-03-01,2,MODEL1,700,0,0.01
```
