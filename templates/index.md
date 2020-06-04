# TIME SERIES FORECASTING
---

## REST API

### Request URL
[please copy this url](/api)

### Request headers
apikey: [AUTH_KEY]

### Request body

```json
{
	"run": "[RUN OPTION(validate/forecast)]",
	"path": "[PATH_TO_RUNNING_CONFIGURATION_FILE].yaml"
}
```


## MODELS
MODEL | DESCRIPTION | FEATURES | INPUT | OUTPUT | YTYPE
--- | --- | --- | --- | --- | ---
expo01 | Single Exponential Smoothing (Simple Smoothing) | - | D/M | M | N
expo02 | Double Exponential Smoothing (Holt’s Method) | - | D/M | M | N
expo03 | Triple Exponential Smoothing (Holt-Winters’ Method) | - | D/M | M | N
arima01 | ARIMA model with fixed parameter | - | D/M | M | N
arima02 | ARIMA model with fixed parameter | - | D/M | M | GR
arimax01 | ARIMAX model with fixed parameter and external features | - | D/M | M | N
arimax02 | ARIMAX model with fixed parameter and external features | - | D/M | M | GR
autoarima01 | ARIMA model with optimal parameter | - | D/M | M | N
autoarima02 | ARIMA model with optimal parameter | - | D/M | M | GR
autoarimax01 | ARIMAX model with optimal parameter and external features | - | D/M | M | N
autoarimax02 | ARIMAX model with optimal parameter and external features | - | D/M | M | GR
prophet01 | Prophet by Facebook | - | D | M | N
prophetd01 | Prophet by Facebook | - | D | D | N
lineard01 | Linear Regression used latest trend to date | trend | D | D | N
lineard02 | Linear Regression used exact trend to date | trend | D | D | N
randomforest01 | Random Forest | month, last month, last year, last momentum | D/M | M | N
randomforest02 | Random Forest | month, last month, last year | D/M | M | GR
randomforestx01 | Random Forest with external features | month, last month, last year, last momentum | D/M | M | N
randomforestx02 | Random Forest with external features | month, last month, last year | D/M | M | GR
xgboost01 | XGBoost | month, last month, last year, last momentum | D/M | M | N
xgboost02 | XGBoost | month, last month, last year | D/M | M | GR
xgboostx01 | XGBoost with external features | month, last month, last year, last momentum | D/M | M | N
xgboostx02 | XGBoost with external features | month, last month, last year | D/M | M | GR
lstm01 | Long Short-Term Memory | month, last year, last momentum | D/M | M | N
lstm02 | Long Short-Term Memory | month, last year, last momentum | D/M | M | GR
lstmr01 | Long Short-Term Memory with rolling forecast | month, last year, last momentum | D/M | M | N
lstmr02 | Long Short-Term Memory with rolling forecast | month, last year, last momentum | D/M | M | GR
lstmx01 | Long Short-Term Memory with external | month, last year, last momentum | D/M | M | N
lstmx02 | Long Short-Term Memory with external | month, last year, last momentum | D/M | M | GR

### Keywords
- INPUT/OUTPUT D - Daily
- INPUT/OUTPUT M - Monthly
- YTYPE N - forecast nominal values
- YTYPE GR - forecast year-on-year growth


## CONFIGURE RUN
- configuration: validate.yaml
```yaml
# actual sales data (daily/monthly)
ACT_PATH: [PATH_TO_FILE].csv
# external data (monthly)
EXT_PATH: [PATH_TO_FILE].csv
# external lagging of each item
EXTLAG_PATH: [PATH_TO_FILE].csv
# output directory
OUTPUT_DIR: [PATH_TO_DIRECTORY]
# actual sales start date
ACT_START: [YYYY-MM-DD]
# test start date
TEST_START: [YYYY-MM-DD]
# number of rolling period to test (months)
TEST_PERIOD: [N]
# list of model to test
TEST_MODEL: [[MODEL1], [MODEL2], [MODEL3], [MODELN]]
# number of periods to forecast for each rolling
FCST_PERIOD: [N]
# starting period for each forecast (default 0/1)
PERIOD_START: [N]
# number of item to validate for each chunk
CHUNKSIZE: [N]
# number of running processors
CPU: [N]
```

- configuration: forecast.yaml
```yaml
# actual sales data (daily/monthly)
ACT_PATH: [PATH_TO_FILE].csv
# forecasting log for walk-forward validation
FCST_PATH: [PATH_TO_FILE].csv
# external data (monthly)
EXT_PATH: [PATH_TO_FILE].csv
# external lagging of each item
EXTLAG_PATH: [PATH_TO_FILE].csv
# output directory
OUTPUT_DIR: [PATH_TO_DIRECTORY]
# actual sales start date
ACT_START: [YYYY-MM-DD]
# forecast date
FCST_START: [YYYY-MM-DD]
# forecast model options for each periods
FCST_MODEL:
  0: [MODEL1, MODEL2, MODEL3, MODELN]
  1: [MODEL1, MODEL2, MODEL3, MODELN]
  2: [MODEL1, MODEL2, MODEL3, MODELN]
  3: [MODEL1, MODEL2, MODEL3, MODELN]
  n: [MODEL1, MODEL2, MODEL3, MODELN]
# type of testing back error by month or day
TEST_TYPE: [TEST_TYPE(monthly/daily)]
# number of months to test back
TEST_BACK: [N]
# starting period for each forecast (default 0/1)
PERIOD_START: [N]
# number of item to validate for each chunk
CHUNKSIZE: [N]
# number of running processors
CPU: [N]
```

### How to specify path?

- file
    - local path: [PATH_TO_FILE]/[FILE_NAME]
    - gcp path: gs://[BUCKET_NAME]/[PATH_TO_FILE]/[FILE_NAME]
  
- directory
    - local path: [PATH_TO_DIR]/
    - gcp path: gs://[BUCKET_NAME]/[PATH_TO_DIR]/


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
- input: input_external.csv
```
id,ds,y
x1,2014-01-01,100
x1,2014-01-02,200
x1,2014-01-03,300
x1,2014-01-04,400
x1,2014-01-05,500
x2,2014-01-01,100
x2,2014-01-02,200
x2,2014-01-03,300
x2,2014-01-04,400
x2,2014-01-05,500
```
- input: input_external_lag.csv
```
y_id,ext_id,lag
a,x1,2
b,x2,4
```
- input: input_forecast.csv
```
id,ds,dsr,period,model,forecast,time
a,2019-01-01,2019-01-01,0,MODEL1,500,0.01
a,2019-02-01,2019-01-01,1,MODEL1,600,0.01
a,2019-03-01,2019-01-01,2,MODEL1,700,0.01
a,2019-01-01,2019-01-01,0,MODEL2,800,0.01
a,2019-02-01,2019-01-01,1,MODEL2,900,0.01
a,2019-03-01,2019-01-01,2,MODEL2,900,0.01
b,2019-01-01,2019-01-01,0,MODEL1,500,0.01
b,2019-02-01,2019-01-01,1,MODEL1,600,0.01
b,2019-03-01,2019-01-01,2,MODEL1,700,0.01
b,2019-01-01,2019-01-01,0,MODEL2,800,0.01
b,2019-02-01,2019-01-01,1,MODEL2,900,0.01
b,2019-03-01,2019-01-01,2,MODEL2,800,0.01
```
- output: output_validate_0001-0100.csv
```
id,ds,dsr,period,model,forecast,time
a,2019-01-01,2019-01-01,0,MODEL1,500,0.01
a,2019-02-01,2019-01-01,1,MODEL1,600,0.01
a,2019-03-01,2019-01-01,2,MODEL1,700,0.01
a,2019-01-01,2019-01-01,0,MODEL2,800,0.01
a,2019-02-01,2019-01-01,1,MODEL2,900,0.01
a,2019-03-01,2019-01-01,2,MODEL2,900,0.01
b,2019-01-01,2019-01-01,0,MODEL1,500,0.01
b,2019-02-01,2019-01-01,1,MODEL1,600,0.01
b,2019-03-01,2019-01-01,2,MODEL1,700,0.01
b,2019-01-01,2019-01-01,0,MODEL2,800,0.01
b,2019-02-01,2019-01-01,1,MODEL2,900,0.01
b,2019-03-01,2019-01-01,2,MODEL2,800,0.01
```
- output: output_forecast_0001-0100.csv
```
id,ds,dsr,period,model,forecast,time
a,2020-01-01,2020-01-01,0,MODEL1,500,0.01
a,2020-02-01,2020-01-01,1,MODEL1,600,0.01
a,2020-03-01,2020-01-01,2,MODEL1,700,0.01
a,2020-01-01,2020-01-01,0,MODEL2,800,0.01
a,2020-02-01,2020-01-01,1,MODEL2,900,0.01
a,2020-03-01,2020-01-01,2,MODEL2,900,0.01
b,2020-01-01,2020-01-01,0,MODEL1,500,0.01
b,2020-02-01,2020-01-01,1,MODEL1,600,0.01
b,2020-03-01,2020-01-01,2,MODEL1,700,0.01
b,2020-01-01,2020-01-01,0,MODEL2,800,0.01
b,2020-02-01,2020-01-01,1,MODEL2,900,0.01
b,2020-03-01,2020-01-01,2,MODEL2,800,0.01
```
- output: output_selection_0001-0100.csv
```
id,ds,dsr,period,model,forecast,error,time
a,2020-01-01,2020-01-01,0,MODEL1,500,0,0.01
a,2020-02-01,2020-01-01,1,MODEL1,600,0,0.01
a,2020-03-01,2020-01-01,2,MODEL1,700,0,0.01
b,2020-01-01,2020-01-01,0,MODEL1,500,0,0.01
b,2020-02-01,2020-01-01,1,MODEL1,600,0,0.01
b,2020-03-01,2020-01-01,2,MODEL1,700,0,0.01
```
