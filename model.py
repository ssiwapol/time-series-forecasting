import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from fbprophet import Prophet
import pmdarima as pm


class TimeSeriesForecasting:
    def __init__(self, df, act_st, fcst_st, fcst_pr, col_ds='ds', col_y='y'):
        self.act_st = datetime.datetime.combine(act_st, datetime.datetime.min.time())
        self.fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        self.df = df.rename(columns={col_ds: 'ds', col_y: 'y'})
        self.df = self.df[(self.df['ds']>=self.act_st) & (self.df['ds']<self.fcst_st)]
        self.fcst_pr = fcst_pr
        self.fcst_dt = pd.date_range(start=self.fcst_st, periods=self.fcst_pr, freq='MS')
        self.df_d = self.filldaily(self.df.copy(), self.act_st, self.fcst_st + datetime.timedelta(days=-1))
        self.df_m = self.daytomth(self.df.copy())

    @staticmethod
    def daytomth(df, col_ds='ds', col_y='y'):
        df[col_ds] = df[col_ds].apply(lambda x: x.replace(day=1))
        df = df.groupby([col_ds], as_index=False).agg({col_y: 'sum'})
        df = df[[col_ds, col_y]].sort_values(by=col_ds, ascending=True).reset_index(drop=True)
        return df

    @staticmethod
    def filldaily(df, start, end, col_ds='ds', col_y='y'):
        d = pd.DataFrame(pd.date_range(start=start, end=end), columns=[col_ds])
        df = pd.merge(d, df, on=col_ds, how='left')
        df = df.groupby([col_ds], as_index=False).agg({col_y: 'sum'})
        df = df[[col_ds, col_y]].sort_values(by=col_ds, ascending=True).reset_index(drop=True)
        return df
    
    @staticmethod
    def correctzero(df, col_ds='ds', col_y='y'):
        df['y'] = df['y'].apply(lambda x: 0 if x<0 else x)
        return df
        
    def forecast(self, i, **kwargs):
        fn = getattr(TimeSeriesForecasting, i)
        return fn(self, **kwargs)
    
    def expo01(self):
        x = list(self.df_m['y'])
        m = SimpleExpSmoothing(x).fit(optimized=True)
        r = m.forecast(self.fcst_pr)
        r = pd.DataFrame(zip(self.fcst_dt, r), columns =['ds', 'y'])
        return self.correctzero(r)
    
    def expo02(self, trend='add'):
        x = list(self.df_m['y'])
        m = ExponentialSmoothing(x, trend=trend).fit(optimized=True)
        r = m.forecast(self.fcst_pr)
        r = pd.DataFrame(zip(self.fcst_dt, r), columns =['ds', 'y'])
        return self.correctzero(r)
    
    def expo03(self, trend='add', seasonal='add'):
        x = list(self.df_m['y'])
        m = ExponentialSmoothing(x, trend=trend, seasonal=seasonal, seasonal_periods=12).fit(optimized=True)
        r = m.forecast(self.fcst_pr)
        r = pd.DataFrame(zip(self.fcst_dt, r), columns =['ds', 'y'])
        return self.correctzero(r)
    
    def arima01(self):
        x = list(self.df_m['y'])
        m = pm.auto_arima(x, start_p=1, start_q=1, max_p=12, max_q=12, d=None,
                          m=12, seasonal=True, trace=False,
                          error_action='ignore', suppress_warnings=True, stepwise=True)
        r = m.predict(n_periods=self.fcst_pr)
        r = pd.DataFrame(zip(self.fcst_dt, r), columns =['ds', 'y'])
        return self.correctzero(r)

    def prophet01(self):
        n = 31 * self.fcst_pr
        m = Prophet()
        m.fit(self.df_d)
        f = m.make_future_dataframe(periods=n)
        r = m.predict(f)
        r = r[(r['ds']>=self.fcst_st) & (r['ds']<self.fcst_st + relativedelta(months=+self.fcst_pr))]
        r = r.rename(columns={'yhat': 'y'})
        r = self.daytomth(r)
        return self.correctzero(r)
