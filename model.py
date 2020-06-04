# -*- coding: utf-8 -*-
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesForecasting:
    def __init__(self, df, act_st, fcst_st, fcst_pr, ext=None, ext_lag=None, col_ex='id', col_ds='ds', col_y='y'):
        self.act_st = datetime.datetime.combine(act_st, datetime.datetime.min.time())
        self.fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        self.df = df.rename(columns={col_ds: 'ds', col_y: 'y'})
        self.df = self.df[(self.df['ds']>=self.act_st) & (self.df['ds']<self.fcst_st)]
        self.fcst_pr = fcst_pr
        self.dt_m = pd.date_range(start=self.fcst_st, periods=self.fcst_pr, freq='MS')
        self.dt_d = pd.date_range(start=self.fcst_st, periods=self.fcst_pr, freq='D')
        self.df_d = self.filldaily(self.df.copy(), self.act_st, self.fcst_st + datetime.timedelta(days=-1))
        self.df_m = self.daytomth(self.df_d.copy())
        self.ext = ext.rename(columns={col_ex: 'id', col_ds: 'ds', col_y: 'y'}) if ext is not None else None
        self.ext_lag = ext_lag.set_index('id')['lag'].to_dict() if ext is not None else None

    @staticmethod
    def filldaily(df, start, end, col_ds='ds', col_y='y'):
        d = pd.DataFrame(pd.date_range(start=start, end=end), columns=[col_ds])
        df = pd.merge(d, df, on=col_ds, how='left')
        df = df.groupby([col_ds], as_index=False).agg({col_y: 'sum'})
        df = df[[col_ds, col_y]].sort_values(by=col_ds, ascending=True).reset_index(drop=True)
        return df

    @staticmethod
    def daytomth(df, col_ds='ds', col_y='y'):
        df[col_ds] = df[col_ds].apply(lambda x: x.replace(day=1))
        df = df.groupby([col_ds], as_index=False).agg({col_y: 'sum'})
        df = df[[col_ds, col_y]].sort_values(by=col_ds, ascending=True).reset_index(drop=True)
        return df
    
    @staticmethod
    def correctzero(df, col_ds='ds', col_y='y'):
        df['y'] = df['y'].apply(lambda x: 0 if x<0 else x)
        return df
    
    @staticmethod
    def valtogr(df, mth_shift=12, col_ds='ds', col_y='y'):
        df = df.copy()
        df['ds_shift'] = df[col_ds].apply(lambda x: x + relativedelta(months=-mth_shift))
        df = pd.merge(df, df[[col_ds, col_y]].rename(columns={col_ds: 'ds_shift', col_y: 'y_shift'}), how='left', on='ds_shift')
        return list((df[col_y] - df['y_shift']) / df['y_shift'])
    
    @staticmethod
    def grtoval(df, df_act, mth_shift=12, col_ds='ds', col_y='y', col_yact='y'):
        df = df.copy()
        df_act = df_act.copy()
        # map actual data
        df_act['y_shift'] = df_act[col_yact]
        df_act['ds_shift'] = df_act[col_ds].apply(lambda x: x + relativedelta(months=+mth_shift))
        dict_y = pd.Series(df_act['y_shift'].values, index=df_act['ds_shift']).to_dict()
        df['y_shift'] = df[col_ds].map(dict_y)
        # map predict data
        while df['y_shift'].isnull().any():
            df_pd = df.copy()
            df_pd['y_shift'] = (1 + df_pd[col_y]) * df_pd['y_shift']
            df_pd['ds_shift'] = df_pd[col_ds].apply(lambda x: x + relativedelta(months=+mth_shift))
            for i, r in df_pd.dropna().iterrows():
                dict_y[r['ds_shift']] = r['y_shift']
            df['y_shift'] = df[col_ds].map(dict_y)
        return list((1 + df[col_y]) * df['y_shift'])
    
    # create monthly features
    def monthlyfeat(self, df, col, rnn_delay=3):
        df = df.copy()
        df_act = self.df_m.copy()
        df_append = df[(df['ds'] < df_act['ds'].min()) | (df['ds'] > df_act['ds'].max())]
        df_act = df_act.append(df_append, ignore_index = True)
        df_act = self.filldaily(df_act, df_act['ds'].min(), df_act['ds'].max())
        df_act = self.daytomth(df_act)
        # monthly feature
        df_act['month'] = df_act['ds'].apply(lambda x: x.month)
        df_act = pd.get_dummies(df_act, columns = ['month'], drop_first = False)
        df_act['last_month'] = df_act['y'].shift(1)
        df_act['last_year'] = df_act['y'].shift(12)
        df_act['last_momentum'] = (df_act['y'].shift(12) - df_act['y'].shift(13)) / df_act['y'].shift(13)
        df_act['last_momentum'] = df_act['last_momentum'].replace([np.inf, -np.inf], [1, -1])
        df_act['gr'] = self.valtogr(df_act, 12)
        df_act['lastgr_month'] = df_act['gr'].shift(1)
        df_act['lastgr_year'] = df_act['gr'].shift(12)
        # external feature
        rnn_lag = self.ext.groupby(['id'], as_index=False).agg({"ds":"max"})
        rnn_lag = rnn_lag.set_index('id')['ds'].to_dict()
        rnn_lag = {k: max(relativedelta(self.fcst_st, v).months, rnn_delay) for k, v in rnn_lag.items()}
        if self.ext is not None and len(self.ext_lag) > 0:
            for i in self.ext_lag:
                # external features with external lag
                df_ex = self.ext[self.ext['id']==i].copy()
                df_ex['gr'] = self.valtogr(df_ex, 12, 'ds', 'y')
                df_ex['ds'] = df_ex['ds'].apply(lambda x: x + relativedelta(months=self.ext_lag[i]))
                ex_col = 'ex_{}'.format(i)
                gr_col = 'exgr_{}'.format(i)
                df_ex[ex_col] = df_ex['y']
                df_ex[gr_col] = df_ex['gr']
                # external features with rnn lag
                df_rnn = self.ext[self.ext['id']==i].copy()
                df_rnn['ds'] = df_rnn['ds'].apply(lambda x: x + relativedelta(months=rnn_lag[i]))
                rnn_col = 'exrnn_{}'.format(i)
                df_rnn[rnn_col] = df_rnn['y']
                # merge with actual
                df_act = pd.merge(df_act, df_ex[['ds', ex_col, gr_col]], how='left', on='ds')
                df_act = pd.merge(df_act, df_rnn[['ds', rnn_col]], how='left', on='ds')
        df_act = df_act[['ds', 'y'] + [x for x in df_act.columns if x.startswith(tuple(col))]]
        df = df_act[(df_act['ds'] >= df['ds'].min()) & (df_act['ds'] <= df['ds'].max())]
        df = df.sort_values(by='ds', ascending=True).reset_index(drop=True)
        return df
    
    # create daily features
    def dailyfeat(self, df, col, decomp_period=None, decomp_method="additive"):
        df = df.copy()
        df_act = self.df_d.copy()
        df_append = df[(df['ds'] < df_act['ds'].min()) | (df['ds'] > df_act['ds'].max())]
        df_act = df_act.append(df_append, ignore_index = True)
        df_act = self.filldaily(df_act, df_act['ds'].min(), df_act['ds'].max())
        # decompose
        df_act = df_act.set_index('ds')
        decomposition = seasonal_decompose(df_act['y'], model = decomp_method, period = decomp_period)
        df_act['trend'] = decomposition.trend
        df_act['seasonal'] = decomposition.seasonal
        df_act['residual'] = decomposition.resid
        df_act = df_act.reset_index()
        # x, y
        df_act['last_trend'] = df_act['trend'].shift(1)
        df_act['last_seasonal'] = df_act['seasonal'].shift(1)
        df_act['last_residual'] = df_act['residual'].shift(1)
        df_act = df_act[['ds', 'y'] + [x for x in df_act.columns if x.startswith(tuple(col))]]
        df = df_act[(df_act['ds'] >= df['ds'].min()) & (df_act['ds'] <= df['ds'].max())]
        df = df.sort_values(by='ds', ascending=True).reset_index(drop=True)
        return df
        
    ### MODEL ###
    # forecast from selected model
    def forecast(self, i, **kwargs):
        fn = getattr(TimeSeriesForecasting, i)
        return fn(self, **kwargs)
    
    # Exponential Smoothing model
    def expo(self, model):
        r = model.forecast(self.fcst_pr)
        r = pd.DataFrame(zip(self.dt_m, r), columns =['ds', 'y'])
        return self.correctzero(r)
    
    # Single Exponential Smoothing (Simple Smoothing)
    def expo01(self):
        x = list(self.df_m['y'])
        m = SimpleExpSmoothing(x).fit(optimized=True)
        r = self.expo(m)
        return r
    
    # Double Exponential Smoothing (Holt’s Method)
    def expo02(self):
        param = {'trend': 'add'}
        x = list(self.df_m['y'])
        m = ExponentialSmoothing(x, trend=param['trend']).fit(optimized=True)
        r = self.expo(m)
        return r
    
    # Triple Exponential Smoothing (Holt-Winters’ Method)
    def expo03(self):
        param = {'trend': 'add', 'seasonal': 'add'}
        x = list(self.df_m['y'])
        m = ExponentialSmoothing(x, trend=param['trend'], seasonal=param['seasonal'], seasonal_periods=12).fit(optimized=True)
        r = self.expo(m)
        return r
    
    def arima(self, gr, param):
        # input monthly data
        df = self.df_m.copy()
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        # prepare tranining data
        x = df['y'].values
        # fit model
        m = SARIMAX(x, order=(param['p'], param['d'], param['q']), initialization='approximate_diffuse')
        m = m.fit(disp = False)
        # forecast
        r = m.predict(start=df.index[-1] + 1, end=df.index[-1] + self.fcst_pr)
        r = pd.DataFrame(zip(self.dt_m, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_m) if gr else r['y']
        return self.correctzero(r)

    def arima01(self):
        param = {'p': 4, 'd': 1, 'q': 4}
        r = self.arima(gr=False, param=param)
        return r

    def arima02(self):
        param = {'p': 4, 'd': 0, 'q': 4}
        r = self.arima(gr=True, param=param)
        return r

    # ARIMAX 
    def arimax(self, gr, feat, param):
        # if no external features, no forecast result
        if self.ext is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        # input monthly data
        df = self.df_m.copy()
        df = self.monthlyfeat(self.df_m, col=feat)
        df['y'] = self.valtogr(df) if gr else df['y']
        # clean data - drop null from growth calculation, fill 0 when no external data
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.fillna(0).reset_index(drop=True)
        # prepare data
        x = df['y'].values
        ex = df.iloc[:, 2:].values
        # fit model1 with external
        m1 = SARIMAX(x, exog=ex, order=(param['p'], param['d'], param['q']), initialization='approximate_diffuse')
        m1 = m1.fit(disp = False)
        # prepare external data
        df_pred = pd.DataFrame(columns = ['ds', 'y'])
        for i in self.dt_m:
            df_pred = df_pred.append({'ds' : i} , ignore_index=True)
            df_pred = self.monthlyfeat(df_pred, col=feat)
            if np.isnan(list(df_pred.iloc[-1, 2:].values)).any():
                df_pred = df_pred.iloc[:-1, :]
                break
        # forecast model1
        ex_pred = df_pred.iloc[:, 2:].values
        r1 = m1.predict(start=df.index[-1] + 1, end=df.index[-1] + ex_pred.shape[0], exog=ex_pred)
        # model2 (used when there is no external features in future prediction)
        if len(r1) < self.fcst_pr:
            # fit model2 without external
            m2 = SARIMAX(x, order=(param['p'], param['d'], param['q']), initialization='approximate_diffuse')
            m2 = m2.fit(disp = False)
            # forecast model2
            r2 = m2.predict(start=df.index[-1] + ex_pred.shape[0] + 1, end=df.index[-1] + self.fcst_pr)
        else:
            r2 = []
        # summarize result
        r = list(r1) + list(r2)
        r = pd.DataFrame(zip(self.dt_m, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_m) if gr else r['y']
        return self.correctzero(r)

    def arimax01(self):
        gr = False
        feat = ['ex_']
        param = {'p': 4, 'd': 1, 'q': 4}
        r = self.arimax(gr, feat, param)
        return r

    def arimax02(self):
        gr = True
        feat = ['exgr_']
        param = {'p': 4, 'd': 1, 'q': 4}
        r = self.arimax(gr, feat, param)
        return r
    
    def autoarima(self, gr, param):
        # input monthly data
        df = self.df_m.copy()
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        # prepare data
        x = df['y'].values
        # fit model
        try:
            m = pm.arima.AutoARIMA(start_p=param['start_p'], max_p=param['max_p'],
                                   start_q=param['start_q'], max_q=param['max_q'], 
                                   d=param['d'], 
                                   m = param['m'], seasonal=param['seasonal'], 
                                   trace=False, error_action='ignore', suppress_warnings=True, 
                                   stepwise=param['stepwise'])
            m = m.fit(x)
        except Exception:
            m = pm.arima.AutoARIMA()
            m = m.fit(x)
        # forecast
        r = m.predict(n_periods=self.fcst_pr)
        r = pd.DataFrame(zip(self.dt_m, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_m) if gr else r['y']
        return self.correctzero(r)
    
    def autoarima01(self):
        gr = False
        param = {'start_p': 1, 'max_p': 12, 'start_q': 1, 'max_q': 12, 'd': None, 
                 'm': 12, 'seasonal': True, 'stepwise': True}
        r = self.autoarima(gr, param)
        return r
    
    def autoarima02(self):
        gr = True
        param = {'start_p': 1, 'max_p': 12, 'start_q': 1, 'max_q': 12, 'd': None, 
                 'm': 12, 'seasonal': True, 'stepwise': True}
        r = self.autoarima(gr, param)
        return r
    
    def autoarimax(self, gr, feat, param):
        # if no external features, no forecast result
        if self.ext is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        # input monthly data
        df = self.df_m.copy()
        df = self.monthlyfeat(self.df_m, col=feat)
        df['y'] = self.valtogr(df) if gr else df['y']
        # clean data - drop null from growth calculation, fill 0 when no external data
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.fillna(0).reset_index(drop=True)
        # prepare data
        x = df['y'].values
        ex = df.iloc[:, 2:].values
        # fit model1 with external
        try:
            m1 = pm.arima.AutoARIMA(start_p=param['start_p'], max_p=param['max_p'],
                                    start_q=param['start_q'], max_q=param['max_q'], 
                                    d=param['d'], 
                                    m = param['m'], seasonal=param['seasonal'], 
                                    trace=False, error_action='ignore', suppress_warnings=True, 
                                    stepwise=param['stepwise'])
            m1 = m1.fit(x, exogenous=ex)
        except Exception:
            m1 = pm.arima.AutoARIMA()
            m1 = m1.fit(x, exogenous=ex)
        # prepare external data
        df_pred = pd.DataFrame(columns = ['ds', 'y'])
        for i in self.dt_m:
            df_pred = df_pred.append({'ds' : i} , ignore_index=True)
            df_pred = self.monthlyfeat(df_pred, col=feat)
            if np.isnan(list(df_pred.iloc[-1, 2:].values)).any():
                df_pred = df_pred.iloc[:-1, :]
                break
        # forecast model1
        ex_pred = df_pred.iloc[:, 2:].values
        r1 = m1.predict(n_periods=ex_pred.shape[0], exogenous=ex_pred)
        # model2 (used when there is no external features in future prediction)
        if len(r1) < self.fcst_pr:
            # fit model2 without external
            try:
                m2 = pm.arima.AutoARIMA(start_p=param['start_p'], max_p=param['max_p'],
                                        start_q=param['start_q'], max_q=param['max_q'], 
                                        d=param['d'], 
                                        m = param['m'], seasonal=param['seasonal'], 
                                        trace=False, error_action='ignore', suppress_warnings=True, 
                                        stepwise=param['stepwise'])
                m2 = m2.fit(x)
            except Exception:
                m2 = pm.arima.AutoARIMA()
                m2 = m2.fit(x)
            # forecast model2
            r2 = m2.predict(n_periods=self.fcst_pr)
            r2 = r2[-(len(r2) - len(r1)):]
        else:
            r2 = []
        # summarize result
        r = list(r1) + list(r2)
        r = pd.DataFrame(zip(self.dt_m, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_m) if gr else r['y']
        return self.correctzero(r)

    def autoarimax01(self):
        gr = False
        feat = ['ex_']
        param = {'start_p': 1, 'max_p': 12, 'start_q': 1, 'max_q': 12, 'd': None, 
                 'm': 12, 'seasonal': True, 'stepwise': True}
        r = self.autoarimax(gr, feat, param)
        return r
    
    def autoarimax02(self):
        gr = True
        feat = ['exgr_']
        param = {'start_p': 1, 'max_p': 12, 'start_q': 1, 'max_q': 12, 'd': None, 
                 'm': 12, 'seasonal': True, 'stepwise': True}
        r = self.autoarimax(gr, feat, param)
        return r
    
    # Prophet by Facebook 
    def prophet(self, daily=False):
        n = self.fcst_pr if daily else 31 * self.fcst_pr
        m = Prophet()
        m.fit(self.df_d)
        f = m.make_future_dataframe(periods=n)
        r = m.predict(f)
        if daily:
            r = r[(r['ds']>=self.fcst_st)]
            r = r[['ds', 'yhat']]
        else:
            r = r[(r['ds']>=self.fcst_st) & (r['ds']<self.fcst_st + relativedelta(months=+self.fcst_pr))]
            r = self.daytomth(r, col_y='yhat')
        r = r.rename(columns={'yhat': 'y'})
        return self.correctzero(r)

    # Prophet: monthly forecast
    def prophet01(self):
        daily = False
        r = self.prophet(daily)
        return r
    
    # Prophet: daily forecast
    def prophetd01(self):
        daily = True
        r = self.prophet(daily)
        return r

    # Linear Regression: daily forecast
    def lineard(self, param):
        # daily features
        feat = ['trend', 'last_trend']
        df = self.dailyfeat(self.df_d, col = feat, decomp_period = param['decomp_period'], decomp_method = param['decomp_method'])
        # prepare training data
        df = df.dropna().reset_index(drop=True)
        X_trn = df['last_trend'].values.reshape(-1, 1)
        y_trn = df['trend']
        # fit model
        m = LinearRegression()
        m.fit(X_trn, y_trn)
        # if extract trend is True, use exact date of trend to forecast
        if param['exact_trend']:
            dt1 = df['ds'].max() + datetime.timedelta(days=+1)
        # if extract trend is False, use recent trend to forecast
        else:
            dt1 = self.dt_d[0]
        # predict first y
        x_pred1 = df.loc[df.index[-1], 'trend']
        y_pred1 = m.predict(x_pred1.reshape(-1, 1))[0]
        r = [{'ds': dt1, 
              'y': y_pred1,
              'x': x_pred1
             }]
        # predict the rest
        dt_list = pd.date_range(start=(dt1 + datetime.timedelta(days=+1)), end=self.dt_d.max(), freq='D')
        for i in dt_list:
            x_pred = r[-1]['y']
            y_pred = m.predict(x_pred.reshape(-1, 1))[0]
            r.append({'ds': i, 
                      'y': y_pred,
                      'x': x_pred
                     })
        r = pd.DataFrame(r)
        r = r[r['ds']>=self.fcst_st]
        r = r[['ds', 'y']].reset_index(drop=True)
        return self.correctzero(r)
    
    # Linear Regression: daily forecast, use recent trend to forecast
    def lineard01(self):
        param = {'exact_trend': False, 'decomp_period': 4, 'decomp_method': 'additive'}
        r = self.lineard(param)
        return r
    
    # Linear Regression: daily forecast, use exact trend by date to forecast
    def lineard02(self):
        param = {'exact_trend': True, 'decomp_period': None, 'decomp_method': 'additive'}
        r = self.lineard(param)
        return r
    
    # Random Forest model without external features
    def randomforest(self, gr, feat, param):
        # prepare data
        df = self.monthlyfeat(self.df_m, col=feat)
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        sc = StandardScaler()
        X_trn = df.iloc[:, 2:]
        X_trn = sc.fit_transform(X_trn)
        y_trn = df.iloc[:, 1].values
        # fit model
        m = RandomForestRegressor(n_estimators = param['n_estimators'], min_samples_split = param['min_samples_split'], 
                                   max_depth = param['max_depth'], max_features = param['max_features'], random_state=1)
        m.fit(X_trn, y_trn)
        # forecast each month
        r = pd.DataFrame(columns = ['ds', 'y', 'y_pred'])
        for i in self.dt_m:
            r = r.append({'ds' : i} , ignore_index=True)
            df_pred = self.monthlyfeat(r, col=feat)
            x = df_pred.iloc[-1, 2:].values
            # predict m
            x_pred = sc.transform(x.reshape(1, -1))
            y_pred = m.predict(x_pred)
            r.iloc[-1, 2] = y_pred
            r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        r = r[['ds', 'y']]
        return self.correctzero(r)
 
    # Random Forest without external: forecast y
    def randomforest01(self):
        gr = False
        feat = ['month_', 'last_']
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        r = self.randomforest(gr, feat, param)
        return r
    
    # Random Forest without external: forecast growth
    def randomforest02(self):
        gr = True
        feat = ['month_', 'lastgr_']
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        r = self.randomforest(gr, feat, param)
        return r
    
    # Random Forest model with external features
    def randomforestx(self, gr, feat, param):
        # if no external features, no forecast result
        if self.ext is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data for model1
        df = self.monthlyfeat(self.df_m, col=feat)
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        sc = StandardScaler()
        X_trn = df.iloc[:, 2:]
        X_trn = sc.fit_transform(X_trn)
        y_trn = df.iloc[:, 1].values
        # fit model1
        m = RandomForestRegressor(n_estimators = param['n_estimators'], min_samples_split = param['min_samples_split'], 
                                  max_depth = param['max_depth'], max_features = param['max_features'], random_state=1)
        m.fit(X_trn, y_trn)
        # forecast each month
        r = pd.DataFrame(columns = ['ds', 'y', 'y_pred'])
        for i in self.dt_m:
            r = r.append({'ds' : i} , ignore_index=True)
            df_pred = self.monthlyfeat(r, col=feat)
            x = df_pred.iloc[-1, 2:].values
            # if no external features for prediction, break and do model2
            if np.isnan(list(x)).any():
                r = r.iloc[:-1, :]
                break
            x_pred = sc.transform(x.reshape(1, -1))
            y_pred = m.predict(x_pred)
            r.iloc[-1, 2] = y_pred
            r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        # model2 (used when there is no external features in future prediction)
        if len(r) < self.fcst_pr:
            # prepare data for model2
            feat = [x for x in feat if not x.startswith("ex")]
            df = self.monthlyfeat(self.df_m, col=feat)
            df['y'] = self.valtogr(df) if gr else df['y']
            df = df.dropna().reset_index(drop=True)
            sc = StandardScaler()
            X_trn = df.iloc[:, 2:]
            X_trn = sc.fit_transform(X_trn)
            y_trn = df.iloc[:, 1].values
            # fit model2
            m = RandomForestRegressor(n_estimators = param['n_estimators'], min_samples_split = param['min_samples_split'], 
                                      max_depth = param['max_depth'], max_features = param['max_features'], random_state=1)
            m.fit(X_trn, y_trn)
            # forecast the rest months
            for i in self.dt_m[len(r):]:
                r = r.append({'ds' : i} , ignore_index=True)
                df_pred = self.monthlyfeat(r, col=feat)
                x = df_pred.iloc[-1, 2:].values
                x_pred = sc.transform(x.reshape(1, -1))
                y_pred = m.predict(x_pred)
                r.iloc[-1, 2] = y_pred
                r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        # summarize result
        r = r[['ds', 'y']]
        return self.correctzero(r)

    # Random Forest with external: forecast y
    def randomforestx01(self):
        gr = False
        feat = ['month_', 'last_', 'ex_']
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        r = self.randomforestx(gr, feat, param)
        return r
    
    # Random Forest with external: forecast growth
    def randomforestx02(self):
        gr = True
        feat = ['month_', 'lastgr_', 'exgr_']
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        r = self.randomforestx(gr, feat, param)
        return r
    
    # XGBoost model without external features
    def xgboost(self, gr, feat, param):
        # prepare data
        df = self.monthlyfeat(self.df_m, col=feat)
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        sc = StandardScaler()
        X_trn = df.iloc[:, 2:]
        X_trn = sc.fit_transform(X_trn)
        y_trn = df.iloc[:, 1].values
        # fit model
        m = XGBRegressor(learning_rate = param['learning_rate'], n_estimators = param['n_estimators'], 
                         max_dept = param['max_dept'], min_child_weight = param['min_child_weight'])
        m.fit(X_trn, y_trn)
        # forecast each month
        r = pd.DataFrame(columns = ['ds', 'y', 'y_pred'])
        for i in self.dt_m:
            r = r.append({'ds' : i} , ignore_index=True)
            df_pred = self.monthlyfeat(r, col=feat)
            x = df_pred.iloc[-1, 2:].values
            # predict m
            x_pred = sc.transform(x.reshape(1, -1))
            y_pred = m.predict(x_pred)
            r.iloc[-1, 2] = y_pred
            r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        r = r[['ds', 'y']]
        return self.correctzero(r)
    
    # XGBoost without external: forecast y
    def xgboost01(self):
        gr = False
        feat = ['month_', 'last_']
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        r = self.xgboost(gr, feat, param)
        return r
    
    # XGBoost without external: forecast growth
    def xgboost02(self):
        gr = True
        feat = ['month_', 'lastgr_']
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        r = self.xgboost(gr, feat, param)
        return r

    # XGBoost model with external features
    def xgboostx(self, gr, feat, param):
        # if no external features, no forecast result
        if self.ext is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data for model1
        df = self.monthlyfeat(self.df_m, col=feat)
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        sc = StandardScaler()
        X_trn = df.iloc[:, 2:]
        X_trn = sc.fit_transform(X_trn)
        y_trn = df.iloc[:, 1].values
        # fit model1
        m = XGBRegressor(learning_rate = param['learning_rate'], n_estimators = param['n_estimators'], 
                         max_dept = param['max_dept'], min_child_weight = param['min_child_weight'])
        m.fit(X_trn, y_trn)
        # forecast each month
        r = pd.DataFrame(columns = ['ds', 'y', 'y_pred'])
        for i in self.dt_m:
            r = r.append({'ds' : i} , ignore_index=True)
            df_pred = self.monthlyfeat(r, col=feat)
            x = df_pred.iloc[-1, 2:].values
            # if no external features for prediction, break and do model2
            if np.isnan(list(x)).any():
                r = r.iloc[:-1, :]
                break
            x_pred = sc.transform(x.reshape(1, -1))
            y_pred = m.predict(x_pred)
            r.iloc[-1, 2] = y_pred
            r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        # model2 (used when there is no external features in future prediction)
        if len(r) < self.fcst_pr:
            # prepare data for model2
            feat = [x for x in feat if not x.startswith("ex")]
            df = self.monthlyfeat(self.df_m, col=feat)
            df['y'] = self.valtogr(df) if gr else df['y']
            df = df.dropna().reset_index(drop=True)
            sc = StandardScaler()
            X_trn = df.iloc[:, 2:]
            X_trn = sc.fit_transform(X_trn)
            y_trn = df.iloc[:, 1].values
            # fit model2
            m = XGBRegressor(learning_rate = param['learning_rate'], n_estimators = param['n_estimators'], 
                             max_dept = param['max_dept'], min_child_weight = param['min_child_weight'])
            m.fit(X_trn, y_trn)
            # forecast the rest months
            for i in self.dt_m[len(r):]:
                r = r.append({'ds' : i} , ignore_index=True)
                df_pred = self.monthlyfeat(r, col=feat)
                x = df_pred.iloc[-1, 2:].values
                x_pred = sc.transform(x.reshape(1, -1))
                y_pred = m.predict(x_pred)
                r.iloc[-1, 2] = y_pred
                r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        # summarize result
        r = r[['ds', 'y']]
        return self.correctzero(r)

    # XGBoost with external: forecast y
    def xgboostx01(self):
        gr = False
        feat = ['month_', 'last_', 'ex_']
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        r = self.xgboostx(gr, feat, param)
        return r
    
    # XGBoost with external: forecast growth
    def xgboostx02(self):
        gr = True
        feat = ['month_', 'lastgr_', 'exgr_']
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        r = self.xgboostx(gr, feat, param)
        return r
    
    # LSTM (Long Short-Term Memory) with or without external
    def lstm(self, gr, feat, param, rolling=False):
        # set parameter
        forward = 1 if rolling else self.fcst_pr
        feat = ['month_', 'last_year', 'last_momentum'] if rolling else feat
        look_back = param['look_back']
        n_val = param['n_val']
        # prepare data
        df = self.monthlyfeat(self.df_m, col=feat, rnn_delay=3)
        df['y'] = self.valtogr(df) if gr else df['y']
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.iloc[len([x for x in df['last_year'] if pd.isnull(x)]):, :] if 'last_year' in df.columns else df
        df = df.iloc[len([x for x in df['last_momentum'] if pd.isnull(x)]):, :] if 'last_momentum' in df.columns else df
        df = df.fillna(0).sort_values(by='ds', ascending=True).reset_index(drop=True)
        # check if data is sufficient to run train and validate
        len_val = n_val + look_back + forward - 1
        if len(df) <= len_val:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare training data and validate data
        trn = df.iloc[:-n_val, :]
        val = df.iloc[-len_val:, :]
        # scaler
        sc = MinMaxScaler()
        trn = trn.iloc[:, 1:]
        trn = sc.fit_transform(trn)
        val = val.iloc[:, 1:]
        val = sc.transform(val)
        # transform to rnn format
        X_trn, y_trn = [], []
        for i in range(len(trn) - look_back - forward + 1):
            X_trn.append(trn[i:(i + look_back), :])
            y_trn.append(trn[(i + look_back):(i + look_back + forward), 0])
        X_trn, y_trn = np.array(X_trn), np.array(y_trn)
        X_val, y_val = [], []
        for i in range(len(val) - look_back - forward + 1):
            X_val.append(val[i:(i + look_back), :])
            y_val.append(val[(i + look_back):(i + look_back + forward), 0])
        X_val, y_val = np.array(X_val), np.array(y_val)
        # lstm model
        m = Sequential()
        m.add(LSTM(param['node1'], return_sequences = True, input_shape = (look_back, X_trn.shape[2])))
        m.add(LSTM(param['node2']))
        m.add(Dense(forward, activation = param['activation']))
        m.compile(loss = param['loss'], optimizer = param['optimizer'])
        # set callbacks
        callbacks = [EarlyStopping(monitor = 'val_loss', patience = param['patience'], mode = 'min', restore_best_weights = True)]
        # fit model
        m.fit(X_trn, y_trn, epochs = param['epochs'], batch_size = param['batch_size'], validation_data = (X_val, y_val), callbacks = callbacks, verbose=0)
        if rolling:
            # if rolling, forecast each month by rolling data
            r = self.df_m[['ds', 'y']]
            r['y_pred'] = self.valtogr(r) if gr else r['y']
            r = r.iloc[len([x for x in r['y_pred'] if pd.isnull(x)]):, :]
            for i in self.dt_m:
                df_pred = self.monthlyfeat(r, col=feat, rnn_delay=3)
                df_pred['y'] = self.valtogr(df_pred) if gr else df_pred['y']
                x_pred = df_pred.iloc[-look_back:, :]
                x_pred = x_pred.iloc[:, 1:]
                x_pred = sc.transform(x_pred)
                x_pred = x_pred.reshape(1, look_back, -1)
                y_pred = m.predict(x_pred)
                y_pred = y_pred.reshape(forward, 1)
                y_pred = np.concatenate((y_pred, np.zeros([forward, X_trn.shape[2]-1])), axis=1)
                y_pred = sc.inverse_transform(y_pred)
                y_pred = list(y_pred[:, 0])
                r = r.append({'ds' : i, 'y_pred': y_pred[0]} , ignore_index=True)
                r['y'] = self.grtoval(r, self.df_m, col_y='y_pred', col_yact='y') if gr else r['y_pred']
            r = r.iloc[-self.fcst_pr:, :][['ds', 'y']].reset_index(drop=True)
        else:
            # prepare data for predict
            df_pred = df.iloc[-look_back:, :]
            x_pred = df_pred.iloc[:, 1:]
            x_pred = sc.transform(x_pred)
            x_pred = x_pred.reshape(1, look_back, -1)
            # batch predict and transform data
            y_pred = m.predict(x_pred)
            y_pred = y_pred.reshape(forward, 1)
            y_pred = np.concatenate((y_pred, np.zeros([forward, X_trn.shape[2]-1])), axis=1)
            y_pred = sc.inverse_transform(y_pred)
            y_pred = list(y_pred[:, 0])
            r = pd.DataFrame(zip(self.dt_m, y_pred), columns =['ds', 'y'])
            r['y'] = self.grtoval(r, self.df_m) if gr else r['y']
        return self.correctzero(r)

    # LSTM without external: forecast y
    def lstm01(self):
        gr = False
        feat = ['month_', 'last_year', 'last_momentum']
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': 24, 'n_val': 12}
        rolling = False
        r = self.lstm(gr, feat, param, rolling)
        return r

    # LSTM without external: forecast growth
    def lstm02(self):
        gr = True
        feat = ['month_', 'last_year', 'last_momentum']
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': 24, 'n_val': 12}
        rolling = False
        r = self.lstm(gr, feat, param, rolling)
        return r

    # LSTM without external and rolling forecast: forecast y
    def lstmr01(self):
        gr = False
        feat = ['month_', 'last_year', 'last_momentum']
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': 24, 'n_val': 12}
        rolling = True
        r = self.lstm(gr, feat, param, rolling)
        return r

    # LSTM without external and rolling forecast: forecast growth
    def lstmr02(self):
        gr = True
        feat = ['month_', 'last_year', 'last_momentum']
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': 24, 'n_val': 12}
        rolling = True
        r = self.lstm(gr, feat, param, rolling)
        return r

    # LSTM with external: forecast y
    def lstmx01(self):
        gr = False
        feat = ['month_', 'last_year', 'last_momentum', 'exrnn_']
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': 24, 'n_val': 12}
        rolling = False
        if self.ext is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        r = self.lstm(gr, feat, param, rolling)
        return r

    # LSTM with external: forecast growth
    def lstmx02(self):
        gr = True
        feat = ['month_', 'last_year', 'last_momentum', 'exrnn_']
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': 24, 'n_val': 12}
        rolling = False
        if self.ext is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        r = self.lstm(gr, feat, param, rolling)
        return r
