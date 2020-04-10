import io
import os
import logging
import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing

from pytz import timezone
import numpy as np
import pandas as pd

from model import TimeSeriesForecasting
from cloud import gcp


class FilePath:
    def __init__(self, platform, cloud_auth="None"):
        self.platform = platform
        self.cloud_auth = cloud_auth

    def loadfile(self, path):
        if self.platform == "gcp":
            return gcp.gcs_download(path, self.cloud_auth)
        else:
            return open(path)
        
    def writecsv(self, df, path):
        output_file = io.StringIO()
        df.to_csv(output_file, encoding='utf-8', index=False)
        output_file.seek(0)
        if self.platform == "gcp":
            gcp.gcs_upload(output_file, path, self.cloud_auth)
        else:
            with open(path, mode='w') as f:
                f.write(output_file.getvalue())
                
    def mkdir(self, path):
        if self.platform == "gcp":
            gcp.gcs_mkdir(path, self.cloud_auth)
        else:
            try:
                os.mkdir(path)
            except OSError:
                pass

    def listfile(self, path):
        if self.platform == "gcp":
            files = gcp.gcs_listfile(path, subfolder=False, service_json=self.cloud_auth)
        else:
            files = []
            for r, d, f in os.walk(path):
                for file in f:
                    files.append(os.path.join(r, file))
        return files
    
    def fileexists(self, path):
        if self.platform == "gcp":
            return gcp.gcs_exists(path, service_json=self.cloud_auth)
        else:
            return os.path.isfile(path)


class Logging:
    def __init__(self, platform, logname, logtag="time-series-forecasting", cloud_auth="None"):
        self.platform = platform
        self.cloud_auth = cloud_auth
        self.logger = logging.getLogger(logname)
        self.logger.setLevel(logging.INFO)
        self.log_capture_string = io.StringIO()
        ch = logging.StreamHandler(self.log_capture_string)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)
        if self.platform == "gcp":
            self.logger.addHandler(gcp.gcl_logging(logtag, self.cloud_auth))
        else:
            pass

    def logtxt(self, txt, error=False):
        if error:
            self.logger.error(txt)
        else:
            self.logger.info(txt)
        if self.platform == "gcp":
            pass
        else:
            print(txt)
            
    def writelog(self, path):
        log_file = io.StringIO()
        log_file.write(self.log_capture_string.getvalue())
        log_file.seek(0)
        if self.platform == "gcp":
            gcp.gcs_upload(log_file, path, self.cloud_auth)
        else:
            with open(path, mode='w') as f:
                f.write(log_file.getvalue())


def mape(act, pred):
    if act == 0 and pred == 0:
        return 0
    elif act == 0 and pred != 0:
        return 1
    else:
        try:
            return np.abs((act - pred) / act)
        except Exception:
            return None


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class ModelValidate:
    def __init__(self, platform, logtag, tz, cloud_auth="None"):
        self.fp = FilePath(platform, cloud_auth)
        self.lg = Logging(platform, "validate", logtag, cloud_auth)
        self.lg.logtxt("[START VALIDATION]")
        self.tz = tz
    
    def loaddata(self, act_path, col_id='id', col_ds='ds', col_y='y'):
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        df = pd.read_csv(self.fp.loadfile(act_path), parse_dates=['ds'], date_parser=dateparse)
        self.df = df.rename(columns={col_id: 'id', col_ds: 'ds', col_y: 'y'})
        self.lg.logtxt("load data: {}".format(act_path))

    def validate_byitem(self, x, df, act_st, act_end, test_date, test_pr, test_model, mth_st, batch_no, lg):
        df_i = df[df['id']==x][['ds', 'y']]
        df_i = TimeSeriesForecasting.filldaily(df_i, act_st, act_end)
        df_r = pd.DataFrame()
        for d in test_date:
            model = TimeSeriesForecasting(df_i, act_st, d, test_pr)
            for m in test_model:
                runitem = {"batch": batch_no, "id": x, "testdate": d, "model": m}
                try:
                    st_time = datetime.datetime.now()
                    r = model.forecast(m)
                    r = r.rename(columns={'y': 'forecast'})
                    r['time'] = (datetime.datetime.now() - st_time).total_seconds()
                    r['id'] = x
                    r['model'] = m
                    r['mth'] = np.arange(mth_st, len(r)+mth_st)
                    act = TimeSeriesForecasting.daytomth(df_i.copy())
                    r = pd.merge(r, act, on=['ds'], how='left')
                    r = r.rename(columns={'y': 'actual'})
                    r['error'] = r.apply(lambda x: mape(x['actual'], x['forecast']), axis=1)
                    r = r[['id', 'ds', 'mth', 'model', 'actual', 'forecast', 'error', 'time']]
                    df_r = df_r.append(r, ignore_index = True)
                except Exception as e:
                    error_item = "batch: {} | id: {} | testdate: {} | model:{}".format(
                        runitem.get('batch'), runitem.get('id'), runitem.get('testdate').strftime("%Y-%m-%d"), runitem.get('model'))
                    error_txt = "ERROR: {} ({})".format(str(e), error_item)
                    lg.logtxt(error_txt, error=True)
        return df_r

    def validate(self, output_dir, act_st, act_end, test_st, test_end, test_pr, test_model, mth_st, chunk_sz, cpu):
        # make output directory
        output_dir = "{}validate_{}/".format(output_dir, datetime.datetime.now(timezone(self.tz)).strftime("%Y%m%d-%H%M%S"))
        self.output_dir = output_dir
        self.fp.mkdir(output_dir)
        self.lg.logtxt("create output directory: {}".format(output_dir))
        self.fp.writecsv(self.df, "{}input.csv".format(output_dir))
        self.lg.logtxt("write input file: {}input.csv".format(output_dir))
        # set parameter
        items = self.df['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        test_date = [x.to_pydatetime() for x in pd.date_range(start=test_st, end=test_end, freq='MS')]
        act_end = act_end + relativedelta(months=+1) + relativedelta(days=-1)
        self.lg.logtxt("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # loop by chunk
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            if cpu_count==1:
                for r in [self.validate_byitem(x, self.df, act_st, act_end, test_date, test_pr, test_model, mth_st, i, self.lg) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.validate_byitem, [[x, self.df, act_st, act_end, test_date, test_pr, test_model, mth_st, i, self.lg] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
            # write csv file
            output_path = "{}output_validate_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_fcst, output_path)
            self.lg.logtxt("write output file ({}/{}): {}".format(i, n_chunk, output_path))
        self.lg.logtxt("[END VALIDATION]")
        self.lg.writelog("{}logfile.log".format(output_dir))


class Forecasting:
    def __init__(self, platform, logtag, tz, cloud_auth=None):
        self.fp = FilePath(platform, cloud_auth)
        self.lg = Logging(platform, "forecast", logtag, cloud_auth)
        self.lg.logtxt("[START FORECASTING]")
        self.tz = tz
    
    def loaddata(self, act_path, fcst_path, col_id='id', col_ds='ds', col_y='y', col_mth='mth', col_model='model', col_fcst='forecast'):
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        df_act = pd.read_csv(self.fp.loadfile(act_path), parse_dates=['ds'], date_parser=dateparse)
        df_fcstlog = pd.read_csv(self.fp.loadfile(fcst_path), parse_dates=['ds'], date_parser=dateparse)
        self.df_act = df_act.rename(columns={col_id:'id', col_ds:'ds', col_y:'y'})
        self.df_fcstlog = df_fcstlog.rename(columns={col_id:'id', col_ds:'ds', col_mth:'mth', col_model:'model', col_fcst:'forecast'})
        self.lg.logtxt("load data: {} | {}".format(act_path, fcst_path))

    def forecast_byitem(self, x, df, act_st, fcst_st, fcst_pr, model_list, mth_st, batch_no, lg):
        df = df[df['id']==x].copy()
        df = df[(df['ds']>=act_st) & (df['ds']<fcst_st)].copy()
        model = TimeSeriesForecasting(df, act_st, fcst_st, fcst_pr)
        df_r = pd.DataFrame()
        for m in model_list:
            try:
                runitem = {"batch": batch_no, "id": x, "model": m}
                st_time = datetime.datetime.now()
                r = model.forecast(m)
                r = r.rename(columns={'y': 'forecast'})
                r['time'] = (datetime.datetime.now() - st_time).total_seconds()
                r['id'] = x
                r['model'] = m
                r['mth'] = np.arange(mth_st, len(r)+mth_st)
                r = r[['id', 'ds', 'mth', 'model', 'forecast', 'time']]
                df_r = df_r.append(r, ignore_index = True)
            except Exception as e:
                error_item = "batch: {} | id: {} | model:{}".format(runitem.get('batch'), runitem.get('id'), runitem.get('model'))
                error_txt = "ERROR: {} ({})".format(str(e), error_item)
                lg.logtxt(error_txt, error=True)
        return df_r

    def rankmodel_byitem(self, x, df_act, df_fcstlog, fcst_model, act_st, fcst_st, test_st):
        df_act = df_act[df_act['id']==x].copy()
        df_fcstlog = df_fcstlog[df_fcstlog['id']==x].copy()
        df_act = df_act[(df_act['ds']>=act_st) & (df_act['ds']<fcst_st)].copy()
        df_fcstlog = df_fcstlog[(df_fcstlog['ds']>=test_st) & (df_fcstlog['ds']<fcst_st)].copy()
        df_rank = df_fcstlog.copy()
        # select only in config file
        df_rank['val'] = df_rank['mth'].map(fcst_model)
        df_rank = df_rank[df_rank['val'].notnull()].copy()
        df_rank['val'] = df_rank.apply(lambda x: True if x['model'] in x['val'] else False, axis=1)
        df_rank = df_rank[df_rank['val']==True].copy()
        # calculate error comparing with actual
        act = TimeSeriesForecasting.daytomth(df_act)
        df_rank = pd.merge(df_rank, act, on=['ds'], how='left')
        df_rank = df_rank.rename(columns={'y': 'actual'})
        df_rank['error'] = df_rank.apply(lambda x: mape(x['actual'], x['forecast']), axis=1)
        df_rank = df_rank[df_rank['error'].notnull()]
        df_rank = df_rank.groupby(['id', 'mth', 'model'], as_index=False).agg({"error":"mean"})
        # ranking error
        df_rank['rank'] = df_rank.groupby("mth")["error"].rank(method="first", ascending=True)
        return df_rank
        
    def forecast(self, output_dir, act_st, fcst_st, fcst_model, mth_st, test_bck, chunk_sz, cpu):
        # make output directory
        output_dir = "{}forecast_{}/".format(output_dir, datetime.datetime.now(timezone(self.tz)).strftime("%Y%m%d-%H%M%S"))
        self.output_dir = output_dir
        self.fp.mkdir(output_dir)
        self.lg.logtxt("create output directory: {}".format(output_dir))
        self.fp.writecsv(self.df_act, "{}input_actual.csv".format(output_dir))
        self.fp.writecsv(self.df_fcstlog, "{}input_forecast.csv".format(output_dir))
        self.lg.logtxt("write input file: {}input_actual.csv | {}input_forecast.csv".format(output_dir, output_dir))
        self.runitem = {}
        # set parameter
        items = self.df_act['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        act_st = datetime.datetime.combine(act_st, datetime.datetime.min.time())
        fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        test_st = fcst_st + relativedelta(months=-test_bck)
        fcst_pr = len(fcst_model.keys())
        model_list = list(set(b for a in fcst_model.values() for b in a))
        self.lg.logtxt("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # forecast
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            df_rank = pd.DataFrame()
            if cpu_count==1:
                for r in [self.forecast_byitem(x, self.df_act, act_st, fcst_st, fcst_pr, model_list, mth_st, i, self.lg) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
                for r in [self.rankmodel_byitem(x, self.df_act, self.df_fcstlog, fcst_model, act_st, fcst_st, test_st) for x in c]:
                    df_rank = df_rank.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.forecast_byitem, [[x, self.df_act, act_st, fcst_st, fcst_pr, model_list, mth_st, i, self.lg] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.rankmodel_byitem, [[x, self.df_act, self.df_fcstlog, fcst_model, act_st, fcst_st, test_st] for x in c]):
                    df_rank = df_rank.append(r, ignore_index = True)
            # find best model
            df_sl = pd.merge(df_fcst, df_rank, on=['id', 'mth', 'model'], how='left')
            df_sl = df_sl[df_sl['rank']==1].copy()
            df_sl = df_sl[['id', 'ds', 'mth', 'model', 'forecast', 'error', 'time']]
            df_sl = df_sl.sort_values(by=['id', 'ds'], ascending=True).reset_index(drop=True)
            # write forecast result
            fcst_path = "{}output_forecast_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_fcst, fcst_path)
            # write selection result
            selection_path = "{}output_selection_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_sl, selection_path)
            self.lg.logtxt("write output file ({}/{}): {} | {}".format(i, n_chunk, fcst_path, selection_path))
        self.lg.logtxt("[END FORECAST]")
        self.lg.writelog("{}logfile.log".format(output_dir))


class GCStoGBQ:
    def __init__(self, platform, logtag, cloud_auth=None):
        self.fp = FilePath(platform, cloud_auth)
        self.lg = Logging(platform, "gcs-gbq", logtag, cloud_auth)
        self.platform = platform
        self.cloud_auth = cloud_auth
        self.lg.logtxt("[START LOAD DATA TO GBQ]")
        
    def listfile(self, path, prefix):
        files = self.fp.listfile(path)
        self.files = [x for x in files if x.split("/")[-1].startswith(prefix)]
        self.lg.logtxt("list files in directory: {}".format(path))
        self.lg.logtxt("total files ({}): {}".format(prefix, len(self.files)))
    
    def togbq(self, gbq_tb):
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        for i, x in enumerate(self.files, 1):
            df = pd.read_csv(self.fp.loadfile(x), parse_dates=['ds'], date_parser=dateparse)
            if i==1:
                gcp.gbq_upload(df, dest=gbq_tb, service_json=self.cloud_auth, action="replace")
            else:
                gcp.gbq_upload(df, dest=gbq_tb, service_json=self.cloud_auth, action="append")
            self.lg.logtxt("write table ({}/{}): {}".format(i, len(self.files), gbq_tb))
        self.lg.logtxt("[END LOAD DATA TO GBQ]")
