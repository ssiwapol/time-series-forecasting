import argparse

import yaml

from mod import FilePath, ModelValidate, Forecasting, GCStoGBQ

parser = argparse.ArgumentParser()
parser.add_argument('run', action='store', help='Running type')
parser.add_argument('runpath', action='store', help='Configuration path')
parser.add_argument('gbqdest', action='store', nargs='?', help='Google BigQuery destination')
args = parser.parse_args()


if __name__=="__main__":
    with open("config.yaml") as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    fp = FilePath(conf['PLATFORM'], cloud_auth=conf['CLOUD_AUTH'])
    with fp.loadfile(args.runpath) as f:
        r = yaml.load(f, Loader=yaml.Loader)
    # run options
    if args.run == "validate":
        try:
            v = ModelValidate(conf['PLATFORM'], conf['LOG_TAG'], conf['TIMEZONE'], conf['CLOUD_AUTH'])
            v.lg.logtxt("run detail: {}".format(r))
            v.loaddata(r['ACT_PATH'])
            v.validate(r['OUTPUT_DIR'], r['ACT_START'], r['ACT_END'], 
                        r['TEST_START'], r['TEST_END'], r['TEST_PERIOD'], r['TEST_MODEL'], 
                        r['MTH_START'], r['CHUNKSIZE'])
        except Exception as e:
            error_item = "batch: {} | id: {} | testdate: {} | model:{}".format(
                v.runitem.get('batch'), v.runitem.get('id'), v.runitem.get('testdate').strftime("%Y-%m-%d"), v.runitem.get('model'))
            error_txt = "ERROR: {} ({})".format(str(e), error_item)
            v.lg.logtxt(error_txt, error=True)
        if args.gbqdest is not None:
            try:
                gbq = GCStoGBQ(conf['PLATFORM'], conf['LOG_TAG'], conf['CLOUD_AUTH'])
                gbq.listfile(v.output_dir, "output_validate")
                gbq.togbq(args.gbqdest)
            except Exception as e:
                gbq.lg.logtxt(str(e), error=True)
    elif args.run == "forecast":
        try:
            f = Forecasting(conf['PLATFORM'], conf['LOG_TAG'], conf['TIMEZONE'], conf['CLOUD_AUTH'])
            f.lg.logtxt("run detail: {}".format(r))
            f.loaddata(r['ACT_PATH'], r['FCST_PATH'])
            f.forecast(r['OUTPUT_DIR'], r['ACT_START'], r['FCST_START'], r['FCST_MODEL'], 
                       r['MTH_START'], r['TEST_BACK'], r['CHUNKSIZE'])
        except Exception as e:
            error_item = "batch: {} | id: {} | model:{}".format(f.runitem.get('batch'), f.runitem.get('id'), f.runitem.get('model'))
            error_txt = "ERROR: {} ({})".format(str(e), error_item)
            f.lg.logtxt(error_txt, error=True)
        if args.gbqdest is not None:
            try:
                gbq = GCStoGBQ(conf['PLATFORM'], conf['LOG_TAG'], conf['CLOUD_AUTH'])
                gbq.listfile(f.output_dir, "output_forecast")
                gbq.togbq(args.gbqdest)
                gbq.listfile(f.output_dir, "output_selection")
                gbq.togbq("{}_selection".format(args.gbqdest))
            except Exception as e:
                gbq.lg.logtxt(str(e), error=True)
    else:
        pass
