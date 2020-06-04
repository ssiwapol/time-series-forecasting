# -*- coding: utf-8 -*-
import os
import argparse

import yaml

from mod import Validation, Forecasting
from utils import FilePath

parser = argparse.ArgumentParser()
parser.add_argument('run', action='store', help='Running type')
parser.add_argument('runpath', action='store', help='Configuration path')
args = parser.parse_args()


if __name__=="__main__":
    with open("ext/config.yaml") as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    fp = FilePath(conf['PLATFORM'], cloud_auth=os.path.join("ext", conf['CLOUD_AUTH']))
    with fp.loadfile(args.runpath) as f:
        r = yaml.load(f, Loader=yaml.Loader)
    # run options
    if args.run == "validate":
        v = Validation(conf['PLATFORM'], conf['LOG_TAG'], conf['TIMEZONE'], conf['CLOUD_AUTH'])
        try:
            v.lg.logtxt("run detail: {}".format(r))
            v.loaddata(r['ACT_PATH'], r['EXT_PATH'], r['EXTLAG_PATH'])
            v.validate(r['OUTPUT_DIR'], r['ACT_START'], \
                        r['TEST_START'], r['TEST_PERIOD'], r['TEST_MODEL'], \
                        r['FCST_PERIOD'], r['PERIOD_START'], \
                        r['CHUNKSIZE'], r['CPU'])
        except Exception as e:
            v.lg.logtxt("ERROR: {}".format(str(e)), error=True)
    elif args.run == "forecast":
        f = Forecasting(conf['PLATFORM'], conf['LOG_TAG'], conf['TIMEZONE'], conf['CLOUD_AUTH'])
        try:
            f.lg.logtxt("run detail: {}".format(r))
            f.loaddata(r['ACT_PATH'], r['FCST_PATH'], r['EXT_PATH'], r['EXTLAG_PATH'])
            f.forecast(r['OUTPUT_DIR'], r['ACT_START'], r['FCST_START'], r['FCST_MODEL'], \
                        r['TEST_TYPE'], r['TEST_BACK'], r['PERIOD_START'], \
                        r['CHUNKSIZE'], r['CPU'])
        except Exception as e:
            f.lg.logtxt("ERROR: {}".format(str(e)), error=True)
    else:
        pass
