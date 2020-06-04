# -*- coding: utf-8 -*-
import io
import os
import logging

import numpy as np

from cloud import gcp


class FilePath:
    def __init__(self, platform, cloud_auth=None):
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
    def __init__(self, platform, logname, logtag="time-series-forecasting", cloud_auth=None):
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


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


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
