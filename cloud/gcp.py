import io

from google.cloud import storage
from google.cloud import logging
from google.cloud.logging.handlers import CloudLoggingHandler


def gcs_download(gcspath, service_json=None):
    if service_json is None:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(service_json)
    bucket = client.get_bucket(gcspath.split("/")[2])
    fullpath = '/'.join(gcspath.split("/")[3:])
    blob = storage.Blob(fullpath, bucket)
    byte_stream = io.BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    return byte_stream

def gcs_upload(file, gcspath, service_json=None, contenttype=None, public=False):
    if service_json is None:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(service_json)
    bucket = client.get_bucket(gcspath.split("/")[2])
    fullpath = '/'.join(gcspath.split("/")[3:])
    blob = storage.Blob(fullpath, bucket)
    if contenttype is None:
        blob.upload_from_file(file)
    else:
        blob.upload_from_file(file, content_type=contenttype)
    if public:
        blob.make_public()

def gcs_mkdir(gcspath, service_json=None):
    if service_json is None:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(service_json)
    bucket = client.get_bucket(gcspath.split("/")[2])
    fullpath = '/'.join(gcspath.split("/")[3:])
    blob = storage.Blob(fullpath, bucket)
    blob.upload_from_string('')
    
def gcs_exists(gcspath, service_json=None):
    if service_json is None:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(service_json)
    bucket = client.get_bucket(gcspath.split("/")[2])
    fullpath = '/'.join(gcspath.split("/")[3:])
    blob = storage.Blob(fullpath, bucket)
    return blob.exists()

def gcs_listfile(gcspath, subfolder=False, service_json=None):
    if service_json is None:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(service_json)
    bucket = gcspath.split("/")
    bucket_name = bucket[2]
    prefix = "/".join(bucket[3:])
    if subfolder:
        blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter=None)
        files = ["gs://" + "/".join([bucket_name, x.name]) for x in blobs if not x.name.endswith("/")]
    else:
        blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
        files = ["gs://" + "/".join([bucket_name, x.name]) for x in blobs if not x.name.endswith("/")]
    return files

def gcl_logging(gclname, service_json=None):
    if service_json is None:
        client = logging.Client()
    else:
        client = logging.Client.from_service_account_json(service_json)
    handler = CloudLoggingHandler(client, name=gclname)
    return handler
