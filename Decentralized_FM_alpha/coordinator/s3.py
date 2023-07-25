import boto3
from uuid import uuid4
from loguru import logger
from botocore.exceptions import ClientError

def upload_file(filename, object_name=None):
    if object_name is None:
        object_name = str(uuid4())+".png"
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(filename, 'toma-all', object_name)
    except ClientError as e:
        logger.error(e)
        return False, None
    return True, object_name