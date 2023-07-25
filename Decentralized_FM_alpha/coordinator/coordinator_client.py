import os
import json
import requests
import boto3
import os
from uuid import uuid4
from botocore.exceptions import ClientError
from loguru import logger
import netifaces as ni


class LocalCoordinatorClient:
    def __init__(self,
                 working_directory: str,
                 coordinator_url: str
                 ) -> None:
        self.working_directory = working_directory
        self.coordinator_url = coordinator_url
        self.global_coordinator_url = "https://planetd.shift.ml/jobs"

    def load_input_job_from_dfs(self, job_id):
        doc_path = os.path.join(self.working_directory,
                                'input_' + job_id + '.json')
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as infile:
                doc = json.load(infile)
            return doc
        else:
            return None

    def save_output_job_to_dfs(self, result_doc):
        output_filename = 'output_' + result_doc['_id'] + '.json'
        output_path = os.path.join(self.working_directory, output_filename)
        with open(output_path, 'w') as outfile:
            json.dump(result_doc, outfile)
        input_filename = 'input_' + result_doc['_id'] + '.json'
        input_path = os.path.join(self.working_directory, input_filename)
        assert os.path.exists(input_path)
        os.remove(input_path)

    def notify_inference_join(self, job_id, netname='access'):
        ip = ni.ifaddresses(netname)[ni.AF_INET][0]['addr']
        return requests.post(self.coordinator_url+f"/rank/"+str(job_id), json={"ip": ip}).json()

    def update_status(self, job_id, new_status, returned_payload=None):
        return requests.post(self.coordinator_url+f"/update_status/{job_id}", json={
            "status": new_status,
            "returned_payload": returned_payload
        })

    def upload_file(self, filename, object_name=None):
        if object_name is None:
            object_name = str(uuid4())+".png"
        s3_client = boto3.client('s3')
        try:
            response = s3_client.upload_file(filename, 'toma-all', object_name)
        except ClientError as e:
            logger.error(e)
            return False, None
        return True, object_name
    
    def fetch_instructions(self, model_name, rank):
        while True:
            try:
                return requests.get(self.coordinator_url + f"/instructions/{model_name}/{rank}").json()
            except Exception as e:
                print('<fetch_instructions> Exception', e)
                raise e

    def update_status_global_coordinator(self, job_id, new_status, returned_payload=None):
        return requests.patch(self.global_coordinator_url + f"/{job_id}", json={
            "type": "general",
            "source": "dalle",
            "status": new_status,
            "returned_payload": returned_payload
        })