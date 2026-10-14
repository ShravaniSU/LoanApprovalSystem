from azure.ai.ml import MLClient, load_job
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

job = load_job("./jobs/train_job.yml")
returned_job = ml_client.jobs.create_or_update(job)

print("Job Submitted successfully")
print(f"Job Name : {returned_job.name}")