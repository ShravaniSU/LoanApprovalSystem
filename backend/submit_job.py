import os
from pathlib import Path

from azure.ai.ml import MLClient, load_job
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent
load_dotenv(backend_dir / ".env")

required_settings = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
)
missing_settings = [name for name in required_settings if not os.environ.get(name)]
if missing_settings:
    raise ValueError(
        f"Missing required Azure settings: {', '.join(missing_settings)}. "
        "Set them in backend/.env or the process environment."
    )

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"],
)

job = load_job(backend_dir / "jobs" / "train_job.yml")
returned_job = ml_client.jobs.create_or_update(job)

print("Job Submitted successfully")
print(f"Job Name : {returned_job.name}")

print("Waiting for Azure ML Training job to complete...")
ml_client.jobs.stream(returned_job.name)