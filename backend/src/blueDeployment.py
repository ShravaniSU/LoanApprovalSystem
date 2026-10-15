import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)

# 1. Connect to your AzureML workspace using environment variables from GitHub Secrets
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"],
)

# 2. Define a unique endpoint name
endpoint_name = "loan-prediction-endpoint"

# Create the online endpoint (if it doesn't already exist)
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Online endpoint for loan prediction model",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 3. Automatically fetch the latest version of the registered model
model = ml_client.models.get(name="loan_prediction_model", label="latest")
inference_env = ml_client.environments.get(name="loan-prediction-inference-env", version="2")

# 4. Configure the deployment using your existing environment and score.py
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=inference_env,
    code_configuration=CodeConfiguration(
        code="src",       # Folder where your score.py is located
        scoring_script="score.py"
    ),
    instance_type="Standard_F2s_v2",
    instance_count=1,
)

# 5. Create or update the deployment
print("Starting deployment (this may take a few minutes)...")
ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
print("Deployment completed successfully!")

# 6. Route 100% of traffic to the blue deployment
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()