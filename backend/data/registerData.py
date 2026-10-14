from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())


my_data = Data(
    name="loan-prediction-data-mltable",
    version="1",
    path=".",
    type=AssetTypes.MLTABLE,
)
ml_client.data.create_or_update(my_data)