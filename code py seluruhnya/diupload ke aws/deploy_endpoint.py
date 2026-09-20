"""Deploy Credit Score XGBoost model to SageMaker endpoint."""

import boto3
import sagemaker

from sagemaker.sklearn.model import SKLearnModel


# ---------------- CONFIG ----------------

BUCKET = "aws-demo-olive-112"

MODEL_S3_KEY = (
    "credit-score/model.tar.gz"
)

ENDPOINT_NAME = (
    "credit-score-endpoint"
)

REGION = "us-east-1"

INSTANCE_TYPE = (
    "ml.m5.large"
)

FRAMEWORK_VERSION = "1.2-1"

# ----------------------------------------


def get_lab_role_arn():

    iam = boto3.client(
        "iam"
    )

    return iam.get_role(
        RoleName="LabRole"
    )["Role"]["Arn"]


def main():

    boto3.setup_default_session(
        region_name=REGION
    )

    sm_session = (
        sagemaker.Session()
    )

    role_arn = (
        get_lab_role_arn()
    )

    model_s3_uri = (
        f"s3://{BUCKET}/"
        f"{MODEL_S3_KEY}"
    )

    print(
        f"Role: {role_arn}"
    )

    print(
        f"Model URI: {model_s3_uri}"
    )

    print(
        f"Endpoint: {ENDPOINT_NAME}"
    )

    model = SKLearnModel(

        model_data=model_s3_uri,

        role=role_arn,

        entry_point="inference.py",

        source_dir="src",

        framework_version=
        FRAMEWORK_VERSION,

        sagemaker_session=
        sm_session
    )

    print(
        "\nDeploying endpoint..."
    )

    predictor = model.deploy(

        initial_instance_count=1,

        instance_type=
        INSTANCE_TYPE,

        endpoint_name=
        ENDPOINT_NAME
    )

    print(
        "\nEndpoint deployed successfully!"
    )

    print(
        f"\nEndpoint Name:"
        f" {ENDPOINT_NAME}"
    )

    print(
        "\nTo delete later:"
    )

    print(
        "predictor.delete_endpoint()"
    )


if __name__ == "__main__":
    main()