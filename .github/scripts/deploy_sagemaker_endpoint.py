import argparse
import boto3


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a SageMaker endpoint.")
    parser.add_argument("--image-uri", required=True, help="ECR Image URI")
    parser.add_argument("--role-arn", required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint name for deployment")
    parser.add_argument("--model-data-s3-uri", required=True, help="S3 URI to model artifact tar.gz from training")
    parser.add_argument("--instance-type", default="ml.m5.large", help="Instance type for inference")
    parser.add_argument("--instance-count", type=int, default=1, help="Instance count for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    sm_client = boto3.client("sagemaker")

    model_name = args.endpoint_name + "-model"

    # Clean up existing model/endpoint config/endpoint (safe re-deploy)
    for resource, delete_fn in [
        ("endpoint", sm_client.delete_endpoint),
        ("endpoint-config", sm_client.delete_endpoint_config),
        ("model", sm_client.delete_model),
    ]:
        try:
            getattr(sm_client, f"delete_{resource}")(EndpointName=args.endpoint_name)
            print(f"Deleted existing SageMaker {resource}: {args.endpoint_name}")
        except sm_client.exceptions.ClientError as e:
            if "Could not find" not in str(e):
                raise

    # Create Model
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": args.image_uri,
            "ModelDataUrl": args.model_data_s3_uri,
            "Environment": {"ENV": "staging"},  # or prod depending on use
        },
        ExecutionRoleArn=args.role_arn,
    )

    # Create Endpoint Config
    endpoint_config_name = args.endpoint_name + "-config"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": args.instance_count,
                "InstanceType": args.instance_type,
            }
        ],
    )

    # Deploy Endpoint
    sm_client.create_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    print(f"✅ SageMaker endpoint '{args.endpoint_name}' created successfully.")


if __name__ == "__main__":
    main()
