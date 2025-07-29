import argparse
import uuid
from sagemaker_pipeline import create_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True, help="ECR image URI")
    parser.add_argument("--role-arn", required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--job-name", required=True, help="Training Job Name")
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Environment")
    parser.add_argument("--wandb-api-key", required=True, help="W&B API Key")
    parser.add_argument("--openai-api-key", required=True, help="Open AI API Key")
    parser.add_argument("--preprocessing-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--preprocessing-instance-count", type=int, default=1)
    parser.add_argument("--training-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--training-instance-count", type=int, default=1)
    parser.add_argument("--evaluation-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--evaluation-instance-count", type=int, default=1)
    parser.add_argument("--deployment-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--lambda-deployment-arn", required=True, help="Lambda deployment function ARN")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline_run_uuid = str(uuid.uuid4())

    pipeline = create_pipeline(role_arn=args.role_arn, pipeline_run_uuid=pipeline_run_uuid)
    pipeline.upsert(role_arn=args.role_arn)

    execution = pipeline.start(
        parameters={
            "PipelineRunID": pipeline_run_uuid,
            "ImageURI": args.image_uri,
            # "RoleARN": args.role_arn,
            # "JobPrefixName": "Text2Cypher",
            "Environment": args.env,
            "WandbApiKey": args.wandb_api_key,
            "OpenAIApiKey": args.openai_api_key,
            "InputDataFolderURI": "s3://bl-portfolio-ml-sagemaker-source-data/notechat-dataset/",
            "PreprocessedOutputS3Uri": f"s3://bl-portfolio-ml-sagemaker-{args.env}/input/preprocessed",
            "TrainingOutputS3Uri": f"s3://bl-portfolio-ml-sagemaker-{args.env}/output/artifacts",
            "EvaluationOutputS3Uri": f"s3://bl-portfolio-ml-sagemaker-{args.env}/output/reports",
            "PreprocessingInstanceType": args.preprocessing_instance_type,
            "PreprocessingInstanceCount": args.preprocessing_instance_count,
            "TrainingInstanceType": args.training_instance_type,
            "TrainingInstanceCount": args.training_instance_count,
            "EvaluationInstanceType": args.evaluation_instance_type,
            "EvaluationInstanceCount": args.evaluation_instance_count,
            # "DeploymentInstanceType": args.deployment_instance_type,
            # "LambdaDeploymentARN": args.lambda_deployment_arn
        }
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        exit(1)
