from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model import Model
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.lambda_step import LambdaOutput
from sagemaker.lambda_helper import Lambda


session = PipelineSession()

pipeline_run_id_param = ParameterString(name="PipelineRunID", default_value="no-pipeline-id")

source_data_folder_uri = ParameterString(name="InputDataFolderURI", default_value="s3://bl-portfolio-ml-sagemaker-source-data/notechat-dataset/")
role = ParameterString(name="RoleARN", default_value="")
job_prefix_name = ParameterString(name="JobPrefixName", default_value="Project")
env_param = ParameterString(name="Environment", default_value="dev")
wandb_api_key = ParameterString(name="WandbApiKey", default_value="")
open_ai_key = ParameterString(name="OpenAIApiKey", default_value="")
image_uri = ParameterString(name="ImageURI", default_value="")
preprocessing_instance_type = ParameterString(name="PreprocessingInstanceType", default_value="ml.m5.large")
preprocessing_instance_count = ParameterInteger(name="PreprocessingInstanceCount", default_value=1)
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
evaluation_instance_type = ParameterString(name="EvaluationInstanceType", default_value="ml.m5.large")
evaluation_instance_count = ParameterInteger(name="EvaluationInstanceCount", default_value=1)
deployment_instance_type = ParameterString(name="DeploymentInstanceType", default_value="ml.m5.large")

preprocessed_data_output_uri = ParameterString(
    name="PreprocessedOutputS3Uri",
    default_value=f"s3://bl-portfolio-ml-sagemaker-dev/input/preprocessed"
)

training_artifacts_output_uri = ParameterString(
    name="TrainingOutputS3Uri",
    default_value=f"s3://bl-portfolio-ml-sagemaker-dev/output/artifacts"
)

evaluation_reports_output_uri = ParameterString(
    name="EvaluationOutputS3Uri",
    default_value=f"s3://bl-portfolio-ml-sagemaker-dev/output/reports"
)

evaluation_report_path = ParameterString(
    name="EvaluationOutputPath",
    default_value=f"s3://bl-portfolio-ml-sagemaker-dev/output/reports/eval_metrics.json"
)

lambda_deployment_arn = ParameterString(name="LambdaDeploymentARN", default_value="")

preprocessing_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=preprocessing_instance_count,
    instance_type=preprocessing_instance_type,
    volume_size_in_gb=30,
    env={
        "ENV": env_param,
        "WANDB_API_KEY": wandb_api_key,
        "PIPELINE_RUN_ID": pipeline_run_id_param,
    },
)

preprocessing_step = ProcessingStep(
    name=f"{job_prefix_name.to_string()}-{env_param.to_string()}-DataPreProcessing",
    processor=preprocessing_processor,
    code="scripts/preprocessing.py",
    source_dir=".",
    job_arguments=[
        "--config-path", "src/text2cypher/finetuning/config",
        "--config-name", f"config.{env_param.to_string()}"
    ],
    inputs=[
        ProcessingInput(
            source=source_data_folder_uri,
            destination="/opt/ml/processing/input/source-data",
            input_name="source-data"
        )
    ],
    outputs=[
        ProcessingOutput(
            source=f"/opt/ml/processing/output/preprocessed-{env_param.to_string()}",
            destination=preprocessed_data_output_uri,
            output_name="training-data"
        )
    ],
)

training_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=training_instance_count,
    instance_type=training_instance_type,
    volume_size_in_gb=30,
    env={
        "ENV": env_param,
        "WANDB_API_KEY": wandb_api_key,
        "PIPELINE_RUN_ID": pipeline_run_id_param,
    },
)

training_step = ProcessingStep(
    name=f"{job_prefix_name.to_string()}-{env_param.to_string()}-TrainModel",
    processor=training_processor,
    code="scripts/train.py",
    source_dir=".",
    job_arguments=[
        "--config-path", "src/text2cypher/finetuning/config",
        "--config-name", f"config.{env_param.to_string()}"
    ],
    inputs=[
        ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["training-data"].S3Output.S3Uri,
            destination=f"/opt/ml/processing/input/preprocessed-{env_param.to_string()}",
            input_name="training-data"
        )
    ],
    outputs=[
        ProcessingOutput(
            source=f"/opt/ml/processing/output/model-artifacts-{env_param.to_string()}",
            destination=training_artifacts_output_uri,
            output_name="model-artifacts"
        )
    ]
)

evaluation_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=evaluation_instance_count,
    instance_type=evaluation_instance_type,
    volume_size_in_gb=30,
    env={
        "ENV": env_param,
        "WANDB_API_KEY": wandb_api_key,
        "OPENAI_API_KEY": open_ai_key,
        "PIPELINE_RUN_ID": pipeline_run_id_param,
    },
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation-metrics",
    path=evaluation_report_path
)

evaluation_step = ProcessingStep(
    name=f"{job_prefix_name}-{env_param}-EvaluateModel",
    processor=evaluation_processor,
    code="scripts/evaluate_model.py",
    source_dir=".",
    job_arguments=[
        "--config-path", "src/text2cypher/finetuning/config",
        "--config-name", f"config.{env_param.to_string()}"
    ],
    inputs=[
        ProcessingInput(
            source=training_step.properties.ProcessingOutputConfig.Outputs["model-artifacts"].S3Output.S3Uri,
            destination=f"/opt/ml/processing/input/model-artifacts-{env_param.to_string()}",
            input_name="model-artifacts"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/reports",
            destination=evaluation_reports_output_uri,
            output_name="evaluation-metrics",
        )
    ],
    #evaluate_models does not outputs nothing, results are logged into weights and biases.
)

model = Model(
    image_uri=image_uri,
    model_data=training_step.properties.ProcessingOutputConfig.Outputs["model-artifacts"].S3Output.S3Uri,
    role=role,
    sagemaker_session=session,
)

model_package_group_name = f"NoteChat-{pipeline_run_id_param}"

register_model_step = ModelStep(
    name="RegisterNoteChatModel",
    step_args=model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[deployment_instance_type],
        transform_instances=[deployment_instance_type],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
        description=f"Registered by pipeline run {pipeline_run_id_param.to_string()}",
    ),
)

deploy_model_step = LambdaStep(
    name="DeployNoteChatModel",
    lambda_func=Lambda(session, function_arn=lambda_deployment_arn),
    inputs={
        "model_name": f"notechat-{pipeline_run_id_param.to_string()}",
        "image_uri": image_uri,
        "model_data": training_step.properties.ProcessingOutputConfig.Outputs["model-artifacts"].S3Output.S3Uri,
        "role": role,
        "endpoint_name": f"notechat-endpoint-{pipeline_run_id_param.to_string()}",
        "instance_type": deployment_instance_type
    },
    outputs=[
        LambdaOutput(output_name="status", output_type="String"),
        LambdaOutput(output_name="endpoint_name", output_type="String")
    ],
)

cond_gte = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="val_loss"
    ),
    right=2.0,
)

condition_step = ConditionStep(
    name="CheckValLossCondition",
    conditions=[cond_gte],
    if_steps=[register_model_step, deploy_model_step],
    else_steps=[],
)

# --- 5️⃣ Pipeline Assembly ---
pipeline = Pipeline(
    name="NoteChatPipeline",
    parameters=[
        source_data_folder_uri,
        preprocessed_data_output_uri,
        training_artifacts_output_uri,
        evaluation_reports_output_uri,
        pipeline_run_id_param,
        role,
        job_prefix_name,
        env_param,
        wandb_api_key,
        open_ai_key,
        image_uri,
        preprocessing_instance_type,
        preprocessing_instance_count,
        training_instance_type,
        training_instance_count,
        evaluation_instance_type,
        evaluation_instance_count,
        deployment_instance_type,
    ],
    steps=[preprocessing_step, training_step, evaluation_step, condition_step],
)


pipeline.upsert(role_arn=role)
