import boto3

def lambda_handler(event, context):
    sagemaker_client = boto3.client('sagemaker')
    model_name = event['model_name']
    image_uri = event['image_uri']
    model_data = event['model_data']
    role = event['role']
    endpoint_name = event['endpoint_name']
    instance_type = event['instance_type']

    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data,
        },
        ExecutionRoleArn=role
    )

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=model_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': instance_type,
        }]
    )

    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=model_name
    )

    return {'status': 'Deployment triggered', 'endpoint_name': endpoint_name}
