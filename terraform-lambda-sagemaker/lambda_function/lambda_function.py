import boto3

def lambda_handler(event, context):
    sagemaker_client = boto3.client('sagemaker')

    model_package_arn = event['model_package_arn']
    endpoint_name = event['endpoint_name']
    instance_type = event['instance_type']
    role = event['role']

    endpoint_config_name = endpoint_name + "-config"

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelPackageName': model_package_arn,
            'InitialInstanceCount': 1,
            'InstanceType': instance_type,
        }]
    )

    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    return {'status': 'Deployment triggered', 'endpoint_name': endpoint_name}
