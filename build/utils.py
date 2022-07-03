import argparse
import os
import sys
import logging
import json

from google.cloud import aiplatform as vertex_ai


SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode', 
        type=str,
    )

    parser.add_argument(
        '--project',  
        type=str,
    )
    
    parser.add_argument(
        '--region',  
        type=str,
    )
    
    parser.add_argument(
        '--endpoint_display_name', 
        type=str,
    )

    parser.add_argument(
        '--model_display_name', 
        type=str,
    )

    parser.add_argument(
        '--job_name',
        type=str,
    )
    
    parser.add_argument(
        '--container_uri', 
        type=str,
    )
    
    parser.add_argument(
        '--model_serving_container_image_uri', 
        type=str,
    )
    return parser.parse_args()


def train_model(project, region, display_name, container_uri, model_serving_container_image_uri, model_display_name):
    vertex_ai.init(
    project=project,
    location=region,
    staging_bucket='gs://sanchit909090-bucket')
    
    job = vertex_ai.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri
    )
    
    model = job.run(
    model_display_name=model_display_name,
    replica_count=1,
    accelerator_count=0,
    base_output_dir='gs://sanchit909090-bucket/wcm/')
        
    return model


def create_endpoint(project, region, endpoint_display_name):
    logging.info(f"Creating endpoint {endpoint_display_name}")
    vertex_ai.init(
        project=project,
        location=region
    )
    
    endpoints = vertex_ai.Endpoint.list(
        filter=f'display_name={endpoint_display_name}', 
        order_by="update_time")
    
    if len(endpoints) > 0:
        logging.info(f"Endpoint {endpoint_display_name} already exists.")
        endpoint = endpoints[-1]
    else:
        endpoint = vertex_ai.Endpoint.create(endpoint_display_name)
    logging.info(f"Endpoint is ready.")
    logging.info(endpoint.gca_resource)
    return endpoint


def deploy_model(project, region, endpoint_display_name, model_display_name):
    logging.info(f"Deploying model {model_display_name} to endpoint {endpoint_display_name}")
    vertex_ai.init(
        project=project,
        location=region
    )
    
    model = vertex_ai.Model.list(
        filter=f'display_name={model_display_name}',
        order_by="update_time"
    )[-1]
    
    endpoint = vertex_ai.Endpoint.list(
        filter=f'display_name={endpoint_display_name}',
        order_by="update_time"
    )[-1]

    deployed_model = endpoint.deploy(model=model, traffic_percentage = 100, machine_type = "n1-standard-4") 

    logging.info(f"Model is deployed.")
    logging.info(deployed_model)
    return deployed_model
    

def main():
    args = get_args()
    
    if args.mode == 'create-endpoint':
        if not args.project:
            raise ValueError("project must be supplied.")
        if not args.region:
            raise ValueError("region must be supplied.")
        if not args.endpoint_display_name:
            raise ValueError("endpoint_display_name must be supplied.")
            
        result = create_endpoint(
            args.project, 
            args.region, 
            args.endpoint_display_name
        )
        
    elif args.mode == 'deploy-model':
        if not args.project:
            raise ValueError("project must be supplied.")
        if not args.region:
            raise ValueError("region must be supplied.")
        if not args.endpoint_display_name:
            raise ValueError("endpoint_display_name must be supplied.")
        if not args.model_display_name:
            raise ValueError("model_display_name must be supplied.")
            
        result = deploy_model(
            args.project, 
            args.region, 
            args.endpoint_display_name, 
            args.model_display_name
        )

    elif args.mode == 'train-model':
        if not args.job_name:
            raise ValueError("job_name must be supplied.")
        if not args.container_uri:
            raise ValueError("container_uri must be supplied.")
        if not args.model_serving_container_image_uri:
            raise ValueError("model_serving_container_image_uri must be supplied.")
            
        result = train_model(
            args.project, 
            args.region,
            args.job_name,
            args.container_uri,
            args.model_serving_container_image_uri,
            args.model_display_name
        )

        result.wait()

    else:
        raise ValueError(f"Invalid mode {args.mode}.")
        
    logging.info(result)
        
    
if __name__ == "__main__":
    main()
