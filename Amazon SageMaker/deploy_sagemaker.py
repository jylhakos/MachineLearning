#!/usr/bin/env python3
"""
AWS SageMaker deployment script for Fish weight prediction
=========================================================

This script handles the complete deployment pipeline:
1. Build and push Docker image to ECR
2. Create SageMaker training job
3. Deploy model to SageMaker endpoint
4. Test the deployed model

Author: Fish ML Project
Date: 2025
"""

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
import json
import time
import os
from datetime import datetime


class FishMLDeployment:
    """Class to handle SageMaker deployment for Fish ML model."""
    
    def __init__(self, region='us-east-1'):
        """Initialize AWS clients and SageMaker session."""
        self.region = region
        self.session = boto3.Session()
        self.sagemaker_session = sagemaker.Session()
        self.ecr_client = self.session.client('ecr', region_name=region)
        
        # Get AWS account ID
        self.account_id = self.session.client('sts').get_caller_identity()['Account']
        
        # Repository and image names
        self.repository_name = 'fish-ml-model'
        self.image_tag = 'latest'
        self.image_uri = f"{self.account_id}.dkr.ecr.{region}.amazonaws.com/{self.repository_name}:{self.image_tag}"
        
        # S3 bucket for data and models
        self.bucket_name = f"fish-ml-{self.account_id}-{region}"
        
        print(f"🔧 Initialized FishMLDeployment")
        print(f"   Region: {region}")
        print(f"   Account ID: {self.account_id}")
        print(f"   Image URI: {self.image_uri}")
        print(f"   S3 Bucket: {self.bucket_name}")
    
    def create_ecr_repository(self):
        """Create ECR repository if it doesn't exist."""
        try:
            # Try to describe the repository
            self.ecr_client.describe_repositories(repositoryNames=[self.repository_name])
            print(f"✅ ECR repository '{self.repository_name}' already exists")
        except self.ecr_client.exceptions.RepositoryNotFoundException:
            # Create the repository
            print(f"🏗️  Creating ECR repository '{self.repository_name}'...")
            self.ecr_client.create_repository(
                repositoryName=self.repository_name,
                imageScanningConfiguration={'scanOnPush': True}
            )
            print(f"✅ ECR repository '{self.repository_name}' created successfully")
    
    def build_and_push_docker_image(self):
        """Build and push Docker image to ECR."""
        import subprocess
        
        print("🐳 Building and pushing Docker image...")
        
        # Get ECR login token
        token_response = self.ecr_client.get_authorization_token()
        token = token_response['authorizationData'][0]['authorizationToken']
        endpoint = token_response['authorizationData'][0]['proxyEndpoint']
        
        # Decode the token and login to ECR
        import base64
        username, password = base64.b64decode(token).decode().split(':')
        
        # Build commands
        commands = [
            f"docker build -t {self.repository_name}:{self.image_tag} .",
            f"docker tag {self.repository_name}:{self.image_tag} {self.image_uri}",
            f"docker login -u {username} -p {password} {endpoint}",
            f"docker push {self.image_uri}"
        ]
        
        for cmd in commands:
            print(f"   Running: {cmd.replace(password, '***')}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Error: {result.stderr}")
                raise Exception(f"Command failed: {cmd}")
        
        print(f"✅ Docker image pushed successfully to {self.image_uri}")
    
    def create_s3_bucket(self):
        """Create S3 bucket for storing data and models."""
        s3_client = self.session.client('s3', region_name=self.region)
        
        try:
            s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ S3 bucket '{self.bucket_name}' already exists")
        except:
            print(f"🪣 Creating S3 bucket '{self.bucket_name}'...")
            if self.region == 'us-east-1':
                s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            print(f"✅ S3 bucket '{self.bucket_name}' created successfully")
    
    def upload_training_data(self):
        """Upload training data to S3."""
        s3_client = self.session.client('s3')
        data_key = 'data/Fish.csv'
        local_path = 'Dataset/Fish.csv'
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Training data not found at {local_path}")
        
        print(f"📤 Uploading training data to S3...")
        s3_client.upload_file(local_path, self.bucket_name, data_key)
        
        s3_uri = f"s3://{self.bucket_name}/data/"
        print(f"✅ Training data uploaded to {s3_uri}")
        return s3_uri
    
    def create_training_job(self, model_type='linear_regression'):
        """Create and run SageMaker training job."""
        print(f"🏋️  Creating SageMaker training job...")
        
        # Upload training data
        training_data_uri = self.upload_training_data()
        
        # Get execution role
        try:
            role = get_execution_role()
        except:
            # Create a basic role ARN if not in SageMaker environment
            role = f"arn:aws:iam::{self.account_id}:role/SageMakerExecutionRole"
            print(f"⚠️  Using role: {role} (make sure this role exists)")
        
        # Create estimator
        estimator = Estimator(
            image_uri=self.image_uri,
            role=role,
            instance_count=1,
            instance_type='ml.m5.large',
            output_path=f"s3://{self.bucket_name}/model/",
            sagemaker_session=self.sagemaker_session,
            hyperparameters={
                'model-type': model_type
            }
        )
        
        # Start training
        job_name = f"fish-ml-{model_type}-{int(time.time())}"
        print(f"🚀 Starting training job: {job_name}")
        
        estimator.fit(
            inputs={'training': training_data_uri},
            job_name=job_name
        )
        
        print(f"✅ Training job completed: {job_name}")
        return estimator
    
    def deploy_model(self, estimator):
        """Deploy trained model to SageMaker endpoint."""
        print(f"🚀 Deploying model to SageMaker endpoint...")
        
        endpoint_name = f"fish-ml-endpoint-{int(time.time())}"
        
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',
            endpoint_name=endpoint_name
        )
        
        print(f"✅ Model deployed to endpoint: {endpoint_name}")
        return predictor, endpoint_name
    
    def test_endpoint(self, predictor):
        """Test the deployed endpoint with sample data."""
        print(f"🧪 Testing deployed endpoint...")
        
        # Sample test data
        test_data = {
            "Species": "Bream",
            "Length1": 23.2,
            "Length2": 25.4,
            "Length3": 30.0,
            "Height": 11.52,
            "Width": 4.02
        }
        
        try:
            result = predictor.predict(test_data)
            print(f"✅ Test prediction successful:")
            print(f"   Input: {test_data}")
            print(f"   Prediction: {result}")
            return True
        except Exception as e:
            print(f"❌ Test prediction failed: {str(e)}")
            return False
    
    def cleanup_endpoint(self, endpoint_name):
        """Clean up the SageMaker endpoint."""
        try:
            sagemaker_client = self.session.client('sagemaker')
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"🗑️  Endpoint {endpoint_name} deleted")
        except Exception as e:
            print(f"⚠️  Error deleting endpoint: {str(e)}")


def main():
    """Main deployment function."""
    print("🐟 Fish ML - AWS SageMaker Deployment")
    print("=" * 60)
    
    # Initialize deployment
    deployment = FishMLDeployment()
    
    try:
        # Step 1: Prepare infrastructure
        print("\n📋 Step 1: Preparing AWS infrastructure...")
        deployment.create_s3_bucket()
        deployment.create_ecr_repository()
        
        # Step 2: Build and push Docker image
        print("\n📋 Step 2: Building and pushing Docker image...")
        deployment.build_and_push_docker_image()
        
        # Step 3: Train model
        print("\n📋 Step 3: Training model...")
        estimator = deployment.create_training_job('linear_regression')
        
        # Step 4: Deploy model
        print("\n📋 Step 4: Deploying model...")
        predictor, endpoint_name = deployment.deploy_model(estimator)
        
        # Step 5: Test endpoint
        print("\n📋 Step 5: Testing endpoint...")
        test_success = deployment.test_endpoint(predictor)
        
        if test_success:
            print("\n🎉 Deployment completed successfully!")
            print(f"   Endpoint name: {endpoint_name}")
            print(f"   S3 bucket: {deployment.bucket_name}")
            print(f"   Docker image: {deployment.image_uri}")
        else:
            print("\n⚠️  Deployment completed but testing failed")
        
        # Optional: Clean up endpoint (uncomment if desired)
        # print("\n🗑️  Cleaning up endpoint...")
        # deployment.cleanup_endpoint(endpoint_name)
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
