#!/usr/bin/env python3
"""
AWS setup and configuration script
=================================

This script helps set up AWS credentials, IAM roles, and other prerequisites
for deploying the Fish ML model to AWS SageMaker.

Author: Fish ML Project
Date: 2025
"""

import boto3
import json
import time
import subprocess
import os
from botocore.exceptions import ClientError


class AWSSetup:
    """Class to handle AWS setup and configuration."""
    
    def __init__(self, region='us-east-1'):
        """Initialize AWS setup."""
        self.region = region
        self.session = None
        self.account_id = None
        
        print(f"🔧 AWS Setup for region: {region}")
    
    def check_aws_credentials(self):
        """Check if AWS credentials are configured."""
        print("🔍 Checking AWS credentials...")
        
        try:
            self.session = boto3.Session()
            sts_client = self.session.client('sts')
            response = sts_client.get_caller_identity()
            self.account_id = response['Account']
            
            print(f"✅ AWS credentials found")
            print(f"   Account ID: {self.account_id}")
            print(f"   User/Role: {response.get('Arn', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ AWS credentials not found or invalid: {str(e)}")
            print("\n📋 To configure AWS credentials, run:")
            print("   aws configure")
            print("\n   You'll need:")
            print("   - AWS Access Key ID")
            print("   - AWS Secret Access Key")
            print("   - Default region name (e.g., us-east-1)")
            print("   - Default output format (json)")
            return False
    
    def install_aws_cli(self):
        """Install AWS CLI if not present."""
        print("🔧 Checking AWS CLI installation...")
        
        try:
            result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ AWS CLI found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("❌ AWS CLI not found")
        print("\n📋 To install AWS CLI:")
        print("   Linux/Mac: pip install awscli")
        print("   Or visit: https://aws.amazon.com/cli/")
        return False
    
    def create_sagemaker_execution_role(self):
        """Create SageMaker execution role."""
        print("👤 Creating SageMaker execution role...")
        
        if not self.session:
            print("❌ AWS session not initialized")
            return None
        
        iam_client = self.session.client('iam')
        role_name = 'SageMakerExecutionRole-FishML'
        
        # Trust policy for SageMaker
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Check if role exists
            try:
                role = iam_client.get_role(RoleName=role_name)
                print(f"✅ Role '{role_name}' already exists")
                return role['Role']['Arn']
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchEntity':
                    raise
            
            # Create the role
            print(f"🏗️  Creating role '{role_name}'...")
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for Fish ML SageMaker jobs'
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach required policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess'
            ]
            
            for policy_arn in policies:
                print(f"   Attaching policy: {policy_arn}")
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
            
            print(f"✅ Role created successfully: {role_arn}")
            
            # Wait for role to propagate
            print("   Waiting for role to propagate...")
            time.sleep(10)
            
            return role_arn
            
        except Exception as e:
            print(f"❌ Error creating role: {str(e)}")
            return None
    
    def setup_s3_bucket(self, bucket_name=None):
        """Set up S3 bucket for the project."""
        if not bucket_name:
            bucket_name = f"fish-ml-{self.account_id}-{self.region}"
        
        print(f"🪣 Setting up S3 bucket: {bucket_name}")
        
        s3_client = self.session.client('s3', region_name=self.region)
        
        try:
            # Check if bucket exists
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Create bucket
                print(f"🏗️  Creating bucket '{bucket_name}'...")
                if self.region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                print(f"✅ Bucket created successfully")
            else:
                print(f"❌ Error checking bucket: {str(e)}")
                return None
        
        return bucket_name
    
    def create_ecr_repository(self, repository_name='fish-ml-model'):
        """Create ECR repository."""
        print(f"🐳 Setting up ECR repository: {repository_name}")
        
        ecr_client = self.session.client('ecr', region_name=self.region)
        
        try:
            # Check if repository exists
            ecr_client.describe_repositories(repositoryNames=[repository_name])
            print(f"✅ Repository '{repository_name}' already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'RepositoryNotFoundException':
                # Create repository
                print(f"🏗️  Creating repository '{repository_name}'...")
                ecr_client.create_repository(
                    repositoryName=repository_name,
                    imageScanningConfiguration={'scanOnPush': True}
                )
                print(f"✅ Repository created successfully")
            else:
                print(f"❌ Error with repository: {str(e)}")
                return None
        
        image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repository_name}:latest"
        return image_uri
    
    def check_docker_installation(self):
        """Check if Docker is installed."""
        print("🐳 Checking Docker installation...")
        
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker found: {result.stdout.strip()}")
                
                # Check if Docker daemon is running
                result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Docker daemon is running")
                    return True
                else:
                    print("❌ Docker daemon is not running")
                    print("   Please start Docker and try again")
                    return False
            else:
                print("❌ Docker command failed")
                return False
                
        except FileNotFoundError:
            print("❌ Docker not found")
            print("\n📋 To install Docker:")
            print("   Visit: https://docs.docker.com/get-docker/")
            return False
    
    def generate_config_file(self, role_arn, bucket_name, image_uri):
        """Generate configuration file for deployment."""
        config = {
            "aws": {
                "region": self.region,
                "account_id": self.account_id,
                "sagemaker_role": role_arn,
                "s3_bucket": bucket_name,
                "ecr_image_uri": image_uri
            },
            "model": {
                "name": "fish-weight-prediction",
                "type": "regression",
                "target": "Weight"
            },
            "deployment": {
                "instance_type": "ml.m5.large",
                "initial_instance_count": 1
            }
        }
        
        config_file = 'aws_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_file}")
        return config_file
    
    def run_full_setup(self):
        """Run the complete AWS setup process."""
        print("🚀 Running complete AWS setup...")
        print("=" * 60)
        
        # Step 1: Check prerequisites
        if not self.install_aws_cli():
            return False
        
        if not self.check_aws_credentials():
            return False
        
        if not self.check_docker_installation():
            return False
        
        # Step 2: Create AWS resources
        print("\n📋 Setting up AWS resources...")
        
        role_arn = self.create_sagemaker_execution_role()
        if not role_arn:
            print("❌ Failed to create SageMaker role")
            return False
        
        bucket_name = self.setup_s3_bucket()
        if not bucket_name:
            print("❌ Failed to setup S3 bucket")
            return False
        
        image_uri = self.create_ecr_repository()
        if not image_uri:
            print("❌ Failed to create ECR repository")
            return False
        
        # Step 3: Generate configuration
        print("\n📋 Generating configuration...")
        config_file = self.generate_config_file(role_arn, bucket_name, image_uri)
        
        print("\n🎉 AWS setup completed successfully!")
        print("=" * 60)
        print(f"📋 Summary:")
        print(f"   SageMaker Role: {role_arn}")
        print(f"   S3 Bucket: {bucket_name}")
        print(f"   ECR Image URI: {image_uri}")
        print(f"   Config file: {config_file}")
        print("\n🚀 You can now run the deployment script:")
        print("   python deploy_sagemaker.py")
        
        return True


def main():
    """Main function."""
    print("🐟 Fish ML - AWS Setup and Configuration")
    print("=" * 60)
    
    # Get region from user
    region = input("Enter AWS region (default: us-east-1): ").strip()
    if not region:
        region = 'us-east-1'
    
    # Initialize setup
    aws_setup = AWSSetup(region)
    
    # Run setup
    success = aws_setup.run_full_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        return 0
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
