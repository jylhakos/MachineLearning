# Fish weight prediction - Machine Learning pipeline

A machine learning pipeline for predicting fish weight based on species and physical measurements, deployable to AWS SageMaker with Docker containerization and FastAPI inference server.

## Project

This project implements supervised learning regression models to predict fish weight using features like species, length measurements, height, and width.

- **Local development**: Python scripts for data analysis and model training
- **Containerization**: Docker setup for consistent deployment
- **Cloud deployment**: AWS SageMaker integration for scalable ML operations
- **API service**: FastAPI server for real-time predictions
- **MLOps Pipeline**: Complete CI/CD workflow for production deployment

## Dataset

The `Fish.csv` dataset contains measurements for different fish species:

- **Species**: Fish type (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)
- **Weight**: Target variable (grams)
- **Length1, Length2, Length3**: Different length measurements (cm)
- **Height**: Vertical height (cm)
- **Width**: Diagonal width (cm)

## Project

```
├── Dataset/
│   └── Fish.csv                 # Training dataset
├── model/                       # Trained model artifacts (generated)
├── fish_analysis.py            # Comprehensive data analysis
├── fish_regression.py          # Basic regression implementation  
├── train.py                    # SageMaker training script
├── inference_server.py         # FastAPI inference server
├── local_dev.py               # Local development tools
├── deploy_sagemaker.py        # AWS SageMaker deployment
├── aws_setup.py               # AWS infrastructure setup
├── Dockerfile                 # Container configuration
├── docker-entrypoint.sh       # Container entry point
├── requirements.txt           # Python dependencies
├── setup.sh                   # Environment setup script
└── README.md                  # Information
```

## Start

### 1. Local development setup

```bash
# Clone and navigate to project
cd "Amazon SageMaker"

# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv and installs dependencies)
./setup.sh

# Or use the interactive development tool
python local_dev.py
```

### 2. Train model locally

```bash
# Activate virtual environment
source venv/bin/activate

# Train the model
python train.py --model-dir ./model --train ./

# Start inference server
python inference_server.py
```

### 3. Test API Endpoints

Access the API documentation at `http://localhost:8000/docs`

```bash
# Test single prediction
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "species": "Bream",
  "length1": 23.2,
  "length2": 25.4,
  "length3": 30.0,
  "height": 11.52,
  "width": 4.02
}'

# Check API health
curl http://localhost:8000/health
```

## 🐳 Docker deployment

### Build and run container

```bash
# Build Docker image
docker build -t fish-ml .

# Train model in container
docker run -v $(pwd)/Dataset:/opt/ml/input/data/training fish-ml train

# Run inference server
docker run -p 8000:8000 -v $(pwd)/model:/opt/ml/model fish-ml
```

## ☁️ AWS SageMaker deployment

### Prerequisites

1. **AWS CLI configuration**:
```bash
aws configure
# Enter your AWS Access Key ID, Secret Key, Region, and Output format
```

2. **Docker**: Check Docker is installed and running

3. **AWS permissions**: IAM user/role with SageMaker, S3, and ECR permissions

### Automatic deployment

```bash
# Run AWS setup (creates IAM roles, S3 buckets, ECR repositories)
python aws_setup.py

# Deploy to SageMaker
python deploy_sagemaker.py
```

### Dataset upload to Amazon AWS

To upload the Fish.csv dataset to Amazon AWS S3 storage:

1. **First, configure AWS credentials:**

```bash
# Option 1: Using AWS CLI
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region, and Output format

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

2. **Then run the AWS setup script:**

```bash
cd "/home/laptop/EXERCISES/MACHINE LEARNING/MachineLearning/Amazon SageMaker"
source venv/bin/activate
python aws_setup.py
```

3. **Or manually upload using AWS CLI:**

```bash
# Create S3 bucket
aws s3 mb s3://fish-ml-your-account-id-region

# Upload the dataset
aws s3 cp Dataset/Fish.csv s3://fish-ml-your-account-id-region/data/Fish.csv

# Verify upload
aws s3 ls s3://fish-ml-your-account-id-region/data/
```

4. **For SageMaker deployment:**

```bash
# After AWS setup, deploy the complete pipeline
python deploy_sagemaker.py
```

#### Verify dataset upload status

```bash
# Check if dataset exists locally
ls -la Dataset/Fish.csv

# Check if AWS credentials are configured
aws configure list

# List S3 bucket contents (after upload)
aws s3 ls s3://your-bucket-name/data/

# Check dataset in S3
aws s3 ls s3://your-bucket-name/data/Fish.csv
```

### Manual deployment

1. **Setup AWS infrastructure**:
```bash
# Create S3 bucket
aws s3 mb s3://fish-ml-your-account-id-region

# Create ECR repository  
aws ecr create-repository --repository-name fish-ml-model
```

2. **Build and push Docker image**:
```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t fish-ml-model .
docker tag fish-ml-model:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fish-ml-model:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fish-ml-model:latest
```

3. **Create SageMaker training job**:
```bash
# Upload training data to S3
aws s3 cp Dataset/Fish.csv s3://fish-ml-your-account-id-region/data/

# Use deploy_sagemaker.py or AWS console to create training job
```

## RESTful API

### Endpoints

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation
- **POST** `/predict` - Single fish weight prediction
- **POST** `/predict/batch` - Batch predictions
- **GET** `/species` - Supported fish species
- **GET** `/model/info` - Model information

### Request Format

```json
{
  "species": "Bream",
  "length1": 23.2,
  "length2": 25.4, 
  "length3": 30.0,
  "height": 11.52,
  "width": 4.02
}
```

### Response Format

```json
{
  "predicted_weight": 242.18,
  "species": "Bream",
  "timestamp": "2025-01-30T10:30:45.123456"
}
```

## Model performance

The project includes multiple regression algorithms:

- **Linear regression**: Baseline model with good interpretability
- **Ridge regression**: Regularized linear model for better generalization
- **Lasso regression**: Feature selection through L1 regularization
- **Random forest**: Ensemble method for non-linear relationships

Typical performance metrics:
- **R² Score**: 0.85-0.95 (varies by model)
- **MAE**: 15-30 grams
- **RMSE**: 25-45 grams

## 🔧 Development tools

### Local development script

```bash
python local_dev.py
```

Features:
- Interactive menu for common tasks
- Environment setup and dependency management
- Local model training and testing
- Docker build and test automation
- Full pipeline execution

### Analysis

```bash
# Comprehensive data analysis
python fish_analysis.py

# Basic regression analysis  
python fish_regression.py
```

## Dependencies

### ML libraries
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

### AWS integration
- boto3 >= 1.34.0
- sagemaker >= 2.190.0
- awscli >= 1.32.0

### API framework
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- pydantic >= 2.5.0

## 🔒 Security

- API input validation using Pydantic models
- Environment variable management for sensitive data
- IAM roles with minimal required permissions
- Container security with non-root user execution
- HTTPS support for production deployments

## Scaling in production

### Performance optimization
- Model caching and warm-up strategies
- Batch prediction capabilities
- Async processing for high throughput
- Auto-scaling with SageMaker endpoints

### Monitoring and logging
- Health check endpoints
- Structured logging with timestamps
- Model performance metrics tracking
- Error handling and alerting

### CI/CD
- Automated testing pipeline
- Model validation and A/B testing
- Blue-green deployment strategies
- Rollback capabilities

- Review the API documentation at `/docs`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## References

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [AWS ML Production Pipeline Guide](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-production-ready-pipelines/)
- [Build and deploy ML inference applications from scratch using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/build-and-deploy-ml-inference-applications-from-scratch-using-amazon-sagemaker)

---

**Fish ML Project** - Predicting fish weight with machine learning 🐟

