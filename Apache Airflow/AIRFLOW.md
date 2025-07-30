# Deployment steps for Apache Airflow ML pipeline

## Deployment options

### Option 1: Local development

1. **Initial setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Start services**
   ```bash
   source venv/bin/activate
   export AIRFLOW_HOME=$(pwd)/airflow_home
   ./quickstart.sh
   ```

3. **Access Web UI**
   - URL: http://localhost:8080
   - Username: admin
   - Password: admin

### Option 2: Docker deployment (Recommended for production)

> **For detailed Docker deployment strategies and business value analysis, see [DOCKER.md](DOCKER.md)**

1. **Prerequisites**
   ```bash
   # Install Docker and Docker Compose
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   ```

2. **Deploy Services**
   ```bash
   docker-compose up -d
   ```

3. **Verify Deployment**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

## Configuration

### Environment variables
```bash
# Required
export AIRFLOW_HOME=/path/to/airflow
export PYTHONPATH=$PYTHONPATH:$AIRFLOW_HOME

# Optional
export ML_DATA_PATH=/path/to/ml/data
export ML_MODEL_PATH=/path/to/ml/models
```

### Database configuration
- **Development**: SQLite (default)
- **Production**: PostgreSQL (via Docker)

### Security
- Change default passwords in docker-compose.yml
- Update Fernet keys for encryption
- Configure SSL/TLS for production

## Monitoring and maintenance

### Health Checks
```bash
# Check pipeline status
python monitor_pipeline.py --action status

# Check model performance
python monitor_pipeline.py --action performance

# Verify file structure
python monitor_pipeline.py --action files
```

### Logs
```bash
# View scheduler logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log

# View webserver logs
tail -f $AIRFLOW_HOME/logs/dag_processor_manager/*.log

# Docker logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver
```

### Backup and recovery
```bash
# Backup models and data
tar -czf ml_backup_$(date +%Y%m%d).tar.gz $AIRFLOW_HOME/ml_*

# Backup Airflow database
cp $AIRFLOW_HOME/airflow.db airflow_backup_$(date +%Y%m%d).db
```

## Pipeline management

### DAG operations
```bash
# List all DAGs
airflow dags list

# Trigger DAG manually
airflow dags trigger supervised_learning_pipeline

# Pause/Unpause DAG
airflow dags pause supervised_learning_pipeline
airflow dags unpause supervised_learning_pipeline

# Test DAG
airflow dags test supervised_learning_pipeline $(date +%Y-%m-%d)
```

### Task management
```bash
# List tasks in a DAG
airflow tasks list supervised_learning_pipeline

# Test individual task
airflow tasks test supervised_learning_pipeline load_data $(date +%Y-%m-%d)

# Clear task instances
airflow tasks clear supervised_learning_pipeline --start-date $(date +%Y-%m-%d)
```

## Troubleshooting

### Issues

1. **Port 8080 Already in Use**
   ```bash
   # Find process using port
   sudo netstat -tulpn | grep :8080
   # Kill process or change port in configuration
   ```

2. **Database Connection Issues**
   ```bash
   # Reset Airflow database
   airflow db reset
   airflow db init
   ```

3. **DAG Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH=$PYTHONPATH:$AIRFLOW_HOME
   # Verify DAG syntax
   python $AIRFLOW_HOME/dags/ml_pipeline_dag.py
   ```

4. **Permission Issues**
   ```bash
   # Fix file permissions
   chmod -R 755 $AIRFLOW_HOME
   chown -R $(whoami):$(whoami) $AIRFLOW_HOME
   ```

### Performance optimization

1. **Increase worker processes**
   ```bash
   # In airflow.cfg
   [webserver]
   workers = 4
   
   [celery]
   worker_concurrency = 8
   ```

2. **Database optimization**
   ```bash
   # Use PostgreSQL
   # Configure connection pooling
   ```

3. **Resource monitoring**
   ```bash
   # Monitor system resources
   htop
   df -h
   free -m
   ```

## 🔐 Security

### Authentication
- Enable LDAP/OAuth authentication
- Use strong passwords
- Implement role-based access control

### Network security
- Use HTTPS in production
- Configure firewall rules
- Implement VPN access

### Data security
- Encrypt sensitive data
- Use secure connections
- Implement data retention policies

## Scaling for production

### Horizontal scaling
- Use CeleryExecutor
- Add multiple worker nodes
- Configure Redis/RabbitMQ

### Vertical scaling
- Increase CPU/Memory resources
- Optimize database performance
- Use SSD storage

### High availability
- Multi-instance deployment
- Load balancing
- Database replication

## CI/CD integration

### Pipeline testing
```bash
# Run unit tests
python -m pytest tests/

# Validate DAG syntax
python -m py_compile dags/ml_pipeline_dag.py

# Test pipeline end-to-end
python supervised_regression_pipeline.py
```

### Automated deployment
```bash
# Example deployment script
#!/bin/bash
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
```

