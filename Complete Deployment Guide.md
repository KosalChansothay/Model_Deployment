# 🌸 Professional Iris Classification API - Complete Deployment Guide

## 🎯 Enhanced Features

- ✅ **3 Machine Learning Models**: Logistic Regression, SVM, Random Forest
- ✅ **Advanced Feature Engineering**: 10+ engineered features for better accuracy
- ✅ **Hyperparameter Tuning**: GridSearchCV optimization for all models
- ✅ **Beautiful Web Interface**: Modern, responsive frontend
- ✅ **Multiple Prediction Modes**: Single model, compare all models, batch predictions
- ✅ **Real-time Model Comparison**: See all models' predictions side-by-side
- ✅ **Production Ready**: Docker, Nginx, SSL support

## 📁 Project Structure

```
iris-api/
├── app.py                  # FastAPI backend with multiple endpoints
├── model.pkl              # Trained models + scaler + metadata
├── train_model.py         # Advanced training with feature engineering
├── requirements.txt       # Python dependencies
├── Dockerfile            # Multi-stage optimized build
├── docker-compose.yml    # Container orchestration
├── static/
│   └── style.css         # Professional CSS styling
└── templates/
    └── index.html        # Interactive frontend
```

## 🚀 Quick Start Guide

### Step 1: Setup Project
```bash
# Create project directory
mkdir iris-api
cd iris-api

# Create subdirectories
mkdir static templates

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas scikit-learn fastapi uvicorn jinja2 python-multipart
```

### Step 2: Create All Files
Create each file as shown in the artifacts above:
- `train_model.py`
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `static/style.css`
- `templates/index.html`

### Step 3: Train Models
```bash
python train_model.py
```

**Expected Output:**
```
==================================================================
IRIS CLASSIFICATION - ADVANCED MODEL TRAINING
==================================================================

📊 Dataset Overview:
Shape: (150, 5)

🔧 Feature Engineering Complete!
Original features: 4
Engineered features: 14

📦 Data Split:
Training set: 120 samples
Test set: 30 samples

==================================================================
MODEL TRAINING & EVALUATION
==================================================================

1️⃣  Training Logistic Regression...
   ✓ Best params: {...}
   ✓ Test Accuracy: 1.0000
   ✓ CV Score: 0.9833 (±0.0272)

2️⃣  Training Support Vector Machine...
   ✓ Best params: {...}
   ✓ Test Accuracy: 1.0000
   ✓ CV Score: 0.9917 (±0.0166)

3️⃣  Training Random Forest...
   ✓ Best params: {...}
   ✓ Test Accuracy: 1.0000
   ✓ CV Score: 0.9750 (±0.0395)

🏆 Best Model: Support Vector Machine
   Accuracy: 1.0000
```

### Step 4: Test Locally (Without Docker)
```bash
# Run the app
python app.py

# Visit in browser:
# http://localhost:8000/          - Web Interface
# http://localhost:8000/api/docs  - API Documentation
```

### Step 5: Test API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Get Models Info:**
```bash
curl http://localhost:8000/api/models/info
```

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    "model_name": "best"
  }'
```

**Compare All Models:**
```bash
curl -X POST "http://localhost:8000/api/predict/all" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.4,
    "sepal_width": 3.2,
    "petal_length": 4.5,
    "petal_width": 1.5
  }'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/api/predict/batch?model_name=svm" \
  -H "Content-Type: application/json" \
  -d '[
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
  ]'
```

### Step 6: Python Testing Script
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Get models info
print("=" * 70)
print("AVAILABLE MODELS")
print("=" * 70)
response = requests.get(f"{BASE_URL}/api/models/info")
data = response.json()
print(f"Best Model: {data['best_model']}")
for name, details in data['model_details'].items():
    print(f"\n{name.upper()}")
    print(f"  Accuracy: {details['accuracy']:.4f}")
    print(f"  CV Score: {details['cv_mean']:.4f} ±{details['cv_std']:.4f}")

# Test 2: Single prediction with specific model
print("\n" + "=" * 70)
print("SINGLE MODEL PREDICTION (SVM)")
print("=" * 70)
payload = {
    "features": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    },
    "model_name": "svm"
}
response = requests.post(f"{BASE_URL}/api/predict", json=payload)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities:")
for species, prob in result['probabilities'].items():
    print(f"  {species}: {prob:.2%}")

# Test 3: Compare all models
print("\n" + "=" * 70)
print("ALL MODELS COMPARISON")
print("=" * 70)
features = {
    "sepal_length": 6.4,
    "sepal_width": 3.2,
    "petal_length": 4.5,
    "petal_width": 1.5
}
response = requests.post(f"{BASE_URL}/api/predict/all", json=features)
result = response.json()
print(f"Consensus: {result['consensus_prediction']}")
print("\nIndividual Predictions:")
for model_name, pred in result['predictions'].items():
    print(f"  {model_name}: {pred['prediction']} ({pred['confidence']:.2%})")

# Test 4: Batch prediction
print("\n" + "=" * 70)
print("BATCH PREDICTION")
print("=" * 70)
batch = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
]
response = requests.post(f"{BASE_URL}/api/predict/batch", json=batch)
result = response.json()
print(f"Total predictions: {result['total_predictions']}")
for i, res in enumerate(result['results'], 1):
    print(f"\nSample {i}: {res['prediction']} ({res['confidence']:.2%})")
```

## 🐳 Docker Deployment

### Local Docker Testing
```bash
# Build the image
docker build -t iris-api:latest .

# Run container
docker run -d -p 8000:8000 --name iris-classifier iris-api:latest

# Check logs
docker logs -f iris-classifier

# Test
curl http://localhost:8000/api/health

# Stop and remove
docker stop iris-classifier
docker rm iris-classifier
```

### Using Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

## ☁️ DigitalOcean Deployment

### Step 1: Create Droplet
1. Go to DigitalOcean dashboard
2. Create Droplet:
   - **OS**: Ubuntu 22.04 LTS
   - **Plan**: Basic ($6/mo - 1GB RAM)
   - **Region**: Choose nearest
   - **Authentication**: SSH key (recommended)
   - **Hostname**: iris-ml-api
3. Note the IP address

### Step 2: Initial Server Setup
```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify installations
docker --version
docker-compose --version

# Enable Docker to start on boot
systemctl enable docker
systemctl start docker
```

### Step 3: Upload Project Files

**Option A: Using SCP (from local machine)**
```bash
# Create directory on server first
ssh root@YOUR_DROPLET_IP "mkdir -p /opt/iris-api/static /opt/iris-api/templates"

# Upload files
scp app.py requirements.txt Dockerfile docker-compose.yml root@YOUR_DROPLET_IP:/opt/iris-api/
scp model.pkl root@YOUR_DROPLET_IP:/opt/iris-api/
scp static/style.css root@YOUR_DROPLET_IP:/opt/iris-api/static/
scp templates/index.html root@YOUR_DROPLET_IP:/opt/iris-api/templates/
```

**Option B: Using Git**
```bash
# On your local machine, create a GitHub repo and push code
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/iris-api.git
git push -u origin main

# On droplet
cd /opt
git clone https://github.com/yourusername/iris-api.git
cd iris-api
```

**Option C: Manual Creation**
```bash
# On droplet
mkdir -p /opt/iris-api/static /opt/iris-api/templates
cd /opt/iris-api

# Create files using nano or vim
nano app.py  # Copy and paste content
nano requirements.txt
nano Dockerfile
nano docker-compose.yml
nano static/style.css
nano templates/index.html

# Upload model.pkl separately
# From local: scp model.pkl root@YOUR_DROPLET_IP:/opt/iris-api/
```

### Step 4: Start Application
```bash
cd /opt/iris-api

# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f

# Verify it's running
curl http://localhost:8000/api/health
```

### Step 5: Configure Firewall
```bash
# Allow necessary ports
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# Check status
ufw status
```

### Step 6: Install and Configure Nginx
```bash
# Install Nginx
apt install nginx -y

# Create configuration
nano /etc/nginx/sites-available/iris-api
```

**Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    # Increase buffer sizes for large requests
    client_max_body_size 10M;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed in future)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Cache static files
    location /static {
        proxy_pass http://localhost:8000/static;
        proxy_cache_valid 200 1d;
        add_header Cache-Control "public, max-age=86400";
    }
}
```

**Enable and test:**
```bash
# Enable site
ln -s /etc/nginx/sites-available/iris-api /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test configuration
nginx -t

# Restart Nginx
systemctl restart nginx
systemctl enable nginx
```

### Step 7: Setup SSL with Let's Encrypt (Optional but Recommended)

**Prerequisites:** You need a domain name pointing to your droplet IP.

```bash
# Install Certbot
apt install certbot python3-certbot-nginx -y

# Obtain certificate
certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Follow prompts:
# - Enter email
# - Agree to terms
# - Choose redirect (option 2 recommended)

# Test auto-renewal
certbot renew --dry-run

# Certificate will auto-renew via cron job
```

### Step 8: Test Deployment

**From your local machine:**
```bash
# Test health endpoint
curl https://yourdomain.com/api/health

# Test web interface
# Open in browser: https://yourdomain.com

# Test prediction
curl -X POST "https://yourdomain.com/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    "model_name": "best"
  }'
```

## 📊 Advanced Testing Scenarios

### Test 1: All Three Species
```bash
# Iris-setosa (small petals)
curl -X POST "https://yourdomain.com/api/predict/all" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Iris-versicolor (medium petals)
curl -X POST "https://yourdomain.com/api/predict/all" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5}'

# Iris-virginica (large petals)
curl -X POST "https://yourdomain.com/api/predict/all" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}'
```

### Test 2: Compare Model Performance
```python
import requests
import pandas as pd

url = "https://yourdomain.com/api/predict/all"

# Test multiple samples
samples = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
    {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9}
]

results = []
for sample in samples:
    response = requests.post(url, json=sample)
    data = response.json()
    
    row = sample.copy()
    row['consensus'] = data['consensus_prediction']
    for model, pred in data['predictions'].items():
        row[f'{model}_pred'] = pred['prediction']
        row[f'{model}_conf'] = pred['confidence']
    results.append(row)

df = pd.DataFrame(results)
print(df)
```

## 🔧 Maintenance & Monitoring

### View Logs
```bash
# Application logs
docker-compose logs -f

# Nginx access logs
tail -f /var/log/nginx/access.log

# Nginx error logs
tail -f /var/log/nginx/error.log

# System logs
journalctl -u docker -f
```

### Update Application
```bash
# On droplet
cd /opt/iris-api

# Pull latest changes (if using Git)
git pull

# Or upload new files via SCP

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Verify
docker-compose ps
curl http://localhost:8000/api/health
```

### Monitor Resources
```bash
# Check disk usage
df -h

# Check memory
free -h

# Check container stats
docker stats iris-classifier

# Check Docker disk usage
docker system df
```

### Backup
```bash
# Create backup
cd /opt
tar -czf iris-api-backup-$(date +%Y%m%d).tar.gz iris-api/

# Download to local machine
scp root@YOUR_DROPLET_IP:/opt/iris-api-backup-*.tar.gz .

# Restore (if needed)
tar -xzf iris-api-backup-YYYYMMDD.tar.gz
```

### Auto-restart on Failure
The docker-compose.yml already includes `restart: unless-stopped`, but you can also set up a systemd service:

```bash
# Create systemd service
nano /etc/systemd/system/iris-api.service
```

```ini
[Unit]
Description=Iris Classification API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/iris-api
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
systemctl enable iris-api
systemctl start iris-api
```

## 🎨 Using the Web Interface

1. **Open in browser**: `https://yourdomain.com`
2. **View model statistics**: See accuracy and CV scores for all models
3. **Enter measurements**: Input sepal and petal dimensions
4. **Select model**: Choose specific model or use "Best Model (Auto)"
5. **Predict**: Click "Predict" for single model or "Compare All Models"
6. **View results**: See predictions with confidence scores and probabilities
7. **Try examples**: Click on example cards to auto-fill measurements

## 🚀 Advanced Features

### API Rate Limiting (Production)
```python
# Add to app.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/predict")
@limiter.limit("100/minute")
async def predict_single(request: Request, prediction: PredictionRequest):
    # ... existing code
```

### Add API Key Authentication
```python
# Add to app.py
from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/api/predict")
async def predict_single(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    # ... existing code
```

### Add Prediction Logging
```python
# Add to app.py
import json
from datetime import datetime

@app.post("/api/predict")
async def predict_single(request: PredictionRequest):
    result = # ... make prediction
    
    # Log prediction
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": request.features.dict(),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "model": result["model_used"]
    }
    
    with open("predictions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return result
```

## 📈 Performance Optimization

### Enable Model Caching
Models are already loaded once at startup. For additional optimization:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(sepal_length, sepal_width, petal_length, petal_width, model_name):
    # ... prediction logic
    pass
```

### Use Gunicorn for Production
```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Update Dockerfile CMD:
```dockerfile
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

## ✅ Deployment Checklist

- [ ] All project files created
- [ ] Models trained successfully (accuracy > 95%)
- [ ] Local testing passed (both web and API)
- [ ] Docker build successful
- [ ] Docker container runs locally
- [ ] Droplet created and accessible
- [ ] Docker and Docker Compose installed on droplet
- [ ] Project files uploaded to droplet
- [ ] Application running in Docker on droplet
- [ ] Firewall configured
- [ ] Nginx installed and configured
- [ ] SSL certificate obtained (if using domain)
- [ ] External access working
- [ ] All API endpoints tested
- [ ] Web interface accessible and functional

## 🎯 Next Steps

1. **Add More Features**:
   - User authentication
   - Prediction history
   - Model retraining interface
   - Data upload for batch predictions

2. **Monitoring & Analytics**:
   - Set up Prometheus + Grafana
   - Add error tracking (Sentry)
   - Monitor prediction accuracy over time

3. **CI/CD Pipeline**:
   - GitHub Actions for automated testing
   - Automated deployment on push
   - Docker image versioning

4. **Scaling**:
   - Load balancer with multiple droplets
   - Kubernetes deployment
   - Database for prediction storage

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)
- [DigitalOcean Tutorials](https://www.digitalocean.com/community/tutorials)

---

**🎉 Congratulations!** You now have a professional, production-ready ML API with multiple models, feature engineering, and a beautiful web interface!