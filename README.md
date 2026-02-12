# MLaaS Platform - Collaborative Machine Learning with Encrypted Data

A comprehensive Machine Learning as a Service (MLaaS) platform that enables multiple users to collaboratively train ML models while maintaining data privacy through encryption and federated learning.

## Architecture

### Frontend (React + Tailwind CSS)
- **Modern UI**: Beautiful, responsive interface with Tailwind CSS
- **Real-time Updates**: Live training progress and model performance
- **Interactive Charts**: Data visualization with Recharts
- **Secure Authentication**: JWT-based authentication system

### Backend (Flask + MongoDB)
- **RESTful API**: Clean, well-documented API endpoints
- **Database**: MongoDB for flexible data storage
- **Cloud Storage**: AWS S3 integration for scalable storage
- **Advanced ML**: PyTorch, Scikit-learn, Concrete-ML integration


## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your configuration

# Start MongoDB (if not running)
mongod

# Run the Flask application
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Full Stack Development
```bash
# Install all dependencies
npm run install-all

# Start both frontend and backend
npm run dev
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:

```env
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=jwt-secret-string
MONGO_URI=mongodb://localhost:27017/mlaas_platform
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_BUCKET_NAME=mlaas-platform-data
ENCRYPTION_KEY=your-encryption-key-here
```


## Usage

### 1. User Registration/Login
- Register with username, email, and password
- Select your domain and subdomain
- Access the collaborative platform

### 2. Data Upload
- Choose your domain and subdomain
- Upload encrypted datasets (CSV, JSON, Excel)
- Data is automatically encrypted and stored securely

### 3. Model Training
- Select training parameters and model type
- Start collaborative training with other users
- Monitor real-time progress and accuracy

### 4. Model Management
- View all trained models
- Test models with new data
- Analyze performance metrics
- Download model artifacts

## Deployment

### Development
```bash
# Start development servers
npm run dev
```

### Production
```bash
# Build frontend
cd frontend && npm run build

# Deploy backend
cd backend && python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
