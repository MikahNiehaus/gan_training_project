# Axolotl GAN API Backend

This is the backend API for the Axolotl GAN image generation application.

## Railway Deployment Instructions

### Prerequisites
- A Railway account
- Railway CLI installed (optional, you can also use the web interface)

### Deployment Steps

1. Login to Railway:
```bash
railway login
```

2. Initialize a new project (if not already done):
```bash
railway init
```

3. Link to your GitHub repository:
```bash
railway link
```

4. Deploy the application:
```bash
railway up
```

5. Set environment variables (if needed):
```bash
railway variables set KEY=VALUE
```

### Manual Deployment via Railway Dashboard

1. Create a new project on [Railway](https://railway.app/)
2. Connect your GitHub repository
3. Select the backend directory as the source directory
4. Railway will automatically detect the Python application and deploy it
5. Your API will be available at the provided Railway URL

## API Endpoints

- `GET /generate` - Generates a new axolotl image and returns it as base64

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The server will start on `http://localhost:5000`
# gan_training_project
