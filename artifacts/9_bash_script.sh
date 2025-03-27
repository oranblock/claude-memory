#!/bin/bash
# deployment.sh
echo "Starting deployment process..."

# Update source code
git pull origin main

# Install dependencies
npm install --production

# Build the application
npm run build

# Restart the service
pm2 restart app_name

echo "Deployment completed successfully."