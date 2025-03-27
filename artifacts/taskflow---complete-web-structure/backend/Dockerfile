dockerfileCopy# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./
COPY backend/package.json backend/package-lock.json ./backend/

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the app
RUN npm run build:backend

# Set environment variables
ENV NODE_ENV=production
ENV PORT=5000

# Expose port
EXPOSE 5000

# Start the application
CMD ["node", "backend/dist/server.js"]