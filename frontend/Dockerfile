# frontend/Dockerfile

# Stage 1 - Build the React app
FROM node:14 as build

WORKDIR /app

# Copy package.json and package-lock.json first to leverage Docker caching
COPY package.json .
COPY package-lock.json .

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Build the React app
RUN npm run build

# Stage 2 - Serve the React app with Nginx
FROM nginx:alpine

# Copy the build output from the previous stage to Nginx server directory
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80 to the outside world
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
