# Step 1: Base image
FROM python:3.9-slim

# Step 2: Set environment variables to avoid input prompts
ENV PYTHONUNBUFFERED=1

# Step 3: Set the working directory
WORKDIR /app

# Step 4: Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the app code into the container
COPY . /app

# Step 6: Expose the port Streamlit runs on
EXPOSE 8501

# Step 7: Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]