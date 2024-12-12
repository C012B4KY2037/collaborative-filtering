FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
