# Use AWS Lambda Python base image
# This includes Lambda runtime environment + Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Set working directory inside container
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements file first (Docker caching optimization)
# If requirements.txt doesn't change, Docker reuses this layer
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't store package cache (keeps image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into container
# Only src and data are required for deployment. Omit evaluation scripts.
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY data/ ${LAMBDA_TASK_ROOT}/data/

# Expose port
EXPOSE 8000

# Run application
CMD ["src.api.main.handler"]
