# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . .

# Step 4: Upgrade pip
RUN python -m pip install --upgrade pip

# Step 5: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Define commands to execute each script in the desired order
CMD ["bash", "-c", "python text_setting.py && \
                   python text_exploration.py && \
                   python text_clustering.py && \
                   python text_classification.py && \
                   python text_clsf_usage_model.py"]
