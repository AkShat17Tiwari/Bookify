# Use the official Python image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies (e.g. for sqlite, building wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as a non-root user (id 1000)
# Create a user to avoid permission issues
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create app directory with user permissions
WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose the default port expected by Hugging Face Spaces
EXPOSE 7860

# Run the application using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "4", "app:app"]
