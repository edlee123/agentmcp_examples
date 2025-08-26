FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY data_tool.py .

# Expose the port
EXPOSE 8080

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set ownership of app files to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Run the server
CMD ["python", "data_tool.py"]