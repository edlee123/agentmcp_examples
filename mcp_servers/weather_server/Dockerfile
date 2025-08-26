FROM python:3.11-slim

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY weather_server.py .

# Set ownership of app files to the non-root user
RUN chown -R appuser:appgroup /app

EXPOSE 8080

# Switch to non-root user
USER appuser

# Run the Python script directly, which will call mcp.run(transport="sse")
CMD ["python3", "weather_server.py"]