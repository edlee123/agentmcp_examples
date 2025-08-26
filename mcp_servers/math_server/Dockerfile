FROM python:3.11-slim

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY math_server.py .

# Set ownership of app files to the non-root user
RUN chown -R appuser:appgroup /app

EXPOSE 8081

# Switch to non-root user
USER appuser

CMD ["python", "math_server.py"]