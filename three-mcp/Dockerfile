FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the requirements file
COPY requirements.txt .

# Install dependencies using uv from the lock file, using system Python
RUN uv pip sync --system requirements.txt

# Copy the server code last, to leverage Docker cache for dependencies
COPY mcp_server.py .

EXPOSE 8000

CMD ["python", "mcp_server.py"] 