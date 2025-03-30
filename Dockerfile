# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Install the export plugin
RUN poetry self add poetry-plugin-export

# --- IMPORTANT: Copy BOTH pyproject.toml AND the lock file ---
# You need the lock file for export to work correctly.
COPY pyproject.toml poetry.lock* ./

# Export dependencies using the --output flag (cleaner than redirection)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy requirements from builder
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY .env .

# Set the entrypoint
CMD ["python", "-m", "src.main"]
