FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

# Hosting platforms such as Render and Fly.io pass the port in $PORT.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn odsim.web.app:app --host 0.0.0.0 --port ${PORT}"]
