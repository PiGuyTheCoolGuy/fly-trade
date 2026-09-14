FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY flytrade ./flytrade
RUN pip install --no-cache-dir .
COPY run.py config.toml ./
VOLUME /app/data
CMD ["python", "run.py", "run", "--no-dashboard"]
