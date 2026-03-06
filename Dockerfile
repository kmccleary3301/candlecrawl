FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
RUN uv sync --no-dev

COPY app ./app
COPY contracts ./contracts

ENV PORT=3010

EXPOSE 3010

CMD ["uv", "run", "python", "-m", "app.main"]
