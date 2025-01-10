FROM python:3.11-alpine

RUN pip install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml LICENSE README.md ./
COPY src src

RUN poetry install --no-cache

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "cloaiservice.main:app", "--host", "0.0.0.0", "--port", "8000"]