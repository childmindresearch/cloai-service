FROM python:3.11-alpine

ENV PORT=8000
RUN pip install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml LICENSE README.md ./
COPY src src

RUN poetry install --no-cache

EXPOSE $PORT

CMD ["sh", "-c", "poetry run uvicorn cloaiservice.main:app --host 0.0.0.0 --port $PORT"]