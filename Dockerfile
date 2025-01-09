FROM python:3.11-slim

RUN pip install poetry

WORKDIR /app
COPY . .

RUN poetry install

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "cloaiservice.main:app", "--host", "0.0.0.0", "--port", "8000"]