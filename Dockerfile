FROM python:3.11-slim
ARG OPENAI_KEY
ENV OPENAI_KEY=$OPENAI_KEY
ENV PORT 8000

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY ./src /src

CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}