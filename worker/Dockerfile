FROM python:3.8

ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0

ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

COPY . /worker
WORKDIR /worker
RUN pip install -r requirements.txt

ENTRYPOINT celery -A tasks worker --loglevel=info