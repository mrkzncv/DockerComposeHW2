FROM python:3.8

ENV HOST 0.0.0.0
ENV PORT 5000

ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0
ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

ENV POSTGRES_USER postgres
ENV POSTGRES_PASSWORD password
ENV POSTGRES_DB postgres_db

COPY . /flask-app
WORKDIR /flask-app
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["cli.py"]