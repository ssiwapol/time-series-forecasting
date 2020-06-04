FROM python:3.7-slim
RUN apt-get update && apt-get install gcc g++ python-dev -y
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN chmod 777 run.sh
CMD ["python", "app.py"]
