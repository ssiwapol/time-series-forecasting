FROM python:3.7-slim
RUN apt-get update && apt-get install gcc g++ python-dev -y
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
