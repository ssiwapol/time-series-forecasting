#!/bin/bash

# set parameter
IMAGENAME=time-series-forecasting
CONTAINERNAME=forecast-api
PORT=$(grep 'PORT:' config.yaml); PORT=${PORT/PORT: /}; PORT=$(echo $PORT|tr -d '\r')

# clone app
git clone https://github.com/ssiwapol/time-series-forecasting
mv config.yaml time-series-forecasting/config.yaml
cd time-series-forecasting

# dockerize app
docker rm -f $CONTAINERNAME
docker rmi $IMAGENAME
docker image build -t $IMAGENAME .
docker run --name $CONTAINERNAME -v $(pwd):/app -d -p $PORT:$PORT $IMAGENAME
