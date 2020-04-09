#!/bin/bash

git clone https://github.com/ssiwapol/time-series-forecasting
mv config.yaml time-series-forecasting/config.yaml
cd time-series-forecasting
chmod 777 run.sh
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install Flask-Markdown
python3 app.py
