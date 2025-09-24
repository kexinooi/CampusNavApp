#!/bin/bash
apt-get update && apt-get install -y libgl1
pip install --upgrade pip
pip install -r requirements.txt
guvicorn main:app --host 0.0.0.0 --port 8000
apt-get update && apt-get install -y libgl1
