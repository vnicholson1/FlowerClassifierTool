#!/bin/bash

cd /home/vincent/FlowerClassifierTool/
source venv/bin/activate
python3 create_data_points.py
sudo systemctl restart flower-classifier.service
