#!/bin/bash

python3 create_data_points.py
sudo systemctl restart flower-classifier.service
