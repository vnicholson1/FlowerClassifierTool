#!/bin/bash

python3 create_data_points.sh
sudo systemctl restart flower-classifier.service