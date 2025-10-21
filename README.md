# FlowerClassiferTool

Data has come from - https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ provided by the Oxford University.

A simple tool using a trained CNN to classify pictures of flowers.

# Updating training data

Add any new images to the `train` folder and run `create_datapoints.py` to create the JSON file of training data.

# Flask app

To launch, go into the root directory and run

    pip install -r requirements.txt

## Run as an application

Once the requirements are installed, use the following commands to launch the server

    cd server
    python app.py

## Run inside docker (recommended)

docker build -t flower-classifier .
docker run -d -p 4000:4000 flower-classifier


# Android app

