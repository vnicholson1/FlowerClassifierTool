# FlowerClassiferTool

A simple tool using kNN and tiny images for flower classifiction to evaluate their effectiveness.

Feature extraction methods (training balanced accuracy):

Best tiny images (colour recognition only): 52.155204692984505% k = 1 and 8x8 size

Best edges only: 28.501297248451742%, k = 1 and 12x12 size

Best tiny images and edges: 55.60685909246514%, k = 1. 8x8 tiny image and 16x16 edge image.

# Flask app

To launch, go into the root directory and run

    pip install -r requirements.txt

## Run as an application

Once the requirements are installed, use the following commands to launch the server

    cd server
    python app.py

## Run as a service

    sudo cp flower-classifier.service /lib/systemd/system/flower-classifier.service
    sudo systemctl enable flower-classifier
    sudo systemctl start flower-classifier
    sudo systemctl status flower-classifier

