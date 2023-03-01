# TFServing

Run MTCNN model on Tensorflow server in Docker and Local server

The original MTCNN model has been separated into three sub-models, including P-net, R-net, and O-net. These models are keras model format, in order to use them on tensorflow server, we need to convert them into savedmodel format. Because of multiple models, we utilized a file, named "serving.config" to list and configure all models.  

For Docker, firstly, we need to pull serving image from tensorflow
* ~$ docker pull tensorflow/serving

Secondly, we need to bind the project folder on Docker to run a container
* ~$ docker run -p 8500:8500 -p 8501:8501 \
* --mount type=bind,source="$(pwd)/TFServing/converted_models/pnet/,target=/models/pnet \
* --mount type=bind,source="$(pwd)/TFServing/converted_models/rnet/,target=/models/rnet \
* --mount type=bind,source="$(pwd)/TFServing/converted_models/onet/,target=/models/onet \
* --mount type=bind,source="$(pwd)/TFServing/serving.config,target=/models/serving.config \
* -t tensorflow/serving --model config_file=/models/serving.config

A container will be created and run, and Server starts.

In the folder "main_execution":
* 
