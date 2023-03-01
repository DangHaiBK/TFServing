import tensorflow as tf

# Path to Keras and TensorFlow SavedModel models
path_models = 'original_model/'

path_saved_models_pnet = 'converted_models/pnet/1'
path_saved_models_rnet = 'converted_models/rnet/1'
path_saved_models_onet = 'converted_models/onet/1'


'''
Convert Keras model into SavedModel TensorFlow
'''

# Load keras model
onet_model = tf.compat.v1.keras.models.load_model(path_models + 'onet.h5')
pnet_model = tf.compat.v1.keras.models.load_model(path_models + 'pnet.h5')
rnet_model = tf.compat.v1.keras.models.load_model(path_models + 'rnet.h5')

# mtcnn_models = [pnet_model, rnet_model, onet_model]

# Convert models
pnet_model.save(path_saved_models_pnet)
rnet_model.save(path_saved_models_rnet)
onet_model.save(path_saved_models_onet)


print("Done to convert models")
