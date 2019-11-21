from tensorflow import keras

print(keras.backend.backend())
print(keras.backend.epsilon())
print(keras.backend.image_data_format())
print(keras.backend.floatx())

keras.backend.set_image_data_format('channels_first')
print(keras.backend.image_data_format())

