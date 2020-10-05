import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras
import argparse
from tensorflow.python.keras.callbacks import Callback
import numpy as np
from tensorflow.python.lib.io import file_io
import json

import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
# parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
# parser.add_argument('--dropout_rate', required=False, type=float, default=0.3)  
parser.add_argument('--model_path', required=False, default='/result/saved_model',type = str)  
parser.add_argument('--model_version', required=False, default='1',type = str)
args = parser.parse_args()    
version = 100

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set. 


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

export_path = os.path.join(args.model_path, str(args.model_version))
model=keras.models.load_model(export_path)

quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


q_aware_model.summary()

train_images_subset = train_images[0:1000] # out of 60000
train_labels_subset = train_labels[0:1000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=5, validation_split=0.1)


results = q_aware_model.evaluate(test_images,test_labels, verbose = 0)
print('test loss, test acc:', results)
loss = results[0]
accuracy = results[1]
metrics = {
    'metrics': [{
        'name': 'accuracy',
        'numberValue': float(accuracy),
        'format': "PERCENTAGE",
    }, {
        'name': 'loss',
        'numberValue': float(loss),
        'format': "RAW",
    }]
}
with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
    json.dump(metrics, f)


# #path = args.saved_model_dir + "/" + args.model_version
# export_path = os.path.join(args.model_path, str(version))
# model.save(export_path, save_format="tf")
# #model.save('fmnist_model.h5')

_, quantized_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(q_aware_model, quantized_keras_file, include_optimizer=False)
print('Saved quantized Keras model to:', quantized_keras_file)


def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes
    import os
    import zipfile
     
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

print("Size of gzipped quantizied Keras model: %.2f bytes\n" % (get_gzipped_model_size(quantized_keras_file)))
print("Size of gzipped quantizied Keras model: %.2f bytes\n" % (get_gzipped_model_size(export_path + '/saved_model.pb')))


