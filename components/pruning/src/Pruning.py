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

# 모델 사이즈를 측정하기 위한 함수
def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
parser.add_argument('--dropout_rate', required=False, type=float, default=0.3)  
parser.add_argument('--model_path', required=False, default='/result/saved_model',type = str)  
parser.add_argument('--model_version', required=False, default='1',type = str)
args = parser.parse_args()    
#version = 1

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 4 epochs.
batch_size = 128
epochs = 5
validation_split = 0.1 # 10% of training set will be used for validation set. 

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.90,
                                                               begin_step=0,
                                                               end_step=end_step)
}

export_path = os.path.join(args.model_path, str(args.model_version))
model = keras.models.load_model(export_path)

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_for_pruning.summary()

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
  
model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

results = model_for_pruning.evaluate(test_images,test_labels, batch_size=128)
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
#path = args.saved_model_dir + "/" + args.model_version
export_path = os.path.join(args.model_path, str(args.model_version))
model.save(export_path, save_format="tf")
#model.save('fmnist_model.h5')

_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   test_images, test_labels, verbose=0)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

################# 정확도 ###############################
print('Pruned test accuracy:', model_for_pruning_accuracy)

################# 모델 size ##############################

print('base model size:', get_gzipped_model_size(export_path:'/saved_model.pb')
print('Pruned model size:', get_gzipped_model_size(pruned_keras_file))


