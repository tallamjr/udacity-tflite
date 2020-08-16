# Udacity's Introduction to TensorFlow Lite

> Course Structure
> In this course you will learn how to deploy TensorFlow lite models on Android, iOS, and IoT devices.
> This program is divided into 4 main lessons:

> Lesson 2 – Introduction to TensorFlow Lite

> Lesson 3 – TF Lite on Android

> Lesson 4 – TF Lite on iOS with Swift

> Lesson 5 – TF Lite on IoT

> Lesson 2 is aimed for all type of developers so that they learn how TensorFlow Lite works under the
> hood. We recommend everybody goes through lesson 2 regardless of their platform. The next 3 lessons
> (lessons 3, 4, and 5) are designed to be completely independent of each other and therefore, they
> all share similar sections that cover the same topics. Consequently, if you are an Android developer
> you can take lessons 2 and 3; if you are an iOS developer you can jump straight to lesson 4 after
> taking lesson 2; and if you are interested in deploying your models on the Raspberry Pi (or other
> IoT devices) you can jump straight to lesson 5 after taking lesson 2.

> We hope you enjoy this course!

> The Apps
> In this course you will deploy TF Lite models in various apps including:

    * Cats vs. Dogs: An app that classifies images of cats and dogs.
    * Image Classification: An app that continuously classifies whatever it sees from your device's back
    camera.
    * Objection Detection: An app that continuously detects the objects (bounding boxes and classes) in
    the frames seen by your device's back camera.
    * Speech Recognition: An app that recognizes the words you say.

> You can find all the apps for this course in this [Zip
file](https://video.udacity-data.com/topher/2019/September/5d8e8cb3_tflite-apps/tflite-apps.zip).
> Once you unzip the file you will find the folders containing the corresponding apps for each
> lesson. We will also provide links to the individual apps later in each lesson.

## Lesson 2: Overview

![](./inc/lesson-2/deployment.png?raw=true)

### TensorFlow Lite Examples

#### Example 1: SavedModel to TFLite

```python
import pathlib

# Export the SavedModel
export_dir = '/tmp/saved_model'
tf.saved_model.save(model, export_dir)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

# Save the model
tflite_model_file = pathlib.Path('/tmp/foo.tflite')
tflite_model_file.write_bytpes(tflite_model)
```

#### Example 2: Keras to TFLite

```python
import tensorflow as tf
import pathlib

# Load the MobileNet tf.keras model
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
tflite_model_file = pathlib.Path('/tmp/foo.tflite')
tflite_model_file.write_bytpes(tflite_model)
```

#### Example 3: Concrete functions to TFLite

```python
import tensorflow as tf

# Load the MobileNet tf.keras model
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

# Get the concrete function from Keras model
run_model = tf.function(lambda x: model(x))

# Save the concrete function
concrete_func = run_model.get_concrete_function(tf.TensorSpec(model.input[0].shape,
                                                              model.inputs[0]dtype))

# Save the model
converter = tf.lite.TFLiteConverter.from_concrete_function([concrete_func])
tflite_model = converter.convert()

```

#### Command-Line usage

```bash
#!/bin/bash

# Saving with the command-line from a SavedModel
tflite_convert --output_file=model.tflite --saved_model_dir=/tmp/saved_model

# Saving with the command-line from Keras model
tflite_convert --output_file=model.tflite --keras_model_file=model.h5
```

### Post-training quantization

![](https://www.tensorflow.org/lite/performance/images/optimization.jpg)

If you don’t intend to quantize your model, you’ll end up with a floating point model. Also,
remember that the converter will do its best to quantize all the operations (ops), but your model
may still end up with a few floating point ops.

It is important to note that even though post-training quantization works really well,
quantization-aware training generally results in a model with higher accuracy because it makes the
model more tolerant to lower precision values. Therefore, quantization-aware training should be used
in cases where the loss of accuracy brought by post-training quantization is beyond acceptable
thresholds.

To learn more about Post-Training Quantization make sure to check out the [TF Lite Documentation](https://www.tensorflow.org/lite/performance/post_training_quantization)

### TF Lite Delegates

What is a TensorFlow Lite delegate?

A TensorFlow Lite delegate is a way to delegate part or all of graph execution to another executor.

![](./inc/lesson-2/delegates.png?raw=true)

* https://www.tensorflow.org/lite/performance/delegates

### TF Lite Models

You can view the collection of pre-trained TFLite models on the link below:

[TF Lite Models](https://www.tensorflow.org/lite/models)

### Colab Notebook

To access the Colab Notebook, login to your Google account and click on the link below:

[Linear Regression](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c01_linear_regression.ipynb)

[Transfer Learning](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c02_transfer_learning.ipynb)

#### Exercise: Fashion MNIST

In this exercise you will train a simple convolutional
neural network on the Fashion MNIST dataset. Once your model is trained, you will convert it to a TF
Lite model and then test it using the TF Lite Interpreter.

[Exercise: Fashion MNIST](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c03_exercise_convert_model_to_tflite.ipynb)

[Solution: Fashion MNIST](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c04_exercise_convert_model_to_tflite_solution.ipynb)

## Lesson 3 – TF Lite on Android

## Lesson 4 – TF Lite on iOS with Swift

## Lesson 5 – TF Lite on IoT
