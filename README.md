# First TFLite Micro Inference

**Author:** suriyakumar P

## Task

Deploy a small `.tflite` model on an ESP32 using TensorFlow Lite Micro and predict a class from a sample input.

## Project Structure

- `sketch.ino` - ESP32 TensorFlow Lite Micro inference program
- `model.h` - C byte-array header generated from the TFLite model
- `convert_model.py` - Converts `light_level_classifier.tflite` into `model.h`
- `Light_Level_Model_Training.ipynb` - Google Colab notebook for training and conversion
- `diagram.json` - Wokwi ESP32 project diagram

## Model

The example classifies a normalized light-level input into:

- **0 - Dark**
- **1 - Normal**
- **2 - Bright**

A sample input of **0.90** represents a bright light level.

## Workflow

1. Train a small TensorFlow classifier in Colab.
2. Convert the Keras model to `light_level_classifier.tflite`.
3. Run `convert_model.py` to generate `model.h`.
4. Copy `model.h` into the ESP32 Arduino project.
5. Compile and upload `sketch.ino`.
6. TensorFlow Lite Micro runs inference on the ESP32 and prints the predicted class.

## ESP32

The sketch allocates a TensorFlow Lite Micro tensor arena and uses the Micro Interpreter to run inference without requiring a full TensorFlow installation on the ESP32.

## Expected Output

For a sample input of 0.90:

```
TFLite Micro Inference
Sample input: 0.90
Predicted class: Bright
```

## Libraries

Install an ESP32-compatible TensorFlow Lite Micro / TensorFlowLite library in Arduino IDE.
