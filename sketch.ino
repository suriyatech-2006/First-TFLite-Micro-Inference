#include <Arduino.h>
#include "model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
  constexpr int kTensorArenaSize = 12 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  tflite::MicroMutableOpResolver<5> resolver;
}

const char* className(int index)
{
  switch (index)
  {
    case 0: return "Dark";
    case 1: return "Normal";
    case 2: return "Bright";
    default: return "Unknown";
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println("TFLite Micro Inference");

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema version mismatch.");
    while (true) delay(1000);
  }

  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddReshape();

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("AllocateTensors() failed.");
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Sample normalized light level: 0.90 = Bright.
  const float sampleInput = 0.90f;

  if (input->type == kTfLiteFloat32)
  {
    input->data.f[0] = sampleInput;
  }
  else
  {
    Serial.println("This example expects a float32 model input.");
    while (true) delay(1000);
  }

  if (interpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("Inference failed.");
    return;
  }

  int predictedClass = 0;

  if (output->type == kTfLiteFloat32)
  {
    float bestScore = output->data.f[0];

    for (int i = 1; i < output->dims->data[output->dims->size - 1]; i++)
    {
      if (output->data.f[i] > bestScore)
      {
        bestScore = output->data.f[i];
        predictedClass = i;
      }
    }

    Serial.print("Sample input: ");
    Serial.println(sampleInput, 2);
    Serial.print("Predicted class: ");
    Serial.println(className(predictedClass));
  }
  else
  {
    Serial.println("This example expects a float32 model output.");
  }
}

void loop()
{
  // Inference is demonstrated once during setup.
}