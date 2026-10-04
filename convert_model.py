from pathlib import Path

INPUT_FILE = Path("light_level_classifier.tflite")
OUTPUT_FILE = Path("model.h")

data = INPUT_FILE.read_bytes()

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    f.write("#ifndef MODEL_H\\n#define MODEL_H\\n\\n")
    f.write("#include <stddef.h>\\n\\n")
    f.write("alignas(8) const unsigned char g_model[] = {\\n")

    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\\n")

    f.write("};\\n")
    f.write("const unsigned int g_model_len = sizeof(g_model);\\n")
    f.write("\\n#endif\\n")

print(f"Generated {OUTPUT_FILE} from {INPUT_FILE}")
