# YOLOv7 on Xilinx KV260

This directory provides a minimal example showing how to run a quantized YOLOv7 model on the Xilinx KV260 acceleration board using the Vitis AI runtime.

## Prerequisites

- A KV260 board running the official Vitis AI image.
- YOLOv7 model compiled to an `.xmodel` file using the Vitis AI compiler (`vai_c_pt`).
- Python dependencies available on the board: `xir`, `vart`, `opencv-python`, `numpy`.

## Usage

1. Copy the compiled model and this script to the KV260.
2. Execute the inference script on an input image:

```bash
python3 kv260_inference.py --model yolov7_kv260.xmodel --image test.jpg --output result.jpg
```

The script loads the `.xmodel`, performs inference on the provided image and saves the result. Postprocessing code such as anchor decoding and NMS should be added in `kv260_inference.py` to draw detection boxes.

## Notes

This example is a starting point and may require adaptation depending on the quantization parameters and target input size used during model compilation.
