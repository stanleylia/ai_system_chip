# KV260 YOLOv7 Accelerator

This directory collects early work toward accelerating YOLOv7 on the Xilinx KV260 board.
It serves as a place holder for future hardware designs.

The software inference example under `yolov7/deploy/kv260` demonstrates how to run a
precompiled `.xmodel` with the Vitis AI runtime. Here we focus on hardware design
options using either Vivado HLS or custom Verilog.

## Reference Designs

* `yolov2_xilinx_fpga/hls` – HLS based accelerator for YOLOv2. Useful as a starting
  point for building new HLS modules for KV260.
* `YOLO-on-PYNQ-Z2/DPU implementation` – Example of integrating the Xilinx DPU in a
  complete Vivado and Petalinux flow. The same approach can be applied to KV260.
* `yolov5-fpga` – Software and scripts for YOLOv5 acceleration which can inspire the
  build and deployment process.

## Directory Layout

```
kv260_yolov7/
  README.md        # High level plan (this file)
  hls/             # Place holder for HLS sources
  verilog/         # Place holder for RTL sources
```

Additional scripts or notebooks can be placed here as development progresses.
