# HLS Design Notes

This folder holds draft C/C++ sources for implementing custom layers or
pre/post-processing blocks in Vivado HLS.

Refer to `yolov2_xilinx_fpga/hls` for example code that targets earlier YOLO
models. The same structure can be reused for convolution and activation
modules adapted to YOLOv7.

The goal is to create IP cores that can be connected to the KV260 programmable
logic either stand‑alone or alongside the Xilinx DPU.
