#!/usr/bin/env python3
"""YOLOv7 inference on KV260 using Vitis AI runtime."""

import argparse
import cv2
import numpy as np
import xir
import vart


def get_dpu_subgraph(graph):
    """Recursively obtain DPU subgraphs from XIR graph."""
    assert graph is not None
    root = graph.get_root_subgraph()
    subgraphs = []

    def dfs(sg):
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
            subgraphs.append(sg)
        for c in sg.get_children():
            dfs(c)

    dfs(root)
    return subgraphs


def preprocess(img, input_shape):
    """Resize and normalize image to the model input size."""
    h, w = input_shape[1], input_shape[2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output, conf_thres=0.4):
    """Placeholder for YOLOv7 postprocessing."""
    # TODO: implement anchor decoding and NMS.
    return output


def main():
    parser = argparse.ArgumentParser(description="YOLOv7 KV260 inference")
    parser.add_argument('--model', required=True, help='Path to compiled .xmodel')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--labels', help='Optional labels file')
    parser.add_argument('--output', default='result.jpg', help='Output image path')
    args = parser.parse_args()

    graph = xir.Graph.deserialize(args.model)
    subgraphs = get_dpu_subgraph(graph)
    runner = vart.Runner.create_runner(subgraphs[0], 'run')

    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]
    input_shape = input_tensor.dims
    output_shape = output_tensor.dims

    image = cv2.imread(args.image)
    input_data = preprocess(image, input_shape)
    output_data = np.empty(output_shape, dtype=np.float32)

    job_id = runner.execute_async([input_data], [output_data])
    runner.wait(job_id)

    detections = postprocess(output_data)
    # TODO: draw boxes on image once postprocess is implemented
    cv2.imwrite(args.output, image)
    print(f'Saved result to {args.output}')


if __name__ == '__main__':
    main()
