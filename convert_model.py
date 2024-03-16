import subprocess
import sys


def run_commands(model_name):
    commands = [
        f"python torch2onnx.py --weights {model_name}.pt --output {model_name}.onnx",
        f"python add_nms_plugins.py --model {model_name}.onnx",
        f"/usr/src/tensorrt/bin/trtexec --onnx={model_name}-nms.onnx --saveEngine={model_name}-nms.trt --minShapes=input:1x3x640x640 --maxShapes=input:3x3x640x640 --optShapes=input:3x3x640x640 --verbose --device=0",
        f"python3 object_detector_trt_nms.py --classes data/tiip.names --weight {model_name}-nms.trt"
    ]

    for cmd in commands:
        process = subprocess.run(cmd, shell=True, check=True)
        print(f"Executed: {cmd}")
        

def run_commands_e2e(model_name):
    e2e_model_name = f"{model_name}-e2e"
    commands = [
        f"python torch2onnx.py --weights {model_name}.pt --output {e2e_model_name}.onnx --e2e",
        f"python add_nms_plugins.py --model {e2e_model_name}.onnx",
        f"/usr/src/tensorrt/bin/trtexec --onnx={e2e_model_name}-nms.onnx --saveEngine={e2e_model_name}-nms.trt --minShapes=input:1x1280x1920x3 --maxShapes=input:3x1280x1920x3 --optShapes=input:3x1280x1920x3 --verbose --device=0",
        f"rm -rf images/infer/",
        f"python3 object_detector_trt_nms.py --classes data/tiip.names --weight {e2e_model_name}-nms.trt --e2e"
    ]

    for cmd in commands:
        process = subprocess.run(cmd, shell=True, check=True)
        print(f"Executed: {cmd}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--e2e', action='store_true', help='Enable end-to-end mode')
    opt = parser.parse_args()

    if opt.e2e:
        run_commands_e2e(opt.weights)
    else:
        run_commands(opt.weights)