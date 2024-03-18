import argparse
import os
import platform
import shutil
import time
import pathlib

import cv2
import numpy as np
from numpy import random

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt


TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


def compute_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def nms(bboxes, scores, score_threshold=0.4, iou_threshold=0.6):
    # Flatten the bboxes and scores array for simplicity
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.flatten()

    # Filter out bboxes with scores below threshold
    idxs = np.where(scores > score_threshold)[0]
    filtered_bboxes = bboxes[idxs]
    filtered_scores = scores[idxs]

    # Sort the indices of the filtered_scores in descending order
    idxs = np.argsort(filtered_scores)[::-1]

    keep = []
    while len(idxs) > 0:
        # Take the top-scored bounding box and add its index to the keep list
        current_idx = idxs[0]
        keep.append(current_idx)

        if len(idxs) == 1:
            break

        # Compute IoU of the top-scored bounding box with all others
        ious = np.array([compute_iou(filtered_bboxes[current_idx], filtered_bboxes[i]) for i in idxs[1:]])

        # Keep only boxes with IoU less than the threshold
        idxs = idxs[1:][ious < iou_threshold]

    # Return the indices of the bounding boxes to keep
    return keep


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, max_boxes, total_classes):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    print('Profile shape: ', engine.get_profile_shape(0, 0))
    # max_batch_size = 1
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        #Fix -1 dimension for proper memory allocation for batch_size > 1
        if binding == 'input':
            max_width = engine.get_profile_shape(0, 0)[2][3]
            max_height = engine.get_profile_shape(0, 0)[2][2]
            size = max_batch_size * max_width * max_height * 3
        else:
            size = max_batch_size * max_boxes * (total_classes + 5)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(engine.get_binding_shape(binding))
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size

def allocate_buffers_nms(engine, max_width, max_height, max_batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    # max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    print('Profile shape: ', engine.get_profile_shape(0, 0))
    # import ipdb; ipdb.set_trace()
    # max_batch_size = 1
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        print('binding:', binding, '- binding_shape:', binding_shape)
        #Fix -1 dimension for proper memory allocation for batch_size > 1
        if binding == 'input':
            # max_width = engine.get_profile_shape(0, 0)[2][3]
            # max_height = engine.get_profile_shape(0, 0)[2][2]
            size = max_batch_size * max_width * max_height * 3
        else:
            binding_shape = (max_batch_size,) + binding_shape[1:]
            size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(binding_shape[1:])
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size
# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class TrtModel(object):
    def __init__(self, model, max_size, total_classes = 80):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.input_shapes = None
        self.out_shapes = None
        self.max_batch_size = 1
        self.max_size = max_size
        self.total_classes = total_classes

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # Allocate
        self.max_boxes = self.get_number_of_boxes(self.max_size, self.max_size)
        print('Maximum image size: {}x{}'.format(self.max_size, self.max_size))
        print('Maximum boxes: {}'.format(self.max_boxes))
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = \
                allocate_buffers(self.engine, max_boxes = self.max_boxes, total_classes = self.total_classes)
        print(self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size)
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0

    def get_number_of_boxes(self, im_width, im_height):
        # Calculate total boxes (3 detect layers)
        assert im_width % 32 == 0 and im_height % 32 == 0
        return (int(im_width*im_height/32/32) + int(im_width*im_height/16/16) + int(im_width*im_height/8/8))*3

    def run(self, input, deflatten: bool = True, as_dict = False):
        # lazy load implementation
        if self.engine is None:
            self.build()

        input = np.asarray(input)
        batch_size, _, im_height, im_width = input.shape
        assert batch_size <= self.max_batch_size
        assert max(im_width, im_height) <= self.max_size, "Invalid shape: {}x{}, max shape: {}".format(im_width, im_height, self.max_size)
        allocate_place = np.prod(input.shape)
        # print('allocate_place', input.shape)
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        self.context.set_binding_shape(0, input.shape)
        trt_outputs = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # print(self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size)
        # Reshape TRT outputs to original shape instead of flattened array
        # print(trt_outputs[0].shape)
        if deflatten:
            out_shapes = [(batch_size, ) + (self.get_number_of_boxes(im_width, im_height), 85)]
            trt_outputs = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        if as_dict:
            return {self.out_names[ix]: trt_output[:batch_size] for ix, trt_output in enumerate(trt_outputs)}
        return [trt_output[:batch_size] for trt_output in trt_outputs]


class TrtModelNMS(object):
    def __init__(self, model, max_size, max_width, max_height, max_batch_size):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.input_shapes = None
        self.out_shapes = None
        self.max_size = max_size

        self.max_width = max_width
        self.max_height = max_height
        self.max_batch_size = max_batch_size

    def build(self):
        print(f'Building engine from {self.engine_file}')
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # Allocate
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = \
                allocate_buffers_nms(self.engine, max_width=self.max_width, max_height=self.max_height, max_batch_size=self.max_batch_size)
        print(self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size)
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0

    def run(self, input, deflatten: bool = True, as_dict = False):
        # lazy load implementation
        if self.engine is None:
            self.build()

        input = np.asarray(input)
        batch_size, _, im_height, im_width = input.shape
        assert batch_size <= self.max_batch_size
        # assert max(im_width, im_height) <= self.max_size, "Invalid shape: {}x{}, max shape: {}".format(im_width, im_height, self.max_size)
        allocate_place = np.prod(input.shape)
        # print('allocate_place', input.shape)
        # import ipdb; ipdb.set_trace()
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        self.context.set_binding_shape(0, input.shape)
        trt_outputs = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # print(self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size)
        # Reshape TRT outputs to original shape instead of flattened array
        # print(trt_outputs[0].shape)
        if deflatten:
            out_shapes = [(batch_size, ) + self.out_shapes[ix] for ix in range(len(self.out_shapes))]
            trt_outputs = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        if as_dict:
            return {self.out_names[ix]: trt_output[:batch_size] for ix, trt_output in enumerate(trt_outputs)}
        return [trt_output[:batch_size] for trt_output in trt_outputs]

# from exec_backends.trt_loader import TrtModelNMS
# from models.models import Darknet

def letterbox(img, new_shape=(448, 448), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class YOLOv9(object):
    def __init__(self, 
            max_width,
            max_height,
            end2end,
            max_batch_size,
            model_weights = 'weights/yolov5-nms.trt', 
            names = 'data/coco.names',
            ):
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        max_size = max(max_width, max_height)
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size, max_width=max_width, max_height=max_height, max_batch_size=max_batch_size)
        self.end2end = end2end


    def detect(self, bgr_img):   
        if self.end2end:
            inp = bgr_img.astype('float32') 
            inp = np.expand_dims(inp, 0)
        else:
            ## Padded resize
            h, w, _ = bgr_img.shape
            scale = min(self.imgsz[0]/w, self.imgsz[1]/h)
            inp = np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype = np.float32)
            nh = int(scale * h)
            nw = int(scale * w)
            inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (nw, nh))
            inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
            inp = np.expand_dims(inp.transpose(2, 0, 1), 0)


        # infer for int8
        confidence_score = 0.6
        ## Inference
        t1 = time.perf_counter_ns()
        bboxes, scores = self.model.run(inp)
        cats = np.argmax(scores, axis=2)
        scores = np.reshape(np.max(scores, axis=2), -1) 
        bboxes = np.reshape(bboxes, (-1, 4))
        nmsed_bboxes = bboxes[scores > confidence_score]
        nmsed_scores  = scores[scores > confidence_score]
        cats = np.reshape(cats, -1)
        nmsed_classes = cats[scores > confidence_score]
        t2 = time.perf_counter_ns()
        num_detection = nmsed_bboxes.shape[0]

        # ## infer for fp32
        # t1 = time.perf_counter_ns()
        # num_detection, nmsed_bboxes, nmsed_scores, nmsed_classes = self.model.run(inp)
        # t2 = time.perf_counter_ns()
        # num_detection = num_detection[0][0]
        # nmsed_bboxes  = nmsed_bboxes[0]
        # nmsed_scores  = nmsed_scores[0]
        # nmsed_classes  = nmsed_classes[0]


        print('Detected {} object(s)'.format(num_detection))
        # Rescale boxes from img_size to im0 size
        _, _, height, width = inp.shape
        h, w, _ = bgr_img.shape

        if not self.end2end:
            nmsed_bboxes /= scale

        visualize_img = bgr_img.copy()
        for ix in range(num_detection):       # x1, y1, x2, y2 in pixel format
            cls = int(nmsed_classes[ix])
            label = '%s %.2f' % (self.names[cls], nmsed_scores[ix])
            x1, y1, x2, y2 = nmsed_bboxes[ix]

            cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(cls)], 2)
            cv2.putText(visualize_img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[int(cls)], 2, cv2.LINE_AA)

        return visualize_img, t2-t1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')
    parser.add_argument('--classes', type=str, default='data/coco.names', help='classes name file path')
    # parser.add_argument('--max_size', type=int, default=640, help='max size of input image')
    parser.add_argument('--e2e', action='store_true', help='Enable end-to-end mode')
    # parser.add_argument('--img_test', type=str, default='images/zidane.jpg', help='image test path')
    opt = parser.parse_args()

    if opt.e2e:
        model = YOLOv9(1920, 1280, True, 3, opt.weights, opt.classes)
    else:
        model = YOLOv9(640, 640, False, 4, opt.weights, opt.classes)


    image_root = 'images/samples' # on msi
    # image_root = '/home/ubuntu/julian/tiip/data/tiip-s4-1000/tiip-s4-1000/' # on orin
    out_root = 'images/infer' # on my orin
    image_list = sorted(pathlib.Path(image_root).glob('*.jpg'))

    # select random 100 images
    num_data = 100
    infer_list = np.random.choice(image_list, 100)
    # infer_list = image_list[:100]
    # infer_list = image_list
    pathlib.Path(out_root).mkdir(exist_ok=True)

    total_elapsed_time_ns = 0
    for image_path in infer_list:
        img = cv2.imread(str(image_path))
        # image = Image.open(str(image_path)).convert("RGB")
        result_img, infer_time = model.detect(img)
        image_name = pathlib.Path(image_path).stem
        cv2.imwrite(f'{out_root}/{image_name}.jpg', result_img)
        print(f'{out_root}/{image_name}.jpg')
        total_elapsed_time_ns += infer_time

    num_data = len(infer_list)
    avg_elapsed_time_sec = total_elapsed_time_ns / 1_000_000_000 / num_data
    print(f"averaged infer time: ({avg_elapsed_time_sec} seconds) ({num_data} samples)")