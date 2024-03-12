# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import pathlib
import time
import random


CLASS_LIST = [
    # 'ANIMAL',
    # 'BARRICADE',
    # 'BICYCLE',
    # 'BUS',
    # 'BUS_STOP_SIGN',
    # 'CAR',
    # 'CONE',
    # 'ETC',
    # 'HUMAN_LIKE',
    # 'JERSEY_BARRIER',
    # 'MOTORCYCLE',
    # 'NON_UPRIGHT',
    # 'PEDESTRIAN',
    # 'POLE',
    # 'RIDER',
    # 'ROAD_CRACKS',
    # 'ROAD_PATCH',
    # 'ROAD_POTHOLES',
    # 'STOP_SIGN',
    # 'TRAFFIC_LIGHT',
    # 'TRAFFIC_SIGN',
    # 'TRUCK',
    # 'UNCLEAR_LANE_MARKING',
    # 'UNCLEAR_ROAD_MARKING',
    # 'UNCLEAR_STOP_LINE',
    # 'WHEELCHAIR',

    # old but wrong
    'CAR',
    'TRUCK',
    'BUS',
    'MOTORCYCLE',
    'BICYCLE',
    'WHEELCHAIR',
    'ETC',
    'PEDESTRIAN',
    'NON_UPRIGHT',
    'HUMAN_LIKE',
    'RIDER',
    'ANIMAL',
    'POLE',
    'TRAFFIC_SIGN',
    'TRAFFIC_LIGHT',
    'BUS_STOP_SIGN',
    'STOP_SIGN',
    'ROAD_CRACKS',
    'ROAD_PATCH',
    'ROAD_POTHOLES',
    'UNCLEAR_LANE_MARKING',
    'UNCLEAR_STOP_LINE',
    'UNCLEAR_ROAD_MARKING',
    'CONE',
    'BARRICADE',
    'JERSEY_BARRIER',
]

ID_TO_CLASS = { i: CLASS_LIST[i] for i in range(len(CLASS_LIST)) }


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def resize_box(box, org_size, new_size):
    org_h, org_w = org_size
    new_h, new_w = new_size
    ratio_h = new_h / org_h
    ratio_w = new_w / org_w
    new_box = np.zeros_like(box)
    new_box[0] = box[0] * ratio_w
    new_box[1] = box[1] * ratio_h
    new_box[2] = box[2] * ratio_w
    new_box[3] = box[3] * ratio_h
    return new_box


url = 'localhost:8001' 
triton_client = grpcclient.InferenceServerClient(
    url=url,
    verbose=False,
    ssl=False,
    root_certificates=None,
    private_key=None,
    certificate_chain=None,)

# image_root = '/home/julian/data/tainan-modelT/front_outside/front_outside'
# image_root = '/home/julian/data/indus-innov/raw-data/convert_data/01_10_2024/sensor_raw-01_10_2024-16_54_16/'
# image_root = '/home/julian/data/indus-innov/raw-data/convert_data/01_10_2024/sensor_raw-01_10_2024-16_22_58/'
# image_root = '/home/julian/data/indus-innov/0216/images_0216/kaohsiung5gsmartcitydemo/tiip-s4-1000/tiip-s4-1000/'
image_root = '/home/julian/data/indus-innov/images/kaohsiung5gsmartcitydemo/tiip-s4-1000/tiip-s4-1000/'
out_root = '/home/julian/work/yolov9-tensorrt/triton-trt-infer'


# load all *.jpg under image_root as list
image_list = sorted(pathlib.Path(image_root).glob('*.jpg'))
# select random 10 images
# infer_list = np.random.choice(image_list, 10)
infer_list = image_list[:10]
pathlib.Path(out_root).mkdir(exist_ok=True)

total_elapsed_time_ns = 0
for image_path in infer_list:
    # img_path = f'{image_root}/{image_name}.jpg'
    img = cv2.imread(str(image_path))
    org_img = img
    im_shape = np.array([[float(img.shape[0]), float(img.shape[1])]]).astype('float32')
    img = cv2.resize(img, (640, 640))
    scale_factor = np.array([[float(640/img.shape[0]), float(640/img.shape[1])]]).astype('float32')
    img = img.astype(np.float32) / 255.0
    input_img = np.transpose(img, [2, 0, 1])
    input_img = input_img[np.newaxis, :, :, :]

    model_name = 'yolov9'
    OUTPUT_NAMES = [
        "num_detections",
        "nmsed_boxes",
        "nmsed_scores",
        "nmsed_classes",
    ]
    INPUT_NAMES = [
        'input',
    ]
    width, height = 640, 640
    inputs = []
    outputs = []
    start_time = time.perf_counter_ns()
    # inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 2], "INT32"))
    # inputs.append(grpcclient.InferInput(INPUT_NAMES[1], [1, 3, width, height], "FP32"))
    inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, width, height], "FP32"))

    for output_name in OUTPUT_NAMES:
        outputs.append(grpcclient.InferRequestedOutput(output_name))

    # inputs[0].set_data_from_numpy(np.array([[width, height]], dtype=np.int32))
    # inputs[1].set_data_from_numpy(input_img.astype(np.float32))
    inputs[0].set_data_from_numpy(input_img.astype(np.float32))

    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    elapsed_time_ns = time.perf_counter_ns() - start_time
    total_elapsed_time_ns += elapsed_time_ns
    print(f"elapsed_time: {elapsed_time_ns} ns")

    # labels, boxes, scores = results.as_numpy(OUTPUT_NAMES[0]), results.as_numpy(OUTPUT_NAMES[1]),results.as_numpy(OUTPUT_NAMES[2])
    num_dets, boxes, scores, labels = results.as_numpy(OUTPUT_NAMES[0]), results.as_numpy(OUTPUT_NAMES[1]),results.as_numpy(OUTPUT_NAMES[2]), results.as_numpy(OUTPUT_NAMES[3])

    # import ipdb; ipdb.set_trace()

    confidence_threshold = 0.1**3
    print(confidence_threshold)

    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        if score < confidence_threshold:
            continue
        label = f'{ID_TO_CLASS[int(label)][:3]} {score:.2f}'
        resized_box = resize_box(box, (640, 640), org_img.shape[:2])
        plot_one_box(resized_box, org_img, label=label, color=(255, 0, 0), line_thickness=2)
    image_name = pathlib.Path(image_path).stem
    cv2.imwrite(f'{out_root}/{image_name}.jpg', org_img)
    print(f'{out_root}/{image_name}.jpg')

num_data = len(infer_list)
elapsed_time_sec = total_elapsed_time_ns / 1_000_000_000 / num_data
print(f"averaged infer time: {elapsed_time_ns} ns ({elapsed_time_sec} seconds) ({num_data} samples)")