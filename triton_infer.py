# %%
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import pathlib
import time
import random


CLASS_LIST = [
    'ANIMAL',
    'BARRICADE',
    'BICYCLE',
    'BUS',
    'BUS_STOP_SIGN',
    'CAR',
    'CONE',
    'ETC',
    'HUMAN_LIKE',
    'JERSEY_BARRIER',
    'MOTORCYCLE',
    'NON_UPRIGHT',
    'PEDESTRIAN',
    'POLE',
    'RIDER',
    'ROAD_CRACKS',
    'ROAD_PATCH',
    'ROAD_POTHOLES',
    'STOP_SIGN',
    'TRAFFIC_LIGHT',
    'TRAFFIC_SIGN',
    'TRUCK',
    'UNCLEAR_LANE_MARKING',
    'UNCLEAR_ROAD_MARKING',
    'UNCLEAR_STOP_LINE',
    'WHEELCHAIR',
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


def preprocess(img, imgsz=(640, 640)):
    h, w, _ = img.shape
    scale = min(imgsz[0]/w, imgsz[1]/h)
    input_img = np.zeros((imgsz[1], imgsz[0], 3), dtype = np.float32)
    nh = int(scale * h)
    nw = int(scale * w)
    input_img[: nh, :nw, :] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh))
    input_img = input_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
    input_img = np.expand_dims(input_img.transpose(2, 0, 1), 0)
    return input_img, scale


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
# image_root = '/home/julian/data/indus-innov/images/kaohsiung5gsmartcitydemo/tiip-s4-1000/tiip-s4-1000/'
image_root = '/home/ubuntu/julian/tiip/data/tiip-s4-1000/tiip-s4-1000/' # on my orin
# out_root = '/home/julian/work/yolov9-tensorrt/triton-trt-infer'
out_root = '/home/ubuntu/julian/tiip/infer/' # on my orin
model_name = 'yolov9-c3'


# load all *.jpg under image_root as list
image_list = sorted(pathlib.Path(image_root).glob('*.jpg'))

# select random 10 images
# infer_list = np.random.choice(image_list, 10)
# infer_list = image_list[:10]
infer_list = image_list
pathlib.Path(out_root).mkdir(exist_ok=True)

total_elapsed_time_ns = 0
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(ID_TO_CLASS))]

for image_path in infer_list:
    org_img = cv2.imread(str(image_path))
    # org_img = org_img
    input_img, scale = preprocess(org_img)

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

    inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, width, height], "FP32"))

    # for batch infer
    # inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [3, 3, width, height], "FP32"))

    for output_name in OUTPUT_NAMES:
        outputs.append(grpcclient.InferRequestedOutput(output_name))

    # for batch infer
    # input2 = np.copy(input_img)
    # input3 = np.copy(input_img)
    # input_img = np.vstack((input_img, input2, input3))

    inputs[0].set_data_from_numpy(input_img.astype(np.float32))

    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    elapsed_time_ns = time.perf_counter_ns() - start_time
    total_elapsed_time_ns += elapsed_time_ns
    print(f"elapsed_time: {elapsed_time_ns} ns")

    num_dets, boxes, scores, labels = results.as_numpy(OUTPUT_NAMES[0]), results.as_numpy(OUTPUT_NAMES[1]),results.as_numpy(OUTPUT_NAMES[2]), results.as_numpy(OUTPUT_NAMES[3])

    ## Apply NMS
    # for batch infer, change first 0 to 0-2
    num_detection = num_dets[0][0]
    reshape_bboxes  = boxes[0]
    nmsed_scores  = scores[0]
    nmsed_classes  = labels[0]
    print('Detected {} object(s)'.format(num_detection))
    # Rescale boxes from img_size to im0 size
    _, _, height, width = input_img.shape
    h, w, _ = org_img.shape

    reshape_bboxes = np.copy(reshape_bboxes)
    reshape_bboxes[:, 0] /= scale
    reshape_bboxes[:, 1] /= scale
    reshape_bboxes[:, 2] /= scale
    reshape_bboxes[:, 3] /= scale
    org_img = org_img.copy()
    for ix in range(num_detection):       # x1, y1, x2, y2 in pixel format
        cls_id = int(nmsed_classes[ix])
        label = '%s %.2f' % (ID_TO_CLASS[cls_id], nmsed_scores[ix])
        x1, y1, x2, y2 = reshape_bboxes[ix]

        cv2.rectangle(org_img, (int(x1), int(y1)), (int(x2), int(y2)), colors[int(cls_id)], 2)
        cv2.putText(org_img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[int(cls_id)], 2, cv2.LINE_AA)

    image_name = pathlib.Path(image_path).stem
    cv2.imwrite(f'{out_root}/{image_name}.jpg', org_img)
    print(f'{out_root}/{image_name}.jpg')

num_data = len(infer_list)
elapsed_time_sec = total_elapsed_time_ns / 1_000_000_000 / num_data
print(f"averaged infer time: {elapsed_time_ns} ns ({elapsed_time_sec} seconds) ({num_data} samples)")