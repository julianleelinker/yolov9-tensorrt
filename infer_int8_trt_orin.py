import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import tritonclient.utils.shared_memory as shm
from tritonclient import utils
import random
from cv_bridge import CvBridge
import pathlib


def preprocess(img, imgsz=(640, 640)):
    h, w, _ = img.shape  # h:640, w:960

    scale = min(imgsz[0] / w, imgsz[1] / h)
    input_img = np.zeros((imgsz[1], imgsz[0], 3), dtype=np.float32)
    nh = int(scale * h)
    nw = int(scale * w)
    input_img[:nh, :nw, :] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh))
    input_img = input_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
    input_img = input_img.transpose(2, 0, 1)  # Change to (C, H, W)
    return input_img, scale, nh, nw, h, w


class ObjectDetection:
    def __init__(self, triton_url='localhost:8001', model_name='yolov9-c7-converted-qat-nms-int8'):
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.colors = [
            [43, 196, 66], [253, 157, 36], [76, 130, 247],
            [110, 18, 194], [184, 167, 139], [108, 123, 17],
            [54, 166, 201], [250, 174, 194], [207, 139, 81],
            [8, 115, 51], [52, 188, 255], [188, 233, 13],
            [213, 29, 143], [131, 247, 38], [222, 245, 94],
            [134, 148, 128]
        ]
        self.ID_TO_CLASS = {
            0: 'BICYCLE', 1: 'BUS', 2: 'CAR', 3: 'CLEAN_TRAFFIC_SIGN', 4: 'CONE',
            5: 'DIRTY_TRAFFIC_SIGN', 6: 'JERSEY_BARRIER', 7: 'MOTORCYCLE', 8: 'PEDESTRIAN',
            9: 'RIDER', 10: 'ROAD_CRACKS', 11: 'ROAD_PATCH', 12: 'ROAD_POTHOLES',
            13: 'TRAFFIC_LIGHT', 14: 'TRUCK', 15: 'UNCLEAR_ROAD_MARKING'
        }

    def run_inference(self, image_path):
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img_resized, scale, nh, nw, orig_h, orig_w = preprocess(img)

        # Prepare inputs and outputs for Triton Inference Server
        input_img = np.array([img_resized])
        inputs = []
        outputs = []

        # Define the input tensor
        inputs.append(grpcclient.InferInput('images', input_img.shape, "FP32"))
        inputs[-1].set_data_from_numpy(input_img.astype(np.float32))

        # Define output tensors
        OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        outputs.append(grpcclient.InferRequestedOutput('num_dets'))
        outputs.append(grpcclient.InferRequestedOutput('det_boxes'))
        outputs.append(grpcclient.InferRequestedOutput('det_scores'))
        outputs.append(grpcclient.InferRequestedOutput('det_classes'))

        # Perform inference
        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)

        # Extract the results
        num_dets = results.as_numpy('num_dets')
        boxes = results.as_numpy('det_boxes')
        scores = results.as_numpy('det_scores')
        labels = results.as_numpy('det_classes')

        # Process results
        bounding_boxes = []
        confidence_threshold = 0.5  # You can adjust the threshold here

        for i in range(num_dets[0][0]):
            if scores[0][i] < confidence_threshold:
                continue
            x1, y1, x2, y2 = boxes[0][i]

            x1 = int((x1 / nw) * orig_w)
            y1 = int((y1 / nh) * orig_h)
            x2 = int((x2 / nw) * orig_w)
            y2 = int((y2 / nh) * orig_h)

            label = self.ID_TO_CLASS[int(labels[0][i])]
            bounding_boxes.append({
                'class': label,
                'confidence': scores[0][i],
                'bbox': [x1, y1, x2, y2]
            })

            # Draw bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(labels[0][i])], 2)
            cv2.putText(img, f"{label} {scores[0][i]:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[int(labels[0][i])], 2)

        # Save the output image
        # output_image_path = "output.jpg"
        # cv2.imwrite(output_image_path, img)
        # print(f"Detection completed. Output saved to {output_image_path}")
        return bounding_boxes, img


# Example usage
if __name__ == "__main__":
    # image_root = ('/home/ubuntu/julian/tiip-s10-500/')
    image_root = ('/home/ubuntu/julian/tiip-images/')
    output_root = pathlib.Path('output')
    output_root.mkdir(exist_ok=True, parents=True)
    image_list = sorted(pathlib.Path(image_root).rglob('*.jpg'))
    print(len(image_list))
    print(image_list)
    for image_path in image_list:
        print(image_path)
        object_detection = ObjectDetection(triton_url='localhost:8001', model_name='yolov9-c7-converted-qat-nms-int8')
        detections, img = object_detection.run_inference(str(image_path))
        output_image_path = output_root / image_path.name
        cv2.imwrite(output_image_path, img)

