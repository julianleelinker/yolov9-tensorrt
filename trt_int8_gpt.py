import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

import sys, os
TRT_LOGGER = trt.Logger()

'''
class CalibrationDataset(object):
    def __init__(self, calibration_data_path):
        # Load or define your calibration dataset here
        # This should be representative of your input data distribution
        self.data = np.load(calibration_data_path)
        self.index = 0

    def get_batch(self, batch_size):
        if self.index + batch_size > len(self.data):
            self.index = 0  # Start over if we run out of data
        batch = self.data[self.index:self.index + batch_size]
        self.index += batch_size
        return batch  # Return a batch of data for calibration

class MyInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_dataset, input_shape, cache_file='calibration_cache.bin'):
        super(MyInt8Calibrator, self).__init__()
        self.dataset = calibration_dataset
        self.batch_size = input_shape[0]
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.current_batch = np.zeros(input_shape, dtype=np.float32)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names, p_str=None):
        batch = self.dataset.get_batch(self.batch_size)
        if not batch.size:  # Check if we've run out of data
            return None
        np.copyto(self.current_batch, batch)
        return [int(self.current_batch.ctypes.data)]
    
    def read_calibration_cache(self):
        # This function is called by TensorRT to load the calibration cache
        try:
            with open(self.cache_file, 'rb') as f:
                return f.read()
        except:
            return None
    
    def write_calibration_cache(self, cache):
        # This function is called by TensorRT to save the calibration cache
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
'''


def load_yolov7_coco_image(cocodir, topn = None):
    
    files = os.listdir(cocodir)
    files = [file for file in files if file.endswith(".jpg")]

    if topn is not None:
        np.random.seed(31)
        np.random.shuffle(files)
        files = files[:topn]

    datas = []
    imgsz = (640, 640)

    # dataloader is setup pad=0.5
    for i, file in enumerate(files):
        if i == 0: continue
        if (i + 1) % 200 == 0:
            print(f"Load {i + 1} / {len(files)} ...")

        '''
        img = cv2.imread(os.path.join(cocodir, file))
        from_ = img.shape[1], img.shape[0]
        to_   = 640, 640
        scale = min(to_[0] / from_[0], to_[1] / from_[1])

        # low accuracy
        # M = np.array([
        #     [scale, 0, 16],
        #     [0, scale, 16],  # same to pytorch
        # ])

        # more accuracy
        M = np.array([
            [scale, 0, -scale * from_[0]  * 0.5  + to_[0] * 0.5 + scale * 0.5 - 0.5 + 16],
            [0, scale, -scale * from_[1] * 0.5 + to_[1] * 0.5 + scale * 0.5 - 0.5 + 16],  # same to pytorch
        ])
        input = cv2.warpAffine(img, M, (672, 672), borderValue=(114, 114, 114))
        input = input[..., ::-1].transpose(2, 0, 1)[None]   # BGR->RGB, HWC->CHW, CHW->1CHW
        input = (input / 255.0).astype(np.float32)
        datas.append(input)
        '''
        img = cv2.imread(os.path.join(cocodir, file))
        h, w, _ = img.shape
        scale = min(imgsz[0]/w, imgsz[1]/h)
        inp = np.zeros((imgsz[1], imgsz[0], 3), dtype = np.float32)
        nh = int(scale * h)
        nw = int(scale * w)
        inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh))
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
        print(np.mean(inp))
        datas.append(inp)
        
    return np.concatenate(datas, axis=0)
    

# class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
class MNISTEntropyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, training_data, cache_file, batch_size=64):

        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        # trt.IInt8EntropyCalibrator2.__init__(self)
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        if not os.path.exists(cache_file):

            # Allocate enough memory for a whole batch.
            self.data = load_yolov7_coco_image(training_data, 1000)
            print(self.data.shape)
            print(self.data[0].nbytes * self.batch_size)
            self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
            print('DONE mem')

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index : self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_and_save_engine_int8(onnx_file_path, engine_file_path, calibrator, device=0):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        
        # Define optimization profiles
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 640, 640), (3, 3, 640, 640), (4, 3, 640, 640))
        config.add_optimization_profile(profile)
        
        # Specify the calibration dataset and create a calibrator
        config.int8_calibrator = calibrator
        
        # Load ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
        
        # Build the engine
        engine = builder.build_engine(network, config)
        
        # Save the engine to a file
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())

def main():
    onnx_file_path = 'fp32-nms.onnx'
    input_shape = (1, 3, 640, 640)  # Adjust based on your calibration dataset

    calibration_cache = 'int8-nms.cache'
    engine_file_path  = 'int8-nms.trt'
    calibrator = MNISTEntropyCalibrator("samples/", cache_file=calibration_cache)

    # calibration_dataset_path = 'path_to_your_calibration_dataset.npy'  # Update this path
    # calibrator = MyInt8Calibrator(CalibrationDataset(calibration_dataset_path), input_shape)

    build_and_save_engine_int8(onnx_file_path, engine_file_path, calibrator)


if __name__ == "__main__":
    main()