import tensorrt as trt
import numpy as np

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

def build_and_save_engine_int8(onnx_file_path, engine_file_path, calibration_dataset, device=0):
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
        input_shape = (1, 3, 640, 640)  # Adjust based on your calibration dataset
        calibrator = MyInt8Calibrator(CalibrationDataset(calibration_dataset), input_shape)
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

# Example usage
onnx_file_path = 'fp32-nms.onnx'
engine_file_path = 'int8-nms.trt'
calibration_dataset_path = 'path_to_your_calibration_dataset.npy'  # Update this path
build_and_save_engine_int8(onnx_file_path, engine_file_path, calibration_dataset_path)
