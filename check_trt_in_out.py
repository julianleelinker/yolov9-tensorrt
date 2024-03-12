import tensorrt as trt


class TRTloader:

    def __init__(self, engine_path) -> None:
        self.logger = trt.Logger(trt.Logger.VERBOSE) 
        self.engine = self.load_engine(engine_path)
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def inspect(self, ):
        import ipdb; ipdb.set_trace()

engine_path = 'yolov9-c-nms.trt'
loader = TRTloader(engine_path)
loader.inspect()