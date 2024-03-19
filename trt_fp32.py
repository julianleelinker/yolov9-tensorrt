import tensorrt as trt

def build_and_save_engine(onnx_file_path, engine_file_path, device=0):
    # Set the device for execution if necessary. This part is more complex in TensorRT as it typically selects the best device automatically.
    # You might need to enforce device selection via CUDA APIs if necessary.
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    
    # Create builder and network
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_batch_size = 4  # Set the max batch size
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        
        # Define optimization profiles
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 640, 640), (3, 3, 640, 640), (4, 3, 640, 640))
        config.add_optimization_profile(profile)
        
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

onnx_file_path = 'fp32-nms.onnx'
engine_file_path = 'fp32-nms.trt'
build_and_save_engine(onnx_file_path, engine_file_path)