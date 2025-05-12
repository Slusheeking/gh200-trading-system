import numpy as np
import argparse
import json
import time
import os
from flask import Flask, request, jsonify

# Import LightGBM for direct inference
import lightgbm as lgb

# Import TensorRT for direct inference
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    print("TensorRT or PyCUDA not available, falling back to LightGBM only")

app = Flask(__name__)
model = None
model_type = None
input_shape = None
output_shape = None
trt_context = None
trt_engine = None
trt_runtime = None
trt_bindings = None
trt_stream = None

def initialize_model(model_path, model_type="lightgbm", use_tensorrt=True, optimization_level=99):
    """Initialize the model with appropriate settings"""
    global model, input_shape, output_shape, trt_context, trt_engine, trt_runtime, trt_bindings, trt_stream
    
    print(f"Loading {model_type} model from {model_path}")
    start_time = time.time()
    
    if model_type == "lightgbm":
        # Load LightGBM model directly
        model = lgb.Booster(model_file=model_path)
        
        # Get input and output shapes
        num_features = model.num_feature()
        input_shape = (1, num_features)  # Batch size, feature size
        
        # Determine output shape based on model objective
        params = model.params
        if 'objective' in params and 'multiclass' in params['objective']:
            num_classes = int(params.get('num_class', 1))
        else:
            num_classes = 1
        output_shape = (1, num_classes)  # Batch size, output size
        
    elif model_type == "tensorrt" and HAS_TENSORRT and use_tensorrt:
        # Load TensorRT engine directly
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        
        with open(model_path, "rb") as f:
            engine_data = f.read()
            trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
        
        if not trt_engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {model_path}")
        
        trt_context = trt_engine.create_execution_context()
        
        # Create CUDA stream
        trt_stream = cuda.Stream()
        
        # Get input and output shapes
        input_shape = []
        output_shape = []
        
        # Allocate device buffers and create bindings
        trt_bindings = []
        for binding_idx in range(trt_engine.num_bindings):
            shape = trt_engine.get_binding_shape(binding_idx)
            size = trt.volume(shape) * trt_engine.max_batch_size
            dtype = trt.nptype(trt_engine.get_binding_dtype(binding_idx))
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            trt_bindings.append(device_mem)
            
            if trt_engine.binding_is_input(binding_idx):
                input_shape = shape
            else:
                output_shape = shape
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for model inference"""
    global model, model_type, input_shape, output_shape, trt_context, trt_engine, trt_bindings, trt_stream
    
    if model is None and trt_engine is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Parse JSON input
    if request.is_json:
        data = request.json
    else:
        try:
            data = json.loads(request.data)
        except:
            return jsonify({'error': 'Invalid input format'}), 400
    
    # Prepare input data
    try:
        if 'features' not in data:
            return jsonify({'error': 'Missing input features'}), 400
        
        features = np.array(data['features'], dtype=np.float32)
        
        # Run inference
        start_time = time.time()
        
        if model_type == "lightgbm":
            # LightGBM inference
            results = model.predict([features], raw_score=True)
            outputs = results.tolist()
        
        elif model_type == "tensorrt":
            # TensorRT inference
            # Prepare input data
            input_data = features.astype(np.float32)
            
            # Allocate host memory for input and output
            host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            
            # Copy input data to host buffer
            np.copyto(host_input, input_data.ravel())
            
            # Transfer input data to device
            cuda.memcpy_htod_async(trt_bindings[0], host_input, trt_stream)
            
            # Run inference
            trt_context.execute_async_v2(bindings=trt_bindings, stream_handle=trt_stream.handle)
            
            # Transfer output data to host
            cuda.memcpy_dtoh_async(host_output, trt_bindings[1], trt_stream)
            
            # Synchronize stream
            trt_stream.synchronize()
            
            # Reshape output
            outputs = host_output.reshape(output_shape).tolist()
        
        end_time = time.time()
        
        # Format results
        response = {
            'outputs': outputs,
            'timing': {
                'inference_time': (end_time - start_time) * 1000,  # ms
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Inference Server")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--model-type", choices=["lightgbm", "tensorrt"], default="lightgbm",
                       help="Type of model (lightgbm or tensorrt)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensorrt", action="store_true", help="Use TensorRT acceleration")
    parser.add_argument("--optimize", type=int, default=99,
                       help="Optimization level (0-99, default: 99)")
    args = parser.parse_args()
    
    # Initialize the model
    initialize_model(args.model, args.model_type, args.tensorrt, args.optimize)
    
    # Start the server
    app.run(host='0.0.0.0', port=args.port)
