import tensorflow as tf

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled on {device}")
except:
    print("Invalid device or cannot modify virtual devices once initialized")

print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices("GPU"))
print("GPU Device: ", tf.test.gpu_device_name())
print("Is built with CUDA:", tf.test.is_built_with_cuda())

# Run a test computation
if len(physical_devices) > 0:
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            print("GPU Test Result:", tf.matmul(a, b))
    except:
        print("Error running GPU test")
else:
    print("No GPU devices found")
