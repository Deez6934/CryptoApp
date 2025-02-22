import os
import tensorflow as tf
import logging


class TensorFlowConfig:
    """Manages TensorFlow configuration and initialization"""

    @staticmethod
    def configure():
        # Suppress TensorFlow logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Disable oneDNN custom operations warning
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        # Configure device
        if len(tf.config.list_physical_devices("GPU")) > 0:
            TensorFlowConfig._configure_gpu()
        else:
            TensorFlowConfig._configure_cpu()

        TensorFlowConfig._print_device_info()
        TensorFlowConfig._test_computation()

    @staticmethod
    def _configure_gpu():
        """Configure GPU settings if available"""
        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU configuration successful")
        except:
            print("Error configuring GPU")

    @staticmethod
    def _configure_cpu():
        """Configure CPU settings"""
        try:
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            print("CPU threading configured for parallel operations")
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except:
            print("Error configuring CPU threading")

    @staticmethod
    def _print_device_info():
        """Print device information"""
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Num CPUs Available: {len(tf.config.list_physical_devices('CPU'))}")
        print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
        print(f"GPU Device Name: {tf.test.gpu_device_name()}")
        print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")

    @staticmethod
    def _test_computation():
        """Test basic computation on configured device"""
        try:
            with tf.device(
                "/CPU:0" if not tf.config.list_physical_devices("GPU") else "/GPU:0"
            ):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                device = "CPU" if not tf.config.list_physical_devices("GPU") else "GPU"
                print(f"Matrix multiplication test successful on {device}:", c)
        except Exception as e:
            print(f"Error running computation test: {e}")
