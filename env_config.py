import os

# Optimize TensorFlow CPU performance
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable optimized CPU operations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging verbosity
os.environ["TF_NUM_INTEROP_THREADS"] = "4"  # Number of inter-op parallelism threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"  # Number of intra-op parallelism threads

# Optional: Enable memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
