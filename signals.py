import signal
import sys


def setup_signal_handlers():
    """Set up signal handlers for the application"""

    def segfault_handler(signum, frame):
        print("Caught segmentation fault")
        # Restore default handler
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        sys.exit(1)

    # Register handlers
    try:
        signal.signal(signal.SIGSEGV, segfault_handler)
        print("Signal handlers registered successfully")
    except Exception as e:
        print(f"Failed to register signal handlers: {e}")
