"""
ASR Training entry point wrapper.
"""
import os
import sys

SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from training.train_asr import main

if __name__ == "__main__":
    main()


