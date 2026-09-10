"""Fall-detection subsystem for DementiaCare.

Standalone package (sibling of ``backend/`` and ``frontend/``). Runs on the machine
that owns the webcam and communicates with the backend only over the HTTP ``/events``
contract, with a direct-MongoDB fallback. See ``fall_detection/README.md``.
"""

__version__ = "0.1.0"
