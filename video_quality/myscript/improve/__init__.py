"""WorldArena video improvement toolkit.

Tier 1 (pure post-processing on already-generated videos):
- ``ops``        - low-level numpy/cv2 frame operations
- ``io_utils``   - mp4/frame-dir I/O helpers
- ``postprocess``- main CLI tying everything together

Tier 2 (inference-side helpers, do not retrain):
- ``seed_select``- pick best seed by aesthetic + image_quality + flow proxies
- ``vfi_interp`` - frame interpolation wrapper

Tier 3 (training / pipeline replacements):
- ``data_filter``       - filter training manifest by aesthetic + motion
- ``perceptual_losses`` - LPIPS / flow-consistency / VFI-recon losses
- ``tracker_cotracker`` - drop-in replacement for detection_tracking.py SAM3 stage
"""
