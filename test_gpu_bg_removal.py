"""
Test script for GPU-accelerated background removal.
"""
from pathlib import Path
from profile_photo import create_headshot
import time
import onnxruntime as ort

test_img = Path('profile_photo/input/drive-download-20260102T074514Z-1-001/20251127-IMG_7326_henrik.jpg')

print('='*60)
print('GPU Acceleration Test for Background Removal')
print('='*60)
print(f'Available providers: {ort.get_available_providers()}')
print('')

# Test with CPU explicitly
print('Test 1: CPU-only processing...')
start = time.time()
photo_cpu = create_headshot(test_img, remove_bg=True, bg_providers=['CPUExecutionProvider'])
cpu_time = time.time() - start
print(f'  CPU time: {cpu_time:.2f} seconds')
print(f'  Output: {photo_cpu.image.mode}, size: {photo_cpu.image.size}')
print('')

# Test with auto-detection (will try GPU, fallback to CPU)
print('Test 2: Auto GPU detection (with fallback)...')
start = time.time()
photo_auto = create_headshot(test_img, remove_bg=True)
auto_time = time.time() - start
print(f'  Auto time: {auto_time:.2f} seconds')
print(f'  Output: {photo_auto.image.mode}, size: {photo_auto.image.size}')
print('')

print('='*60)
print('Summary:')
print(f'  CPU time: {cpu_time:.2f}s')
print(f'  Auto time: {auto_time:.2f}s')
if auto_time < cpu_time:
    speedup = cpu_time / auto_time
    print(f'  Speedup: {speedup:.2f}x faster with auto-detection')
else:
    print('  Note: Both using CPU (GPU dependencies missing)')
print('='*60)

