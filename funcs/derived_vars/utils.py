import numpy as np
import multiprocessing

pyfftw_import = False
try: 
    import pyfftw
    pyfftw_import = True
    pyfftw.interfaces.cache.enable()
    threads = multiprocessing.cpu_count()
except ImportError:
    print("pyfftw not installed, using scipy's serial fft")

if pyfftw_import:
    # Use pyfftw for faster FFTs if available
    pyfftw.config.NUM_THREADS = threads
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
else:
    # Use numpy's FFT functions
    fftfreq = np.fft.fftfreq
    
    
def ensure_float32(field, field_name="field"):
    """Utility to convert arrays to float32 with consistent messaging."""
    if field.dtype != np.float32:
        print(f"Converting {field.dtype} {field_name} to float32 for memory efficiency")
        return field.astype(np.float32)
    return field