"""
Utility functions for spectral analysis including FFT caching and helpers.
"""
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
    fftn = pyfftw.interfaces.numpy_fft.fftn
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    rfftn = pyfftw.interfaces.numpy_fft.rfftn
    irfftn = pyfftw.interfaces.numpy_fft.irfftn
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
else:
    # Use numpy's FFT functions
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    rfftn = np.fft.rfftn
    irfftn = np.fft.irfftn
    fftfreq = np.fft.fftfreq
    fftshift = np.fft.fftshift
    
class FFTWPlanCache:
    """Cache for FFTW plans to avoid recreation overhead."""
    
    def __init__(self, 
                 max_plans=10):
        self.plans = {}
        self.max_plans = max_plans
        self.enabled = pyfftw_import
        
        
    def get_fft_plan(self, 
                     shape, 
                     axes, 
                     forward=True, 
                     real=False):
        """Get or create an FFTW plan."""
        
        if not self.enabled:
            return None
            
        key = (shape, axes, forward, real)
        
        if key in self.plans:
            return self.plans[key]
        
        # Create new plan
        if len(self.plans) >= self.max_plans:
            # Remove oldest plan
            oldest_key = next(iter(self.plans))
            del self.plans[oldest_key]
        
        # Create aligned arrays for plan
        if real:
            if forward:
                input_array = pyfftw.empty_aligned(shape, dtype='float64')
                output_shape = list(shape)
                output_shape[axes[-1]] = shape[axes[-1]] // 2 + 1
                output_array = pyfftw.empty_aligned(output_shape, dtype='complex128')
                plan = pyfftw.FFTW(input_array, output_array, axes=axes,
                                  flags=['FFTW_MEASURE'], threads=threads)
            else:
                input_shape = list(shape)
                input_shape[axes[-1]] = shape[axes[-1]] // 2 + 1
                input_array = pyfftw.empty_aligned(input_shape, dtype='complex128')
                output_array = pyfftw.empty_aligned(shape, dtype='float64')
                plan = pyfftw.FFTW(input_array, output_array, axes=axes,
                                  direction='FFTW_BACKWARD',
                                  flags=['FFTW_MEASURE'], threads=threads)
        else:
            input_array = pyfftw.empty_aligned(shape, dtype='complex128')
            output_array = pyfftw.empty_aligned(shape, dtype='complex128')
            direction = 'FFTW_FORWARD' if forward else 'FFTW_BACKWARD'
            plan = pyfftw.FFTW(input_array, output_array, axes=axes,
                              direction=direction,
                              flags=['FFTW_MEASURE'], threads=threads)
        
        self.plans[key] = plan
        return plan
  
    
    def execute_fft(self, 
                    data, 
                    axes, 
                    forward=True, 
                    real=False, 
                    norm='forward'):
        
        """Execute FFT using cached plan if available."""
        if not self.enabled:
            # Fallback to numpy
            if real:
                if forward:
                    return rfftn(data, axes=axes, norm=norm)
                else:
                    return irfftn(data, axes=axes, norm=norm)
            else:
                if forward:
                    return fftn(data, axes=axes, norm=norm)
                else:
                    return ifftn(data, axes=axes, norm=norm)
        
        plan = self.get_fft_plan(data.shape, axes, forward, real)
        
        if plan is None:
            # Fallback if plan creation failed
            if real:
                if forward:
                    return rfftn(data, axes=axes, norm=norm)
                else:
                    return irfftn(data, axes=axes, norm=norm)
            else:
                if forward:
                    return fftn(data, axes=axes, norm=norm)
                else:
                    return ifftn(data, axes=axes, norm=norm)
        
        # Copy data to aligned array
        plan.input_array[:] = data
        
        # Execute
        result = plan()
        
        # Apply normalization
        if norm == 'forward' and forward:
            n = np.prod([data.shape[i] for i in axes])
            result = result / n
        elif norm == 'forward' and not forward:
            pass  # No normalization for inverse
        
        return result.copy()

def ensure_float32(field, field_name="field"):
    """Utility to convert arrays to float32 with consistent messaging."""
    if field.dtype != np.float32:
        print(f"Converting {field.dtype} {field_name} to float32 for memory efficiency")
        return field.astype(np.float32)
    return field

def validate_field_shape(field, expected_dims, field_name="field"):
    """Validate field has expected dimensions."""
    if len(field.shape) != expected_dims:
        raise ValueError(f"{field_name} should have {expected_dims} dimensions")