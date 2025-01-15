## ###############################################################
## MODULES
## ###############################################################
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn
from skimage.exposure import equalize_adapthist as eq_hist 
from skimage.filters import sobel, gaussian
import cmasher as cmr
from numba import njit, prange


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
np.random.seed(420)

## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
        
def timeFunc(func):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_elapsed = time.time() - time_start
        print(f"{func.__name__}() took {time_elapsed:.3f} seconds to execute.")
        return result
    return wrapper


def generate_2d_gaussian_field(size : tuple,
                                correlation_length: float) -> np.ndarray:
    # Generate white noise in Fourier space
    noise = np.random.normal(0, 1, size)
    # Create a grid of frequencies
    kx = np.fft.fftfreq(size[0])
    ky = np.fft.fftfreq(size[1])
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    # Compute the magnitude of the wave vector
    k = np.sqrt(kx**2 + ky**2)
    # Create a Gaussian filter in Fourier space
    filter = np.exp(-0.5 * (k * correlation_length)**2)
    # Apply the filter to the noise in Fourier space
    field_ft = fftn(noise) * filter
    # Transform back to real space
    return np.real(ifftn(field_ft))


## ###############################################################
## LIC IMPLEMENTATION
## ###############################################################
@njit
def smoothing_kernel(streamline_length: int,
                     step_index: int) -> float:
  return 0.5 * (1 + np.cos(np.pi * step_index / streamline_length))

@njit
def advectStreamline(
    vfield: np.ndarray,
    sfield_in: np.ndarray,
    start_row: int,
    start_col: int, 
    dir_sgn: int,
    streamline_length: int,
    bool_periodic_BCs: bool = True) -> tuple:
  weighted_sum = 0.0
  total_weight = 0.0
  row_float, col_float = start_row, start_col
  num_rows, num_cols = vfield.shape[1], vfield.shape[2]
  for step in range(streamline_length):
    row_int = int(np.floor(row_float))
    col_int = int(np.floor(col_float))
    vel_col = dir_sgn * vfield[0, row_int, col_int] # vel-x
    vel_row = dir_sgn * vfield[1, row_int, col_int] # vel-y
    ## skip if the field magnitude is zero: advection has halted
    if abs(vel_row) == 0.0 and abs(vel_col) == 0.0: break
    ## compute how long the streamline advects before it leaves the current cell region (cell-centered boundaries)
    if   vel_row > 0.0: delta_time_row = (np.floor(row_float) + 1 - row_float) / vel_row
    elif vel_row < 0.0: delta_time_row = (np.ceil(row_float)  - 1 - row_float) / vel_row
    else:               delta_time_row = np.inf
    if   vel_col > 0.0: delta_time_col = (np.floor(col_float) + 1 - col_float) / vel_col
    elif vel_col < 0.0: delta_time_col = (np.ceil(col_float)  - 1 - col_float) / vel_col
    else:               delta_time_col = np.inf
    ## equivelant to a CFL condition
    time_step = min(delta_time_col, delta_time_row)
    ## advect the streamline to the next cell region
    col_float += vel_col * time_step
    row_float += vel_row * time_step
    if bool_periodic_BCs:
      row_float = (row_float + num_rows) % num_rows
      col_float = (col_float + num_cols) % num_cols
    else:
      row_float = max(0.0, min(row_float, num_rows - 1))
      col_float = max(0.0, min(col_float, num_cols - 1))
    ## weight the contribution of the current pixel based on its distance from the start of the streamline
    contribution_weight = smoothing_kernel(streamline_length, step)
    weighted_sum += contribution_weight * sfield_in[row_int, col_int]
    total_weight += contribution_weight
  return weighted_sum, total_weight


@njit(parallel=True)
def _computeLIC(
    vfield: np.ndarray,
    sfield_in: np.ndarray,
    sfield_out: np.ndarray, 
    streamline_length: int,
    num_rows: int,
    num_cols: int) -> np.ndarray:
  for row in prange(num_rows):
    for col in range(num_cols):
      forward_sum, forward_total = advectStreamline(vfield, sfield_in, row, col, +1, streamline_length)
      backward_sum, backward_total = advectStreamline(vfield, sfield_in, row, col, -1, streamline_length)
      total_sum = forward_sum + backward_sum
      total_weight = forward_total + backward_total
      sfield_out[row, col] = total_sum / total_weight if total_weight > 0.0 else 0.0
  return sfield_out

@timeFunc
def computeLIC(vfield, sfield_in: np.ndarray = None, streamline_length: int = None):
    num_comps, num_rows, num_cols = vfield.shape
    sfield_out = np.zeros((num_rows, num_cols), dtype=np.float32)
    if sfield_in is None: sfield_in = np.random.rand(num_rows, num_cols).astype(np.float32)
    if streamline_length is None: streamline_length = 10
    return _computeLIC(vfield, sfield_in, sfield_out, streamline_length, num_rows, num_cols)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  print("Started running demo scripts...")
  ## create figure canvas
  fig, ax = plt.subplots(figsize=(6,6))
  ## define domain
  print("Initialising parameters...")
  size = 1000
  kernel_size = 150
  ## define vector field
  vfield = np.array([generate_2d_gaussian_field([size,size], 500),
                     generate_2d_gaussian_field([size,size], 500)])
  ## create a randomly varying background
  sfield = np.random.rand(size, size)
  ## apply the LIC
  print("Computing LIC...")
  for _ in range(9): 
    sfield = computeLIC(vfield, sfield, kernel_size)
    sfield = eq_hist(sfield)
  sfield = sobel(gaussian(sfield,sigma=1))
  ## visualise the LIC
  print("Plotting data...")
  ax.imshow(sfield/sfield.max(), cmap=cmr.cosmic,
            origin="lower",
            extent = [-5, 5, -5, 5],
            vmax=0.8)
  ## tidy up the figure
  ax.set_xticks([])
  ax.set_yticks([])
  ## save and close the figure
  print("Saving figure...")
  fig_name = "/Users/beattijr/Desktop/LIC_test.png"
  fig.savefig(fig_name, dpi=200)
  plt.close(fig)
  print("Saved:", fig_name)
  return 1

if __name__ == "__main__":
  main()