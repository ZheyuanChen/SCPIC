import numpy as np
import os

def export_field_binary(filepath_base, field_array, dtype=np.float64):
    """
    Exports a complex field array into two raw binary files:
    1. A file containing the amplitude: <filepath_base>_amp.bin
    2. A file containing the phase: <filepath_base>_phase.bin
    
    Parameters:
    - filepath_base: Base path/name of the file (e.g., 'workspace/Ex')
    - field_array: Complex-valued NumPy array of the field
    - dtype: Precision to write (np.float64 for double, np.float32 for single)
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(filepath_base), exist_ok=True)
    
    # Extract amplitude and phase
    amplitude = np.abs(field_array).astype(dtype)
    phase = np.angle(field_array).astype(dtype)
    
    # Define filenames
    amp_filename = f"{filepath_base}_amp.bin"
    phase_filename = f"{filepath_base}_phase.bin"
    
    # Write raw binary files
    # Note: NumPy's tofile() writes raw C-contiguous memory.
    # If your Fortran compiler expects big-endian, use field_array.byteswap() first.
    amplitude.tofile(amp_filename)
    phase.tofile(phase_filename)
    
    print(f"Exported: {amp_filename}")
    print(f"Exported: {phase_filename}")

def export_all_fields(directory, Ex, Ez, By, dtype=np.float64):
    """
    Convenience function to dump all 2D TM fields to the target directory.
    """
    export_field_binary(os.path.join(directory, "Ex"), Ex, dtype)
    export_field_binary(os.path.join(directory, "Ez"), Ez, dtype)
    export_field_binary(os.path.join(directory, "By"), By, dtype)