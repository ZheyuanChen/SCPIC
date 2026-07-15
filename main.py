import os
import numpy as np
from scpic.mirrors import ParabolicMirror2D
from scpic.fields import IncidentFieldTM
from scpic.solvers import evaluate_SC_2D
from scpic.export import export_all_fields

def main():
    # ---------------------------------------------------------
    # 1. Define Physics Parameters
    # ---------------------------------------------------------
    wavelength = 1e-6          # 1 micron
    k = 2 * np.pi / wavelength
    c = 299792458.0
    
    # Mirror parameters
    f0 = 10e-6                 # 10 microns focal length
    D = 20e-6                  # 20 microns mirror diameter
    
    # Beam parameters
    w0 = 8e-6                  # Beam waist
    E0 = 1.0                   # Peak electric field (V/m)

    # ---------------------------------------------------------
    # 2. Instantiate Objects
    # ---------------------------------------------------------
    print("Setting up mirror and field...")
    mirror = ParabolicMirror2D(f0=f0, D=D, mirror_type='OAP90')
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=2000)
    
    field = IncidentFieldTM(w0=w0, wavelength=wavelength, E0=E0)
    
    # Calculate incident fields ON the mirror surface
    By_inc = field.B_y(x_m, z_m, x_center=x_center)
    dBy_dn_inc = field.dBy_dn(x_m, z_m, nx, nz, x_center=x_center)

    # ---------------------------------------------------------
    # 3. Define the Observation Grid (EPOCH Injection Boundary)
    # ---------------------------------------------------------
    print("Defining observation grid near focus...")
    # Creating a 2D grid around the focal point (x=0, z=0)
    Nx, Nz = 150, 150
    x_vec = np.linspace(-5e-6, 5e-6, Nx)
    z_vec = np.linspace(-5e-6, 5e-6, Nz)
    
    dx = x_vec[1] - x_vec[0]
    dz = z_vec[1] - z_vec[0]
    
    # indexing='ij' ensures shape is (Nx, Nz)
    X_obs, Z_obs = np.meshgrid(x_vec, z_vec, indexing='ij') 
    
    # Flatten arrays for the 1D solver loop
    x_obs_flat = X_obs.flatten()
    z_obs_flat = Z_obs.flatten()

    # ---------------------------------------------------------
    # 4. Run the Stratton-Chu Solver
    # ---------------------------------------------------------
    print(f"Evaluating Stratton-Chu integral for {len(x_obs_flat)} points...")
    By_flat = evaluate_SC_2D(
        x_obs_flat, z_obs_flat, 
        x_m, z_m, nx, nz, dl, 
        By_inc, dBy_dn_inc, k
    )
    
    # Reshape back to 2D grid
    By_2d = By_flat.reshape((Nx, Nz))

    # ---------------------------------------------------------
    # 5. Calculate Ex and Ez via Finite Differences
    # ---------------------------------------------------------
    print("Computing Ex and Ez from By...")
    # np.gradient returns derivatives along axis 0 (x) and axis 1 (z)
    dBy_dx, dBy_dz = np.gradient(By_2d, dx, dz)
    
    # Maxwell's Equations in 2D TM Vacuum:
    Ex_2d = -(1j * c / k) * dBy_dz
    Ez_2d =  (1j * c / k) * dBy_dx

    # ---------------------------------------------------------
    # 6. Export to Raw Binary
    # ---------------------------------------------------------
    output_dir = "epoch_injection_data"
    print(f"Exporting binary fields to ./{output_dir} ...")
    
    # We export as float32 to fit typical Fortran raw binary readers.
    export_all_fields(output_dir, Ex_2d, Ez_2d, By_2d, dtype=np.float32)
    
    print("Done! Setup complete.")

if __name__ == "__main__":
    main()