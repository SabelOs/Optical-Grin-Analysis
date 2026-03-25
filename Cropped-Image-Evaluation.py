#%%
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import laserbeamsize as lbs

# -----------------------------
# PARAMETERS
# -----------------------------
base_folder = "./"   # contains lens1, lens2, ...
num_lenses = 1

pixel_size = 5.2e-6      # meters
z_step = 1e-3            # 1 mm

# -----------------------------
# PROCESS EACH LENS
# -----------------------------
all_diameters = []
z_positions = None

for lens_idx in range(1, num_lenses+1):
    folder = os.path.join(base_folder, f"lens{lens_idx}")

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])

    diameters = []
    
    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)

        img = iio.imread(path).astype(float)
        # convert RGB → grayscale if needed
        if img.ndim == 3:
            img = img.mean(axis=2)   # simple and robust
        # normalize (important!)
        img = img / 255.0

        # subtract background
        img = img - np.min(img)

        try:
            cx, cy, d_major, d_minor, phi = lbs.beam_size(img)

            d = 0.5 * (d_major + d_minor)
            diameters.append(d * pixel_size)
            
            # plot every 5th image
            if i % 5 == 0:
                lbs.plot_image_analysis(img)
                plt.title(f"Lens {lens_idx}, image {fname}")
                plt.show()

        except Exception as e:
            print(f"Error in lens {lens_idx}, image {fname}: {e}")
            diameters.append(np.nan)

    diameters = np.array(diameters)
    all_diameters.append(diameters)

    # define z positions once
    if z_positions is None:
        z_positions = np.arange(len(diameters)) * z_step

# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure(figsize=(8, 6))

for i, d in enumerate(all_diameters):
    plt.plot(z_positions * 1e3, d * 1e6, label=f"Lens {i+1}")

plt.xlabel("z position (mm)")
plt.ylabel("Beam diameter (µm)")
plt.title("Beam size evolution (manual crops)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("beam_size_plot.png", dpi=300)
plt.show()
#%%