#%%
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import laserbeamsize as lbs

from scipy.optimize import curve_fit

def gaussian(x, A, x0, sigma, offset):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset


def plot_beam_profiles(img, xc, yc, d_major, d_minor, phi):
    """
    Plot intensity profiles along major and minor axes with Gaussian fits
    """

    # coordinate grid
    y, x = np.indices(img.shape)

    # rotate coordinates into beam frame
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    x_shift = x - xc
    y_shift = y - yc

    x_rot =  cos_p * x_shift + sin_p * y_shift   # major axis
    y_rot = -sin_p * x_shift + cos_p * y_shift   # minor axis

    # -----------------------------
    # MAJOR AXIS PROFILE
    # -----------------------------
    width = int(d_minor / 2)

    mask_major = np.abs(y_rot) < width
    x_vals = x_rot[mask_major]
    intens_major = img[mask_major]

    # sort for plotting
    idx = np.argsort(x_vals)
    x_vals = x_vals[idx]
    intens_major = intens_major[idx]

    # fit Gaussian
    try:
        p0 = [np.max(intens_major), 0, d_major/4, np.min(intens_major)]
        popt, _ = curve_fit(gaussian, x_vals, intens_major, p0=p0)

        x_fit = np.linspace(np.min(x_vals), np.max(x_vals), 300)
        y_fit = gaussian(x_fit, *popt)
    except:
        x_fit, y_fit = None, None

    # -----------------------------
    # MINOR AXIS PROFILE
    # -----------------------------
    width = int(d_major / 2)

    mask_minor = np.abs(x_rot) < width
    y_vals = y_rot[mask_minor]
    intens_minor = img[mask_minor]

    idx = np.argsort(y_vals)
    y_vals = y_vals[idx]
    intens_minor = intens_minor[idx]

    try:
        p0 = [np.max(intens_minor), 0, d_minor/4, np.min(intens_minor)]
        popt2, _ = curve_fit(gaussian, y_vals, intens_minor, p0=p0)

        y_fit_x = np.linspace(np.min(y_vals), np.max(y_vals), 300)
        y_fit_y = gaussian(y_fit_x, *popt2)
    except:
        y_fit_x, y_fit_y = None, None

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(10, 4))

    # Major axis
    plt.subplot(1, 2, 1)
    plt.scatter(x_vals, intens_major, s=5, label="Data")
    if x_fit is not None:
        plt.plot(x_fit, y_fit, 'r', label="Gaussian fit")
    plt.title("Major axis profile")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid()

    # Minor axis
    plt.subplot(1, 2, 2)
    plt.scatter(y_vals, intens_minor, s=5, label="Data")
    if y_fit_x is not None:
        plt.plot(y_fit_x, y_fit_y, 'r', label="Gaussian fit")
    plt.title("Minor axis profile")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# -----------------------------
# PARAMETERS
# -----------------------------
base_folder = "./Sample9-26-03-2026/green-He-Ne"   # contains lens1, lens2, ...
# base_folder = "./NoSample"
num_lenses = 1

pixel_size = 5.2e-6      # meters
z_step = 1e-3            # 1 mm

# -----------------------------
# PROCESS EACH LENS
# -----------------------------
all_diameters = []
z_positions = None

for lens_idx in range(1, num_lenses + 1):
    folder = os.path.join(base_folder, f"lens{lens_idx}")

    files = sorted(
        [f for f in os.listdir(folder)
         if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    diameters = []

    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        print("Now reading:", fname, "\n")

        img = iio.imread(path).astype(float)

        # convert RGB → grayscale if needed
        if img.ndim == 3:
            img = img.mean(axis=2)

        # -----------------------------
        # BACKGROUND REMOVAL
        # -----------------------------
        img = lbs.subtract_corner_background(img, nT = 3, iso_noise=False)
        #img = lbs.subtract_tilted_background(img)
        
        try:
            # -----------------------------
            # INITIAL BEAM ESTIMATE
            # -----------------------------
            xc, yc, d_major, d_minor, phi = lbs.basic_beam_size(img)

            # -----------------------------
            # ITERATIVE REFINEMENT
            # -----------------------------
            for _ in range(2):   # 2–3 iterations usually enough
                mask = lbs.rotated_rect_mask(
                    img,
                    xc, yc,
                    3 * d_major,
                    3 * d_minor,
                    -phi
                )

                masked_img = np.copy(img)
                masked_img[mask < 1] = 0

                xc, yc, d_major, d_minor, phi = lbs.basic_beam_size(masked_img)

            print("Angle of fitting:", phi)
            print("Major / Minor:", d_major, d_minor)

            # average diameter
            d = 0.5 * (d_major + d_minor)
            diameters.append(d * pixel_size)

            print("Found diameter:", d, "\n")

            # -----------------------------
            # DEBUG PLOT (every 2nd image)
            # -----------------------------
            if i % 10 == 0:
                plt.figure(figsize=(6, 5))
                plt.imshow(img)
                plt.colorbar()

                # ellipse (fit result)
                xp, yp = lbs.ellipse_arrays(xc, yc, d_major, d_minor, phi)
                plt.plot(xp, yp, ":y", label="Ellipse fit")

                # rectangle (integration region)
                xp, yp = lbs.rotated_rect_arrays(xc, yc, 3*d_major, 3*d_minor, phi)
                plt.plot(xp, yp, ":r", label="Mask region")

                plt.scatter(xc, yc, color="red", label="Center")

                plt.title(f"Lens {lens_idx}, {fname}")
                plt.legend()
                plt.tight_layout()
                plt.show()
                
                lbs.plot_image_analysis(img)
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
plt.title("Beam size evolution (iterative fit)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("beam_size_plot.png", dpi=300)
plt.show()
#%%