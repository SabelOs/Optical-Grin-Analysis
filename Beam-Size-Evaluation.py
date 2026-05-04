import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os

from skimage.feature import blob_log
from scipy.spatial.distance import cdist
import laserbeamsize as lbs

# -----------------------------
# PARAMETERS (adjust if needed)
# -----------------------------
folder = "./OptischeMessungSample9"  # image folder
prefix = "sample9_1"
num_images = 2

pixel_size = 5.2e-6  # meters
z_step = 1e-3        # 1 mm

crop_radius = 120    # pixels (adjust!)
num_lenses = 6

# -----------------------------
# LOAD IMAGES
# -----------------------------
images = []
for i in range(num_images):
    fname = os.path.join(folder, f"{prefix}{i}.png")
    img = iio.imread(fname).astype(float)
    img = img / 255.0
    images.append(img.astype(float))

# -----------------------------
# DETECT BLOBS (FIRST IMAGE)
# -----------------------------
first = images[0]

blobs = blob_log(first, min_sigma=10, max_sigma=50, num_sigma=10, threshold=0.05)

# blobs: (y, x, sigma)
centers = blobs[:, :2]

# take 6 brightest blobs
intensities = [first[int(y), int(x)] for y, x in centers]
idx = np.argsort(intensities)[-num_lenses:]

centers = centers[idx]

# sort left to right (optional but nice)
centers = centers[np.argsort(centers[:, 1])]

tracked_centers = [centers]

plt.imshow(first, cmap='gray')
plt.scatter(centers[:,1], centers[:,0], c='r')
plt.savefig("trackingtest.png")
plt.show()
# -----------------------------
# TRACK CENTERS OVER IMAGES
# -----------------------------
for i in range(1, num_images):
    img = images[i]

    blobs = blob_log(img, min_sigma=10, max_sigma=50, num_sigma=10, threshold=0.05)

    if len(blobs) == 0:
        print(f"Warning: No blobs detected in image {i}")
        tracked_centers.append(tracked_centers[-1])
        continue

    new_centers = blobs[:, :2]

    prev = tracked_centers[-1]

    # enforce correct shape
    prev = np.atleast_2d(prev)
    new_centers = np.atleast_2d(new_centers)

    dist = cdist(prev, new_centers)

    assignment = np.argmin(dist, axis=1)

    matched = new_centers[assignment]

    tracked_centers.append(matched)

# -----------------------------
# BEAM SIZE CALCULATION
# -----------------------------
z_positions = np.arange(num_images) * z_step

diameters = np.zeros((num_images, num_lenses))

for i in range(num_images):
    img = images[i]

    for j in range(num_lenses):
        y, x = tracked_centers[i, j]

        y = int(y)
        x = int(x)

        # crop
        sub = img[
            y - crop_radius:y + crop_radius,
            x - crop_radius:x + crop_radius
        ]
        
        plt.imshow(sub, cmap='gray')
        plt.colorbar()
        plt.show()

        # skip if crop invalid
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            diameters[i, j] = np.nan
            continue

        # subtract background
        sub = sub - np.min(sub)

        try:
            cx, cy, d_major, d_minor, phi = lbs.beam_size(sub)

            # average diameter (or use major axis)
            d = 0.5 * (d_major + d_minor)

            diameters[i, j] = d * pixel_size

        except Exception:
            diameters[i, j] = np.nan


# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure(figsize=(8, 6))

for j in range(num_lenses):
    plt.plot(z_positions * 1e3, diameters[:, j] * 1e6, label=f"Lens {j+1}")

plt.xlabel("z position (mm)")
plt.ylabel("Beam diameter (µm)")
plt.title("Beam size evolution")
plt.legend()
plt.grid()

plt.show()