#%%
import numpy as np
import os
import pandas as pd
import imageio.v3 as iio
import laserbeamsize as lbs

# -----------------------------
# PARAMETERS
# -----------------------------
root = "./10-04-2026-Sample9-Camera-Brewster/"
light_sources = ["green-He-Ne", "red-He-Ne"]

pixel_size = 5.2      # µm
distance_offset = 38   # mm
z_step = 1            # mm

beam_divergence_angle_fwhm = 0
crop_padding = 20

# -----------------------------
# STORAGE
# -----------------------------
results = []
fit_results = []

# -----------------------------
# MAIN LOOP OVER LIGHT SOURCES
# -----------------------------
for source in light_sources:
    base_folder = os.path.join(root, source)

    folders = [
        f for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f))
    ]
    folders.sort()

    for folder_name in folders:
        folder_path = os.path.join(base_folder, folder_name)

        print(f"[{source}] Processing: {folder_name}")

        files = sorted(
            [f for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".tif", ".bmp"))],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        diameters = []

        for fname in files:
            path = os.path.join(folder_path, fname)
            img = iio.imread(path).astype(float)

            if img.ndim == 3:
                img = img.mean(axis=2)

            img_bg = lbs.subtract_corner_background(img, nT=3)

            peak = np.max(img_bg)
            half_max = peak / 2.0

            x_mask = ~np.all(img_bg < half_max, axis=1)
            y_mask = ~np.all(img_bg < half_max, axis=0)

            x_idx = np.where(x_mask)[0]
            y_idx = np.where(y_mask)[0]

            if len(x_idx) == 0 or len(y_idx) == 0:
                diameters.append(np.nan)
                continue

            x_min = max(x_idx[0] - crop_padding, 0)
            x_max = min(x_idx[-1] + crop_padding, img_bg.shape[0])
            y_min = max(y_idx[0] - crop_padding, 0)
            y_max = min(y_idx[-1] + crop_padding, img_bg.shape[1])

            img_crop = img_bg[x_min:x_max, y_min:y_max]

            diameter = (img_crop.shape[1] - 2 * crop_padding) * pixel_size
            diameters.append(diameter)

        diameters = np.array(diameters)

        # position
        z = np.arange(1, len(diameters) + 1) * z_step + distance_offset
        z_rel = np.arange(1, len(diameters) + 1)

        diameters_corr = diameters - beam_divergence_angle_fwhm * z_rel

        valid = ~np.isnan(diameters_corr)

        if np.sum(valid) < 2:
            continue

        slope, intercept = np.polyfit(z[valid], diameters_corr[valid], 1)
        focal_point = -intercept / slope

        # -----------------------------
        # STORE FIT
        # -----------------------------
        fit_results.append({
            "light_source": source,
            "measurement": folder_name,
            "slope": slope,
            "intercept": intercept,
            "focal_point": focal_point
        })

        # -----------------------------
        # STORE DATA
        # -----------------------------
        for zi, zi_rel, di in zip(z, z_rel, diameters_corr):
            results.append({
                "light_source": source,
                "measurement": folder_name,
                "z": zi,
                "z_rel": zi_rel,
                "diameter": di
            })

# -----------------------------
# DATAFRAMES
# -----------------------------
df = pd.DataFrame(results)
df_fit = pd.DataFrame(fit_results)

# -----------------------------
# SAVE
# -----------------------------
#df.to_parquet("beam_data.parquet")
#df_fit.to_parquet("fit_results.parquet")

# optional csv
df.to_csv("beam_data.csv", index=False)
df_fit.to_csv("fit_results.csv", index=False)



# %% START PLOTTING RESULTS
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("beam_data.csv")
df_fit = pd.read_csv("fit_results.csv")

plt.figure()

for source, color in zip(["green-He-Ne", "red-He-Ne"], ["green", "red"]):
    sub = df[(df["measurement"].str.contains("noSample")) &
             (df["light_source"] == source)]

    z_rel = sub["z"] - sub["z"].min()

    plt.plot(z_rel, sub["diameter"], color=color, label=source)

    # divergence angle
    fit_row = df_fit[(df_fit["measurement"].str.contains("noSample")) &
                     (df_fit["light_source"] == source)].iloc[0]

    phi = np.arctan(fit_row["slope"])
    print(f"{source} divergence angle: {np.degrees(phi):.4f}°")

plt.xlabel("Relative Distance (mm)")
plt.ylabel("FWHM (µm)")
plt.legend()
plt.title("Laser Divergence (no sample)")
plt.show()


# %%
plt.figure()

greens = plt.cm.Greens(np.linspace(0.3, 1, 10))
reds = plt.cm.Reds(np.linspace(0.3, 1, 10))

for source, cmap in zip(["green-He-Ne", "red-He-Ne"], [greens, reds]):
    sub_df = df[df["light_source"] == source]
    measurements = [m for m in sub_df["measurement"].unique() if "lens" in m]

    for i, m in enumerate(measurements):
        sub = sub_df[sub_df["measurement"] == m]

        plt.plot(sub["z"], sub["diameter"],
                 color=cmap[i % len(cmap)],
                 label=f"{m} ({source})")

plt.xlabel("Distance from Lens (mm)")
plt.ylabel("FWHM (µm)")
plt.legend(fontsize=8)
plt.title("All Lens Measurements")
plt.show()


# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax = plt.subplots(figsize=(8,6))

offset = 1.0  # adjustable spacing

lens_names = sorted([m for m in df["measurement"].unique() if "lens" in m])

greens = plt.cm.Greens(np.linspace(0.3, 1, len(lens_names)))
reds = plt.cm.Reds(np.linspace(0.3, 1, len(lens_names)))

# main plot
for i, lens in enumerate(lens_names):
    for source, cmap in zip(["green-He-Ne", "red-He-Ne"], [greens, reds]):
        sub = df[(df["measurement"] == lens) &
                 (df["light_source"] == source)]

        ax.plot(sub["z"], sub["diameter"],
                color=cmap[i],
                label=f"{lens} ({source})")

# inset
axins = inset_axes(ax, width="40%", height="30%", loc="upper left")

x_positions = np.arange(len(lens_names)) * offset

for i, lens in enumerate(lens_names):
    for source, color in zip(["green-He-Ne", "red-He-Ne"], ["green", "red"]):
        row = df_fit[(df_fit["measurement"] == lens) &
                     (df_fit["light_source"] == source)]

        if len(row) == 0:
            continue

        axins.bar(x_positions[i], row["focal_point"].values[0],
                  color=color)

axins.set_xticks(x_positions)
axins.set_xticklabels(lens_names, rotation=45)
axins.set_title("Focal Points")

ax.set_xlabel("Distance (mm)")
ax.set_ylabel("FWHM (µm)")
ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.show()
# %%
# %%
plt.figure()

# -----------------------------
# FILTER LIST (EDIT THIS)
# -----------------------------
selected_lenses = [
    "lens-2W-040s",
    "lens-2W-100s",
    "lens-3W-040s",
    "lens-3W-100s"
]

greens = plt.cm.Greens(np.linspace(0.5, 1, len(selected_lenses)))
reds = plt.cm.Reds(np.linspace(1, 0.5, len(selected_lenses)))

for source, cmap in zip(["green-He-Ne", "red-He-Ne"], [greens, reds]):
    sub_df = df[
        (df["light_source"] == source) &
        (df["measurement"].isin(selected_lenses))
    ]

    for i, m in enumerate(selected_lenses):
        sub = sub_df[sub_df["measurement"] == m]

        if sub.empty:
            continue

        # remove "lens-" from label
        clean_label = m.replace("lens-", "")

        plt.plot(
            sub["z"],
            sub["diameter"],
            color=cmap[i],
            label=f"{clean_label} ({source.split('-')[0]})"
        )

plt.xlabel("Distance from Lens (mm)")
plt.ylabel("FWHM (µm)")
plt.legend(fontsize=9)
plt.title("Selected Lens Measurements")
plt.show()
# %%
# %% Plot of selected lenses with focal point inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8,6))

# -----------------------------
# FILTER LIST
# -----------------------------
selected_lenses = [
    "lens-2W-100s",
    "lens-3W-040s",
    "lens-3W-100s"
]

# full list for inset (DO NOT FILTER)
lens_names = sorted([m for m in df["measurement"].unique() if "lens" in m])

# -----------------------------
# CUSTOM COLOR MAPS
# -----------------------------
# red: pure red -> orange
red_cmap = LinearSegmentedColormap.from_list(
    "custom_red",
    [
        (1.0, 0.0, 0.0),   # pure red   (255,0,0)
        (1.0, 0.7, 0.0)    # orange     (255,128,0)
    ]
)

# green: pure green -> light blue
green_cmap = LinearSegmentedColormap.from_list(
    "custom_green",
    [
        (0.0, 1.0, 0.0),   # pure green (0,255,0)
        (0.0, 1.0, 0.7)    # light blue (0,128,255)
    ]
)

greens = green_cmap(np.linspace(0, 1, len(selected_lenses)))
reds = red_cmap(np.linspace(0, 1, len(selected_lenses)))

# -----------------------------
# FONT SIZES
# -----------------------------
main_font = 14
inset_font = 12

# -----------------------------
# MAIN PLOT
# -----------------------------
for i, lens in enumerate(selected_lenses):

    for source, cmap in zip(
        ["green-He-Ne", "red-He-Ne"],
        [greens, reds]
    ):

        sub = df[
            (df["measurement"] == lens) &
            (df["light_source"] == source)
        ]

        if sub.empty:
            continue

        clean_label = lens.replace("lens-", "")

        ax.plot(
            sub["z"],
            sub["diameter"],
            color=cmap[i],
            linewidth=2,
            label=f"{clean_label} ({source.split('-')[0]})"
        )

# remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))

ax.legend(
    unique.values(),
    unique.keys(),
    loc="lower right",
    fontsize=main_font
)

# -----------------------------
# INSET
# -----------------------------
inset = False
if inset == True:
    axins = inset_axes(ax, width="38%", height="28%", loc="upper left",borderpad=1)

    offset = 1.5
    bar_width = 0.4

    x_positions = np.arange(len(lens_names)) * offset

    for i, lens in enumerate(lens_names):

        row_green = df_fit[
            (df_fit["measurement"] == lens) &
            (df_fit["light_source"] == "green-He-Ne")
        ]

        row_red = df_fit[
            (df_fit["measurement"] == lens) &
            (df_fit["light_source"] == "red-He-Ne")
        ]

        if not row_green.empty:
            axins.bar(
                x_positions[i] - bar_width/2,
                row_green["focal_point"].values[0],
                width=bar_width,
                color=(0.0, 0.8, 0.2)
            )

        if not row_red.empty:
            axins.bar(
                x_positions[i] + bar_width/2,
                row_red["focal_point"].values[0],
                width=bar_width,
                color=(1.0, 0.3, 0.0)
            )

    # -----------------------------
    # INSET STYLING
    # -----------------------------
    axins.set_xticks(x_positions)

    axins.set_xticklabels(
        [l.replace("lens-", "") for l in lens_names],
        rotation=45,
        fontsize=inset_font
    )

    # y-axis on the right
    axins.yaxis.tick_right()
    axins.yaxis.set_label_position("right")

    # invert y-axis direction
    axins.invert_yaxis()

    axins.set_ylabel(
        "$f_{theo}$ / mm",
        fontsize=inset_font
    )

    #axins.set_title("Theoretical Focal Points", fontsize=inset_font)

    axins.tick_params(axis='y', labelsize=inset_font)
    axins.tick_params(axis='x', labelsize=inset_font)

# -----------------------------
# MAIN AXIS LABELS
# -----------------------------
ax.set_xlabel("Distance / mm", fontsize=main_font)
ax.set_ylabel("Spotsize (FWHM) / µm", fontsize=main_font)

ax.tick_params(axis='both', labelsize=main_font)

plt.tight_layout()
plt.savefig("Beam_Divergence_Lenses.png")
plt.show()






# %%
# %% Plot of selected lenses with focal point inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# HELPER: DARKER COLOR
# -----------------------------
def darker(color, factor=0.5):
    """
    factor=0.5 -> 50% darker
    """
    color = np.array(color[:3])
    return tuple(color * factor)

fig, ax = plt.subplots(figsize=(8,6))

# -----------------------------
# FILTER LIST
# -----------------------------
selected_lenses = [
    "lens-2W-100s",
    "lens-3W-040s",
    "lens-3W-100s"
]

# full list for inset (DO NOT FILTER)
lens_names = sorted([m for m in df["measurement"].unique() if "lens" in m])

# -----------------------------
# CUSTOM COLOR MAPS
# -----------------------------
# red: pure red -> orange
red_cmap = LinearSegmentedColormap.from_list(
    "custom_red",
    [
        (1.0, 0.0, 0.0),   # pure red   (255,0,0)
        (1.0, 0.7, 0.0)    # orange     (255,128,0)
    ]
)

# green: pure green -> light blue
green_cmap = LinearSegmentedColormap.from_list(
    "custom_green",
    [
        (0.0, 1.0, 0.0),   # pure green (0,255,0)
        (0.0, 1.0, 0.7)    # light blue (0,128,255)
    ]
)text

greens = green_cmap(np.linspace(0, 1, len(selected_lenses)))
reds = red_cmap(np.linspace(0, 1, len(selected_lenses)))

# -----------------------------
# FONT SIZES
# -----------------------------
main_font = 14
inset_font = 12

# -----------------------------
# MAIN PLOT
# -----------------------------
fit_style = "darker"
# OPTIONS:
# "same"   -> same color as data
# "black"  -> black fit lines
# "darker" -> 50% darker version

for i, lens in enumerate(selected_lenses):

    for source, cmap in zip(
        ["green-He-Ne", "red-He-Ne"],
        [greens, reds]
    ):

        sub = df[
            (df["measurement"] == lens) &
            (df["light_source"] == source)
        ]

        if sub.empty:
            continue

        clean_label = lens.replace("lens-", "")

        # -----------------------------
        # DATA COLOR
        # -----------------------------
        data_color = cmap[i]

        # -----------------------------
        # FIT COLOR OPTIONS
        # -----------------------------
        if fit_style == "same":
            fit_color = data_color

        elif fit_style == "black":
            fit_color = "black"

        elif fit_style == "darker":
            fit_color = darker(data_color, factor=0.5)

        else:
            fit_color = data_color

        # -----------------------------
        # RAW DATA
        # -----------------------------
        ax.plot(
            sub["z"],
            sub["diameter"],
            color=data_color,
            linewidth=2,
            label=f"{clean_label} ({source.split('-')[0]})"
        )

        # -----------------------------
        # LINEAR FIT
        # -----------------------------
        fit_row = df_fit[
            (df_fit["measurement"] == lens) &
            (df_fit["light_source"] == source)
        ]

        if not fit_row.empty:

            slope = fit_row["slope"].values[0]
            intercept = fit_row["intercept"].values[0]

            z_fit = np.linspace(
                sub["z"].min(),
                sub["z"].max(),
                200
            )

            y_fit = slope * z_fit + intercept

            ax.plot(
                z_fit,
                y_fit,
                linestyle="--",
                linewidth=2,
                color=fit_color,
                alpha=0.9
            )

# remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))

ax.legend(
    unique.values(),
    unique.keys(),
    loc="lower right",
    fontsize=main_font
)

# -----------------------------
# INSET
# -----------------------------
axins = inset_axes(ax, width="38%", height="28%", loc="upper left",borderpad=1)

offset = 1.5
bar_width = 0.4

x_positions = np.arange(len(lens_names)) * offset

for i, lens in enumerate(lens_names):

    row_green = df_fit[
        (df_fit["measurement"] == lens) &
        (df_fit["light_source"] == "green-He-Ne")
    ]

    row_red = df_fit[
        (df_fit["measurement"] == lens) &
        (df_fit["light_source"] == "red-He-Ne")
    ]

    if not row_green.empty:
        axins.bar(
            x_positions[i] - bar_width/2,
            row_green["focal_point"].values[0],
            width=bar_width,
            color=(0.0, 0.8, 0.2)
        )

    if not row_red.empty:
        axins.bar(
            x_positions[i] + bar_width/2,
            row_red["focal_point"].values[0],
            width=bar_width,
            color=(1.0, 0.3, 0.0)
        )

# -----------------------------
# INSET STYLING
# -----------------------------
axins.set_xticks(x_positions)

axins.set_xticklabels(
    [l.replace("lens-", "") for l in lens_names],
    rotation=45,
    fontsize=inset_font
)

# y-axis on the right
axins.yaxis.tick_right()
axins.yaxis.set_label_position("right")

# invert y-axis direction
axins.invert_yaxis()

axins.set_ylabel(
    "$f_{theo}$ / mm",
    fontsize=inset_font
)

#axins.set_title("Theoretical Focal Points", fontsize=inset_font)

axins.tick_params(axis='y', labelsize=inset_font)
axins.tick_params(axis='x', labelsize=inset_font)

# -----------------------------
# MAIN AXIS LABELS
# -----------------------------
ax.set_xlabel("Distance / mm", fontsize=main_font)
ax.set_ylabel("Spotsize (FWHM) / µm", fontsize=main_font)

ax.tick_params(axis='both', labelsize=main_font)

plt.tight_layout()
plt.savefig("Beam_Divergence_Lenses_inset_and_fits.png")
plt.show()







# %% Good plot of Laser Divergence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("beam_data.csv")
df_fit = pd.read_csv("fit_results.csv")

plt.figure(figsize=(7,5))

for source, color in zip(["green-He-Ne", "red-He-Ne"], ["green", "red"]):
    sub = df[(df["measurement"].str.contains("noSample")) &
             (df["light_source"] == source)]

    # relative distance
    z_rel = sub["z"] - sub["z"].min()

    # raw data
    plt.plot(
        z_rel,
        sub["diameter"],
        color=color,
        linewidth=2,
        label=f"{source.split('-')[0]} data"
    )

    # -----------------------------
    # FIT + DIVERGENCE
    # -----------------------------
    fit_row = df_fit[(df_fit["measurement"].str.contains("noSample")) &
                     (df_fit["light_source"] == source)].iloc[0]

    slope = fit_row["slope"]
    intercept = fit_row["intercept"]

    # IMPORTANT: adjust intercept to relative coordinate system
    z0 = sub["z"].min()
    intercept_rel = slope * z0 + intercept

    y_fit = slope * z_rel + intercept_rel

    phi = np.arctan(slope)
    phi_deg = np.degrees(phi)

    print(f"{source} divergence angle: {phi:.4f}°")

    plt.plot(
        z_rel,
        y_fit,
        linestyle="--",
        color=color,
        linewidth=2,
        label=f"{source.split('-')[0]} fit (φ = {phi:.3f}°)"
    )

# -----------------------------
# STYLING
# -----------------------------
plt.xlabel("Relative Distance / mm", fontsize=14)
plt.ylabel("Spotsize (FWHM) / µm", fontsize=14)
#plt.title("Laser Divergence (no sample)", fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("Laser-Divergence")
plt.show()
# %%
