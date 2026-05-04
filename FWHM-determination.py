#%%
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import laserbeamsize as lbs


from skimage.measure import regionprops
from scipy.optimize import curve_fit

# -----------------------------
# PARAMETERS
# -----------------------------
#base_folder = "./01-04-2026-Sample9/green-He-Ne"   # contains lens1, lens2, ...
base_folder = "./07-04-2026-Sample9/green-He-Ne-Sample9"
num_lenses = 7

pixel_size = 5.2      # µm
z_step = 1e-3            # 1 mm per mesurement point (away from camera)
distance_offset = 0#38 #mm from sample to camera!


#Parameters for green He-Ne
beam_divergence_angle_gauss = -0.4365821072136346 #0.77534
beam_divergence_angle_fwhm = -0.9599788471708077 #1.09208

#Parameters for 980nm diode:
#beam_divergence_angle_fwhm = 1.6486 #slope of fitting without sample
#beam_divergence_angle_gauss = 1.87269 # in µm/mm

# -----------------------------
# PROCESS EACH LENS
# -----------------------------
all_diameters = []
z_positions = None
debug = False
debug_interval = 5

for lens_idx in range(1, num_lenses + 1):
    folder = os.path.join(base_folder, f"lens{lens_idx}")

    files = sorted(
        [f for f in os.listdir(folder)
         if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    diameters = []
    diameters_FWHM = []

    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        if debug:
            print("Now reading:", fname, "\n")

        img_raw = iio.imread(path).astype(float)

        # convert RGB → grayscale if needed
        if img_raw.ndim == 3:
            img_raw = img_raw.mean(axis=2)

        # -----------------------------
        # BACKGROUND REMOVAL
        # -----------------------------
        img_bg_filter = lbs.subtract_corner_background(img_raw, nT = 3, iso_noise=False)
        peak_pixel = np.unravel_index(np.argmax(img_bg_filter, axis=None), img_bg_filter.shape)
        peak_value = img_bg_filter[peak_pixel]

        half_max = peak_value / 2.0
        
        #crop image to usefull size:
        crop_padding= 20 # optionally add padding to the cropped area
        x_crop_filter = ~np.all(img_bg_filter<half_max,axis=1)
        x_min_crop_filter = min([i for i, val in enumerate(x_crop_filter) if val]) - crop_padding
        x_max_crop_filter = max([i for i, val in enumerate(x_crop_filter) if val]) + crop_padding
        
        y_crop_filter = ~np.all(img_bg_filter<half_max,axis=0)
        y_min_crop_filter = min([i for i, val in enumerate(y_crop_filter) if val]) - crop_padding
        y_max_crop_filter = max([i for i, val in enumerate(y_crop_filter) if val]) + crop_padding

        img_cropped = img_bg_filter[x_min_crop_filter:x_max_crop_filter, y_min_crop_filter:y_max_crop_filter]
        #img_cropped = img_bg_filter[:,~np.all(img_bg_filter<half_max,axis=0)]
        #img_cropped = img_cropped[~np.all(img_cropped<half_max,axis=1),:]
        
        #re compute the position of the center:
        peak_pixel = np.unravel_index(np.argmax(img_cropped, axis=None), img_cropped.shape)
        peak_value = img_cropped[peak_pixel]
        
        #determine FWHM from size of image:
        d_x = img_cropped.shape[0] - 2 * crop_padding
        d_y = img_cropped.shape[1] - 2 * crop_padding
        diameters_FWHM.append(((d_x + d_y)/2)*pixel_size)
        
        #do Gaussian Fitting to determine the beam size
        try:
            # -----------------------------
            # INITIAL BEAM ESTIMATE
            # -----------------------------
            xc, yc, d_major, d_minor, phi = lbs.basic_beam_size(img_cropped)

            # -----------------------------
            # ITERATIVE REFINEMENT
            # -----------------------------
            for _ in range(2):   # 2–3 iterations usually enough
                mask = lbs.rotated_rect_mask(
                    img_cropped,
                    xc, yc,
                    3 * d_major,
                    3 * d_minor,
                    -phi
                )

                masked_img = np.copy(img_cropped)
                masked_img[mask < 1] = 0

                xc, yc, d_major, d_minor, phi = lbs.basic_beam_size(masked_img)

            if debug and i%debug_interval ==0:
                print("Angle of fitting:", phi)
                print("Major / Minor:", d_major, d_minor)

            # average diameter
            d = 0.5 * (d_major + d_minor)
            diameters.append(d * pixel_size)

            if debug and i%debug_interval ==0: 
                print("Found diameter:", d, "\n")

            # -----------------------------
            # DEBUG PLOT (every 2nd image)
            # -----------------------------
            if debug and i%debug_interval ==0:
                plt.figure(figsize=(6, 5))
                plt.imshow(img_cropped)
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
                
                lbs.plot_image_analysis(img_cropped)
        except Exception as e:
            print(f"Error in lens {lens_idx}, image {fname}: {e}")
            diameters.append(np.nan)

        if debug and i % debug_interval == 0:
            plt.figure()
            plt.imshow(img_bg_filter)
            pltTitle = "Background removed, measurement:" + fname + "\n"
            plt.title(pltTitle)
            plt.show()
            plt.figure()
            plt.imshow(img_cropped,cmap="jet")
            #plt.scatter(peak_pixel[1], peak_pixel[0], color="red")
            pltTitle = "Cropped FWHM, measurement:" + fname + "\n"
            plt.title(pltTitle)
             

    x = np.arange(1, len(diameters) + 1) + distance_offset
    x_relative = np.arange(1, len(diameters)+1)
    
    # Original plots
    plt.figure()
    plt.plot(x, diameters, color="blue", label="Gaussian Fit")
    plt.plot(x, np.array(diameters_FWHM), color="red", label="FWHM of cropped region")

    # Linear fits
    fit_gauss = np.polyfit(x, diameters, 1)          # slope, intercept
    fit_fwhm = np.polyfit(x, diameters_FWHM, 1)

    y_fit_gauss = np.polyval(fit_gauss, x)
    y_fit_fwhm = np.polyval(fit_fwhm, x)

    # Plot fitted lines (black dashed)
    plt.plot(x, y_fit_gauss, 'k--')
    plt.plot(x, y_fit_fwhm, 'k--')

    # Labels and legend
    plt.xlabel("Distance from Lens / mm")
    plt.ylabel("Lens Diameter / µm")
    plt.legend()
    plt.title("Raw data")
    plt.show()
    
    # Original plots
    plt.figure()
    diameter_gauss_corrected = np.array(diameters) - beam_divergence_angle_gauss * x_relative
    diameter_fwhm_corrected = np.array(diameters_FWHM) - beam_divergence_angle_fwhm * x_relative 
    plt.plot(x, diameter_gauss_corrected, color="blue", label="Gaussian Fit")
    plt.plot(x, diameter_fwhm_corrected, color="red", label="FWHM of cropped region")

    # Linear fits
    fit_gauss_corrected = np.polyfit(x, diameter_gauss_corrected, 1)          # slope, intercept
    fit_fwhm_corrected = np.polyfit(x, diameter_fwhm_corrected, 1)

    y_fit_gauss_corrected = np.polyval(fit_gauss_corrected, x)
    y_fit_fwhm_corrected = np.polyval(fit_fwhm_corrected, x)

    # Plot fitted lines (black dashed)
    plt.plot(x, y_fit_gauss_corrected, 'k--')
    plt.plot(x, y_fit_fwhm_corrected, 'k--')

    # Labels and legend
    plt.xlabel("Distance from Lens / mm")
    plt.ylabel("Lens Diameter / µm")
    plt.legend()
    plt.title("Slope corrected beam diameters")
    plt.show()
    
    #compute the focal length of the lens:
    divergence_angle_gauss = np.arctan(fit_gauss[0])
    divergence_angle_FWHM = np.arctan(fit_fwhm[0])
    
    lens_diameter =  280 #µm
    
    f_gauss = lens_diameter / (2*np.sin(-divergence_angle_gauss))
    f_fwhm = lens_diameter / (2*np.sin(-divergence_angle_FWHM))
    
    #print("Lens focal length (from Gauss fit):", str(round(f_gauss,ndigits=2)),"µm", "\n")
    #print("Lens focal length (from FWHM):", str(round(f_fwhm,ndigits=2)),"µm", "\n")
    
    zero_intersect_fwhm = (-fit_fwhm_corrected[1])/(fit_fwhm_corrected[0])
    zero_intersect_gauss = (-fit_gauss_corrected[1])/(fit_gauss_corrected[0])
    print("Theoretical focal point from beam divergence slope:\n", str(round(zero_intersect_gauss,ndigits=2)), "(gauss) and:", str(round(zero_intersect_fwhm,ndigits=2)), "(FWHM) \n")
    
    print("Slope Gauss:", fit_gauss[0], " (mm) Slope FWHM:", fit_fwhm[0] ," (mm)\n")
    
    #%%
    plt.figure()
    x_ax = np.arange(-200,200)
    plt.scatter(x_ax, np.polyval(fit_gauss,x_ax))
    plt.show()