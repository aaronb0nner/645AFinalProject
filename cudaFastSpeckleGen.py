import numpy as np
from PIL import Image, ImageSequence
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.fftpack import get_fft_plan
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc  # Import garbage collector

# Function to create a detector blur
def detector_blur(blurx, blury, si):
    import numpy as np
    from scipy.fft import fft2, ifft2, fftshift
    psf = np.zeros((si, si))  # Initialize the point spread function (PSF) array
    for ii in range(0, blurx - 1):
        for jj in range(0, blury - 1):
            ycord = int(np.round(jj + si / 2))
            xcord = int(np.round(ii + si / 2))
            psf[ycord][xcord] = 1  # Set the PSF values
    otf = np.abs(fft2(psf))  # Compute the optical transfer function (OTF) by taking the FFT of the PSF
    return otf

# Function to create a pupil mask
def make_pupil(r1, r2, si):
    import matplotlib.pyplot as plt
    import numpy as np
    if 2 * np.floor(si / 2) == si:  # Check if si is even
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi)
    if 2 * np.floor(si / 2) != si:  # Check if si is odd
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi + 1)
    pupily = np.transpose(pupilx)  # Transpose to get y-coordinates
    dist2 = np.multiply(pupilx, pupilx) + np.multiply(pupily, pupily)  # Calculate squared distance from center
    dist = np.sqrt(dist2)  # Calculate distance from center
    pupil2 = (dist < r1)  # Mask for points within inner radius
    pupil3 = (dist > r2)  # Mask for points outside outer radius
    pupil = np.multiply(pupil2.astype(int), pupil3.astype(int))  # Combine masks to create pupil
    return pupil

# Function to create a complex pupil with focus
def make_cpupil_focus(r1, r2, si, z, lam, dxx, focal):
    import numpy as np
    if 2 * np.floor(si / 2) == si:  # Check if si is even
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi)
    if 2 * np.floor(si / 2) != si:  # Check if si is odd
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi + 1)
    pupily = np.transpose(pupilx)  # Transpose to get y-coordinates
    dist2 = np.multiply(pupilx, pupilx) + np.multiply(pupily, pupily)  # Calculate squared distance from center
    dist = np.sqrt(dist2)  # Calculate distance from center
    pupil2 = (dist < r1)  # Mask for points within inner radius
    pupil3 = (dist > r2)  # Mask for points outside outer radius
    pupil = np.multiply(pupil2.astype(int), pupil3.astype(int))  # Combine masks to create pupil
    lens_phase = dxx * dxx * np.pi * dist2 / (lam * focal)  # Calculate lens phase
    phase1 = 2 * np.pi * np.sqrt(dxx * dxx * dist2 + z * z) / lam  # Calculate phase term 1
    phase2 = 2 * np.pi * np.sqrt(dxx * dxx * dist2 + (focal + .0001) * (focal + .0001)) / lam  # Calculate phase term 2
    cpupil = np.multiply(np.exp(1j * (phase1 + phase2 - lens_phase)), pupil)  # Create complex pupil
    return cpupil

# Function to create an optical transfer function (OTF)
def make_otf2(scale, cpupil):
    from scipy.fft import fft2
    import numpy as np
    psf = fft2(cpupil)  # Compute the point spread function (PSF) by taking the FFT of the complex pupil
    psf = abs(psf)
    psf = np.multiply(psf, psf)  # Square the PSF
    spsf = np.sum(psf)  # Sum of PSF values
    norm_psf = scale * psf / spsf  # Normalize the PSF
    otf = fft2(norm_psf)  # Compute the optical transfer function (OTF) by taking the FFT of the normalized PSF
    return otf

# Function to create a Gaussian beam
def gaussian_beam(amp, xc, yc, wx, wy, si, dx):
    # amp is the beam amplitude
    # xc is the horizontal center of the beam in units of meters
    # yc is the vertical center of the beam in units of meters 
    # wx is the horizontal beam waist
    # wy is the vertical beam waist
    # si is half the width of the beam array
    # dx is the sample spacing in the beam array
    from scipy.fft import fft2
    import numpy as np
    beam = np.zeros((2 * si, 2 * si))  # Initialize the beam array
    for x in range(0, 2 * si):
        xx = dx * (x - si)
        for y in range(0, 2 * si):
            yy = dx * (y - si)
            beam[x, y] = amp * np.exp(-(xx - xc) * (xx - xc) / wx - (yy - yc) * (yy - yc) / wy) / (np.pi * wx * wy)  # Calculate Gaussian beam values
    return beam

def generate_speckle_pattern(lam, glass, fft_plan, si, lam_min, increment_scale, pupil, speckles, iter, pbar, lock):
    lamda = lam / increment_scale * 1e-9 + lam_min  # Calculate the wavelength in meters
    phase = 2 * np.pi * glass / lamda
    cpupil = np.multiply(pupil, np.exp((0 + 1j) * phase))
    #print(f"Generating Speckle Pattern with λ(nm): {round(lamda * 1e9, 2)}")
    
    # Convert cpupil to CuPy array for FFT
    cpupil_cp = cp.asarray(cpupil)
    
    # Perform FFT using CuPy with precomputed plan
    with fft_plan:
        speckle_field_cp = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cpupil_cp)))
    
    # Convert result back to NumPy array
    speckle_field = cp.asnumpy(speckle_field_cp)
    
    speckle_intensity = np.multiply(np.abs(speckle_field), np.abs(speckle_field))
    speckles[lam] = speckle_intensity  # Store the speckle intensity in the 2D array
    current_speckle = (speckles[lam] - np.mean(speckles[lam])) / np.std(speckles[lam])
    filename = f'speckle_{round(lamda * 1e9)}_phase_{iter}.png'

    if round(lamda * 1e9) == 425 or round(lamda * 1e9) == 625:
        cropped_speckle = current_speckle[900:1100, 900:1100]  # Crop the speckle pattern to 200x200
        plt.imshow(cropped_speckle, cmap='magma')
        plt.title(f'Speckle Pattern at {round(lamda * 1e9)} nm')
        plt.axis('off')  # Turn off axis labels
        plt.savefig(os.path.join('speckle_images', filename), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Perform 2D Fourier Transform on the current speckle pattern
        fft_speckle = np.abs(fftshift(fft2(fftshift(current_speckle))))
        fft_slice = fft_speckle[1000]
        fft_slice_cropped = fft_slice[1100:2000]

        # Plot the Fourier Transform magnitude
        plt.plot(fft_slice)
        plt.title(f'Speckle Pattern Intensity (fft_slice) at {round(lamda * 1e9)} nm')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.savefig(os.path.join('speckle_images', f'speckle_pattern_slice_{round(lamda * 1e9)}_phase_{iter}.png'))
        plt.close()

        plt.plot(fft_slice_cropped)
        plt.title(f'Fourier Transform Information (fft_slice_cropped) at {round(lamda * 1e9)} nm')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.savefig(os.path.join('speckle_images', f'fourier_transform_{round(lamda * 1e9)}_phase_{iter}.png'))
        plt.close()

    # Update progress bar
    with lock:
        pbar.update(1)

def correlate_speckles(speckle_data, iteration_index):
    num_speckles = speckle_data.shape[1] # change me dumbass
    print(speckle_data.shape)
    reference_index = 175
    reference_speckle = (speckle_data[iteration_index, reference_index] - np.mean(speckle_data[iteration_index, reference_index])) / np.std(speckle_data[iteration_index, reference_index])
    correlations = np.zeros(num_speckles)
    wavelengths = np.zeros(num_speckles)

    for lam in range(num_speckles):
        speckle = (speckle_data[iteration_index, lam] - np.mean(speckle_data[iteration_index, lam])) / np.std(speckle_data[iteration_index, lam])
        correlations[lam] = np.mean(np.multiply(reference_speckle, speckle))
        wavelengths[lam] = lam / increment_scale  # Calculate the wavelength increment in nanometers

    wavelengths = wavelengths + 400  # Adjust wavelengths to start from 400 nm

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, correlations, marker='o')
    plt.title(f'Correlation of Speckle Patterns for Against {reference_index+400} nm')
    plt.xlabel('Wavelength of Speckle Pattern (nm)')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.savefig('correlation_plot_generation.png')  # Save the plot to a file
    plt.close()

pca_components = 144  # Number of PCA components to keep
# Function to apply PCA to each individual speckle pattern
def apply_pca_to_speckles(speckles, num_components=pca_components):
    num_speckles = speckles.shape[0]
    flattened_size = speckles.shape[1] * speckles.shape[2]
    
    pca_speckles = np.zeros((num_speckles, num_components))
    reshaped_speckles = speckles.reshape(num_speckles, flattened_size)
    pca = PCA(n_components=num_components)
    pca_speckles = pca.fit_transform(reshaped_speckles)

    # Reshape the PCA output to match the expected dimensions
    pca_speckles = pca_speckles.reshape(num_speckles, int(np.sqrt(num_components)), int(np.sqrt(num_components)))

    return pca_speckles


si = 2000  # Size of the image
fl = .1  # Focal length of the lens
D = .01  # Diameter of the lens
lam_min = 400e-9  # Minimum wavelength
lam_max = 700e-9  # Maximum wavelength
infield = gaussian_beam(1, 0, 0, .002, .002, 1000, 1e-5)

# #Label the graph
# plt.title("Gaussian Beam")
# plt.xlabel("x(um)")
# plt.ylabel("y(um)")
# plt.imshow(infield, cmap='gray')
# plt.show()
#plt.imsave(f'gaussian_beam.png', infield, cmap='gray')  # Create the input field using az Gaussian beam

numIterations = 1  # Number of iterations
numSpeckles = 256  # Total number of speckle patterns to generate per iteration
increment_scale = 1  # Inverse Scale factor to increment the wavelength by, 1 is 1nm, 4 is 0.25nm

#make a directory to store speckle images, move on if it already exists
if not os.path.exists('speckle_images'):
    os.makedirs('speckle_images')
# Initialize the array to store speckle images.
# The first dimension isolates speckle patterns to use the same phase screen
# The second dimension for the different wavelengths of speckle patterns for a given phase screen
# The third and fourth dimensions are the image data itself, assuming after PCA
speckle_imgs = np.zeros((numIterations, numSpeckles, int(np.sqrt(pca_components)), int(np.sqrt(pca_components)))) 

print("Generating speckle patterns...")
print(f"Number of speckle patterns: {numSpeckles}, Increment: {1/increment_scale} nm")

# Precompute the FFT plan for CuPy
fft_plan = get_fft_plan(cp.zeros((si, si), dtype=cp.complex128))

for iter in range(numIterations):
    # Generate a new random phase screen
    #print("Generating speckles on iteration: ", iter)
    glass = 0.4 * np.random.uniform(0, .001, (si, si))  # Generate a random phase screen from a uniform distribution. 0.4 represent the index of refraction of the glass and the air
    speckles = np.zeros((numSpeckles, si, si))  # Initialize a temporary array to store speckle patterns

    lock = threading.Lock()
    with ThreadPoolExecutor() as executor:
        with tqdm(total=numSpeckles, desc=f"Processing speckle patterns on iteration {iter}") as pbar:
            futures = [executor.submit(generate_speckle_pattern, lam, glass, fft_plan, si, lam_min, increment_scale, make_pupil(500 * lam_min / (lam / increment_scale * 1e-9 + lam_min), 0, si), speckles, iter, pbar, lock) for lam in range(numSpeckles)]
            for future in as_completed(futures):
                future.result()

    print(f"Speckle pattern {iter} generated. Converting to PCA format...")
    # Convert the speckle pattern to PCA format and store it to temporary array of PCA transformed images

    #correlate_speckles(speckles, 0) change index on shape dumbass

    pca_speckles = apply_pca_to_speckles(speckles)
    speckle_imgs[iter] = pca_speckles
    print(f"Speckle pattern {iter} converted to PCA format.")
    #correlate_speckles(speckles, 0)
    # Clear variables and run garbage collector
    del speckles, pca_speckles
    gc.collect()

#print the storage size of the speckle images array in a nicely formatted way
print(f"Speckle Images Array Shape: {speckle_imgs.shape}")
print(f"Speckle Images Array Size: {speckle_imgs.nbytes / 1e6} MB")

# Save the speckle_imgs variable to a file including the shape
np.save('speckle_imgs.npy', speckle_imgs)

# Example usage
correlate_speckles(speckle_imgs, 0)

# # Apply PCA to each individual speckle pattern
# pca_speckle_imgs = apply_pca_to_speckles(speckle_imgs)
# print(f"PCA Speckle Images Array Size: {pca_speckle_imgs.nbytes / 1e6} MB")

# # replace the raw speckle images with the PCA transformed images
# speckle_imgs = pca_speckle_imgs
# print(speckle_imgs.shape)

# # Example usage: Plot the first PCA transformed speckle pattern
# plt.imshow(pca_speckle_imgs[0, 0].reshape(8, 8), cmap='gray')
# plt.title('PCA Transformed Speckle Pattern')
# plt.xlabel('x(um)')
# plt.ylabel('y(um)')
# #plt.show()

# correlate_speckles(pca_speckle_imgs, 0)

print("Done")


