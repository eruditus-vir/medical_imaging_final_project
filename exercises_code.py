from skimage import measure
from skimage.morphology import binary_erosion
from utils import filepath
import scipy
import skimage.feature
from scipy import ndimage
from skimage.filters import gabor_kernel
import typing as tp
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage.morphology import binary_dilation
from scipy.optimize import least_squares


def load_dcm(filepath: str):
    """ Load a DICOM file. """
    # YOUR CODE HERE: use pydicom.dcmread(...)
    # ...
    return pydicom.dcmread(filepath)


def estimate_noisy_pixels(img: np.ndarray):
    """ Estimate the noisy pixels in the background of an image. """
    noise_threshold = 300  # Medido en [T1]
    noise_mask = (img < noise_threshold) * (img > 0)
    return noise_mask


def power_of_signal(signal_or_img: np.ndarray):
    """ Compute the power of a signal.

    """
    # YOUR CODE HERE
    # ...
    return np.sqrt(np.mean(signal_or_img ** 2))


def contrast_of_signal(signal_or_img: np.ndarray):
    """ Compute the contrast of a signal. """
    # YOUR CODE HERE
    # ...
    return np.abs(np.max(signal_or_img) - np.min(signal_or_img))


def compute_snr(signal_power: float, noise_power: float):
    """ Compute the signal-to-noise ratio (SNR) of a signal. """
    # YOUR CODE HERE
    # ...
    return signal_power / noise_power


def compute_cnr(signal_contrast: float, noise_power: float):
    """ Compute the contrast-to-noise ratio (CNR) of a signal. """
    # YOUR CODE HERE
    # ...
    return signal_contrast / noise_power


def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    _, _, sz_z = img_dcm.shape
    sagittal_plane = img_dcm[:, :, sz_z // 2]  # Why //2? Why on third dimension?
    sagittal_plane = np.rot90(sagittal_plane, k=-1)  # Better visualization
    return sagittal_plane


def segment_bones(img_ct: np.ndarray) -> np.ndarray:
    """ Segment the bones of a CT image. """
    # Your code here:
    #   should return a boolean mask (positive/negative) or an integer mask (labels)?
    #   See `skimage.measure.label(...)`.
    # ...
    return measure.label(img_ct)


def visualize_side_by_side(img: np.ndarray, mask: np.ndarray):
    """ Visualize image and mask in two different subplots. """
    # Your code here:
    #   See `plt.subplot(...)`, `plt.imshow(...)`, `plt.show(...)`.
    #   Which colormap should you choose?
    #   Which aspect ratio should you choose?
    # ...
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    # Plot image in the first subplot
    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')  # Turn off axis labels

    # Plot mask in the second subplot
    axes[1].imshow(mask, cmap='gray')  # Use gray colormap for masks
    axes[1].set_title('Mask')
    axes[1].axis('off')  # Turn off axis labels

    # Show the plot
    plt.show()

    return


def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ Apply a colormap to a 2D image. """
    # Your code here: See `matplotlib.colormaps[...]`.
    # ...
    # cm = plt.get_cmap(cmap_name)
    cm = matplotlib.cm.get_cmap(cmap_name)
    return cm(img)


def visualize_alpha_fusion(img: np.ndarray,
                           mask: np.ndarray,
                           alpha: float = 0.25,
                           visualize=True,
                           lower_filter=-700,
                           upper_filter=500,):
    """ Visualize both image and mask in the same plot. """
    # Your code here:
    #   Remember the Painter's Algorithm with alpha blending
    #   https://en.wikipedia.org/wiki/Alpha_compositing
    # ...
    # assume not normalize so we normalize first
    img = img.copy()
    # This depends on your image, I used hist to find out best
    img[img < lower_filter] = lower_filter
    img[img > upper_filter] = upper_filter
    norm = matplotlib.colors.Normalize(vmin=np.min(img), vmax=np.max(img))
    img = norm(img)
    bone_img = apply_cmap(img, cmap_name='bone')
    norm_mask = matplotlib.colors.Normalize(vmin=np.min(mask), vmax=np.max(mask))
    prism_mask = apply_cmap(norm_mask(mask), cmap_name='prism') * mask[..., np.newaxis].astype('bool')
    fused_img = bone_img * (1-alpha) + prism_mask * alpha

    # we will use this one func to also get alpha fusion for making gif
    if visualize:
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))
        # Plot CT scan
        axes[0].imshow(bone_img)
        axes[0].set_title('CT Scan')

        # Plot segmentation mask
        axes[1].imshow(prism_mask)
        axes[1].set_title('Segmentation')

        # Plot fused image
        axes[2].imshow(fused_img, aspect=1)
        axes[2].set_title(f'Segmentation with alpha {alpha}')
    return fused_img


def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, :, img_dcm.shape[2] // 2]  # Why //2? -> need integer

def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, img_dcm.shape[1] // 2, :]

def median_axial_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median axial plane of the CT image provided. """
    return img_dcm[img_dcm.shape[0] // 2, :, :]

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    # Your code here:
    #   See `np.max(...)`
    # ...
    return np.max(img_dcm, axis=2)


def AIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the sagittal orientation. """
    # Your code here:
    #   See `np.mean(...)`
    # ...
    return np.mean(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    # Your code here:
    # ...
    return np.max(img_dcm, axis=1)


def AIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the coronal orientation. """
    # Your code here:
    # ...
    return np.mean(img_dcm, axis=1)

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    # Your code here:
    #   See `scipy.ndimage.rotate(...)`
    # ...
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False, mode='nearest', order=0)


def apply_canny(img, sigma):
    """ Apply Canny edge detector to image. """
    # Your code here:
    #   See `skimage.feature.canny(...)`
    # ...

    return skimage.feature.canny(img, sigma=sigma)


def visualize_img_and_edges(grayscale_image, edges_image):
    """ Visualize original image and edges. """
    # Your code here:
    #   Remember `plt.subplots(...)` and `plt.imshow(...)`.
    # ...
    fig, ax = plt.subplots(1, 2, clear=True)
    ax[0].imshow(grayscale_image)
    ax[1].imshow(edges_image)


def create_filter_bank():
    """ Adapted from skimage documentation """
    kernels = []
    for theta in range(6):
        theta = theta / 4. * np.pi
        for sigma in (1, 3, 5):
            for frequency in (0.05, 0.15, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels


def visualize_filter_bank(kernel1, kernel2, kernel3, kernel4, kernel5, kernel6):
    """ Visualize filter bank. """
    # Your code here:
    #   Remember `plt.subplots(...)` and `plt.imshow(...)`.
    # ...
    fig, ax = plt.subplots(2, 3, clear=True)
    ax[0, 0].imshow(kernel1)
    ax[0, 1].imshow(kernel2)
    ax[0, 2].imshow(kernel3)
    ax[1, 0].imshow(kernel4)
    ax[1, 1].imshow(kernel5)
    ax[1, 2].imshow(kernel6)


def apply_filter(image, kernel):
    """ Apply linear filter to image. """
    # Your code here:
    #   See `ndimage.convolve(...)`
    # ...
    return ndimage.convolve(image, kernel)


def get_marker(img: np.ndarray, position: tp.Tuple):
    """ Create a boolean mask of 0s, except for a 1 at the location `position`."""
    marker = np.empty_like(img)
    marker[position] = 1
    return marker


def multiply_quaternions(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[
    float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    a, b, c, d = q1
    w, x, y, z = q2

    return (a * w - b * x - c * y - d * z, (a * x + b * w + c * z - d * y), (a * y - b * z + c * w + d * x),
            (a * z + b * y - c * x + d * w))


def conjugate_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return q[0], -q[1], -q[2], -q[3]


def translation(point: tuple[float, float, float], translation_vector: tuple[float, float, float]) -> tuple[
    float, float, float]:
    """ Perform translation of `point` by `translation_vector`. """
    x, y, z = point
    v1, v2, v3 = translation_vector
    # Your code here
    # ...
    return (x + v1), (y + v2), (z + v3)


def axial_rotation(point: tuple[float, float, float], angle_in_rads: float,
        axis_of_rotation: tuple[float, float, float]) -> tuple[float, float, float]:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """
    x, y, z = point
    v1, v2, v3 = axis_of_rotation
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    # Your code here:
    #   ...
    s = angle_in_rads / 2.0
    cos_item = math.cos(s)
    sin_item = math.sin(s)
    rotation_quaternian = (cos_item, v1 * sin_item, v2 * sin_item, v3 * sin_item)
    point_quaternian = (0, x, y, z)
    rotated_point = multiply_quaternions(rotation_quaternian, point_quaternian)
    rotated_point = multiply_quaternions(rotated_point, conjugate_quaternion(rotation_quaternian))
    return rotated_point[1], rotated_point[2], rotated_point[3]


def get_amygdala_mask(img_atlas: np.ndarray) -> np.ndarray:
    # Your code here:
    #   ...
    mask = np.zeros_like(img_atlas)
    mask[(img_atlas == 45) | (img_atlas == 46)] = 1
    return mask


def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    idcs = np.where(mask == 1)
    centroid = np.stack([np.mean(idcs[0]), np.mean(idcs[1]), np.mean(idcs[2]), ])
    return centroid


def visualize_axial_slice(img: np.ndarray,
                          mask: np.ndarray,
                          mask_centroid: np.ndarray,
                          dim=0):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    # Your code here
    #   Remember `matplotlib.colormaps['cmap_name'](...)`
    #   See also `matplotlib.colors.Normalize(vmin=..., vmax=...)`
    #   ...
    img_slice = img[mask_centroid[dim].astype('int'), :, :]
    mask_slice = mask[mask_centroid[dim].astype('int'), :, :]

    cmap = matplotlib.cm.get_cmap('bone')
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.5*cmap(norm(img_slice))[..., :3] + \
        0.5*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)
    plt.imshow(fused_slice)
    plt.show()



def find_region_volume(region_mask):
    """ Returns the volume of the region in mm^3. """
    # Your code here:
    #   ...
    return np.sum(region_mask)

# def find_region_surface(mask):
#     """ Returns the surface of the region in mm^2. """
#     # Your code here:
#     #   See `skimage.morphology.binary_erosion()` and `skimage.morphology.binary_dilation()`
#     #   ...
#     eroded_mask = binary_erosion(mask, np.ones((3, 3, 3)))
#     inner = mask - eroded_mask
#     dilated_mask = binary_dilation(mask, np.ones((3, 3, 3)))
#     outer = dilated_mask - mask
#     return np.mean([np.sum(inner), np.sum(outer)])
def find_region_surface(mask):
    """ Returns the surface of the region in mm^2. """
    inner_surface = mask - binary_erosion(mask, np.ones((3, 3, 3)))
    outer_surface = binary_dilation(mask, np.ones((3, 3, 3))) - mask
    return (np.sum(inner_surface) + np.sum(outer_surface) ) / 2     # Average of inner and outer surface




def translation_then_axialrotation(point: tuple[float, float, float], parameters: tuple[float, ...]):
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """
    x, y, z = point
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    # no need, we alr did no 5 above
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1/v_norm, v2/v_norm, v3/v_norm
    #Your code here:
    #   ...
    translated = translation(point, translation_vector=(t1,t2,t3))
    rotated = axial_rotation(
        point=translated,
        angle_in_rads=angle_in_rads,
        axis_of_rotation=(v1,v2,v3)
    )
    return rotated


def screw_displacement(point: tuple[float, float, float], parameters: tuple[float, ...]):
    """ Apply to `point` the screw displacement defined by `parameters`. """
    x, y, z = point
    v1, v2, v3, angle_in_rads, displacement = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1/v_norm, v2/v_norm, v3/v_norm
    # Your code here:
    #   ...
    translation_vector = (v1*displacement, v2*displacement, v3*displacement)

    #
    displaced_point = translation_then_axialrotation(point=point, parameters=(translation_vector[0], translation_vector[1], translation_vector[2], angle_in_rads, v1, v2, v3))
    return displaced_point

def vector_of_residuals(ref_points: np.ndarray, inp_points: np.ndarray) -> np.ndarray:
    """ Given arrays of 3D points with shape (point_idx, 3), compute vector of residuals as their respective distance """
    # Your code here:
    #   ...
    # diffs = ref_points - inp_points
    # # Compute the Euclidean distance (norm) for each pair of points
    # residuals = np.linalg.norm(diffs, axis=1)
    # return tuple(residuals)

    # same as above
    return np.sqrt(np.sum((ref_points-inp_points)**2, axis=1))


def coregister_landmarks(ref_landmarks: np.ndarray, inp_landmarks: np.ndarray):
    """ Coregister two sets of landmarks using a rigid transformation. """
    initial_parameters = [
        0, 0, 0,    # Translation vector
        0,          # Angle in rads
        1, 0, 0,    # Axis of rotation
    ]
    # Find better initial parameters
    centroid_ref = np.mean(ref_landmarks, axis=0)
    centroid_inp = np.mean(inp_landmarks, axis=0)
    # Your code here:
    # translation vector? should move towards this basically
    diff = centroid_inp - centroid_ref
    # use centroid diff as translation vector
    initial_parameters[0] = diff[0]
    initial_parameters[1] = diff[1]
    initial_parameters[2] = diff[2]
    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        # Your code here:
        # this should each input landmarks should first be translated then
        all_points = []
        for pt in range(inp_landmarks.shape[0]):
            all_points.append(translation_then_axialrotation(inp_landmarks[pt, :], parameters))
        all_points = np.array(all_points)
        return sum(vector_of_residuals(ref_landmarks, all_points))

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        verbose=1)
    return result
