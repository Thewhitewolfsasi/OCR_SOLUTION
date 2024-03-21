import cv2
import os
import numpy as np
import pandas as pd
import pytesseract as pt
from PIL import Image, ImageEnhance
import matplotlib.image as image

og = '/usr/share/tesseract-ocr/4.00/tessdata'
custom_config = f"--tessdata-dir {og}"

# # from skimage.transform import radon
# # from PIL import Image
# # from numpy import asarray, mean, array, blackman
# # import numpy as np
# # from numpy.fft import rfft
# # import matplotlib.pyplot as plt
# # try:
# #     # More accurate peak finding from
# #     # https://gist.github.com/endolith/255291#file-parabolic-py
# #     from parabolic import parabolic

# #     def argmax(x):
# #         return parabolic(x, np.argmax(x))[0]
# # except ImportError:
# #     from numpy import argmax


# # def rms_flat(a):
# #     """
# #     Return the root mean square of all the elements of a, flattened out.
# #     """
# #     return np.sqrt(np.mean(np.abs(a) ** 2))


# # filename = '/home/sasi/intern/tesseract/casting_image/cast.lang.exp0.jpeg'

# # # Load file, converting to grayscale
# # I = asarray(Image.open(filename).convert('L'))
# # I = I - mean(I)  # Demean; make the brightness extend above and below zero
# # plt.subplot(2, 2, 1)
# # plt.imshow(I)

# # # Do the radon transform and display the result
# # sinogram = radon(I)

# # plt.subplot(2, 2, 2)
# # plt.imshow(sinogram.T, aspect='auto')
# # plt.gray()

# # # Find the RMS value of each row and find "busiest" rotation,
# # # where the transform is lined up perfectly with the alternating dark
# # # text and white lines
# # r = array([rms_flat(line) for line in sinogram.transpose()])
# # rotation = argmax(r)
# # print('Rotation: {:.2f} degrees'.format(90 - rotation))
# # plt.axhline(rotation, color='r')

# # # Plot the busy row
# # row = sinogram[:, rotation]
# # N = len(row)
# # plt.subplot(2, 2, 3)
# # plt.plot(row)

# # # Take spectrum of busy row and find line spacing
# # window = blackman(N)
# # spectrum = rfft(row * window)
# # plt.plot(row * window)
# # frequency = argmax(abs(spectrum))
# # line_spacing = N / frequency  # pixels
# # print('Line spacing: {:.2f} pixels'.format(line_spacing))

# # plt.subplot(2, 2, 4)
# # plt.plot(abs(spectrum))
# # plt.axvline(frequency, color='r')
# # plt.yscale('log')
# # plt.show()


# #!/usr/bin/env python3
# # from PIL import Image
# # import cv2 
# # import numpy as np
# # import pytesseract as pt

# # og = '/usr/share/tesseract-ocr/4.00/tessdata'
# # custom_config = f"--tessdata-dir {og}"

# # # get grayscale image
# # def get_grayscale(image):
# #     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # noise removal
# # def remove_noise(image):
# #     return cv2.medianBlur(image,5)
 
# # #thresholding
# # def thresholding(image):
# #     return cv2.threshold(image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # #dilation
# # def dilate(image):
# #     kernel = np.ones((5,5),np.uint8)
# #     return cv2.dilate(image, kernel, iterations = 1)
    
# # #erosion
# # def erode(image):
# #     kernel = np.ones((5,5),np.uint8)
# #     return cv2.erode(image, kernel, iterations = 1)

# # #opening - erosion followed by dilation
# # def opening(image):
# #     kernel = np.ones((5,5),np.uint8)
# #     return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# # #canny edge detection
# # def canny(image):
# #     return cv2.Canny(image, 100, 200)

# # #skew correction
# # def deskew(image):
# #     coords = np.column_stack(np.where(image > 0))
# #     angle = cv2.minAreaRect(coords)[-1]
# #     if angle < -45:
# #         angle = -(90 + angle)
# #     else:
# #         angle = -angle
# #     (h, w) = image.shape[:2]
# #     center = (w // 2, h // 2)
# #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
# #     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# #     return rotated

# # #template matching
# # def match_template(image, template):
# #     return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

# # #Resizing images
# # def resized(image, new_width=760):
# #     w, h = image.shape[:2]
# #     ratio = h/w
# #     new_height = int(ratio*new_width)
# #     resized_img = cv2.resize(img, dsize=(new_height,new_width))
# #     return resized_img

# img = cv2.imread("/home/sasi/Desktop/sample.png")
# norm_img = np.zeros((img.shape[0], img.shape[1]))
# img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

# def deskew(image):
#     co_ords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(co_ords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# def set_image_dpi(file_path):
#     im = Image.open(file_path)
#     length_x, width_y = im.size
#     factor = min(1, float(1024.0 / length_x))
#     size = int(factor * length_x), int(factor * width_y)
#     im_resized = im.resize(size, Image.ANTIALIAS)
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
#     temp_filename = temp_file.name
#     im_resized.save(temp_filename, dpi=(300, 300))
#     return temp_filename

# def remove_noise(image):
#     return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

# img = cv2.imread('j.png',0)
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img, kernel, iterations = 1)
# # gray = get_grayscale(img)
# # thresh = thresholding(gray)
# # open = opening(gray)
# # # skw = resized(open)
# # canny = canny(open)

# text = pt.image_to_string(canny, lang='eng', config=custom_config)
# print('text;-\n'+text)

# cv2.imshow('Original Image', canny)
# d = resized(canny)
# cv2.imshow('Resized Image',d)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import matplotlib.pyplot as plt
# import numpy as np
# import cv2


# def get_cdf_hist(image_input):
#     """
#     Method to compute histogram and cumulative distribution function

#     :param image_input: input image
#     :return: cdf
#     """
#     hist, bins = np.histogram(image_input.flatten(), 256, [0, 256])
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * float(hist.max()) / cdf.max()
#     return cdf_normalized


# def contrast_brightness(image, alpha, beta):
#     """
#     Linear transformation function to enhance brightness and contrast

#     :param image: input image
#     :param alpha: contrast factor
#     :param beta: brightness factor
#     :return: enhanced image
#     """
#     enhanced_image = np.array(alpha*image + beta)
#     enhanced_image[enhanced_image > 255] = 255
#     cdf = get_cdf_hist(enhanced_image)
#     return enhanced_image, cdf


# def gamma_enhancement(image, gamma):
#     """
#     Non-linear transformation function to enhance brightness and contrast
#     :param image: input image
#     :param gamma: contrast enhancement factor
#     :return: enhanced image
#     """
#     normalized_image = image / np.max(image)
#     enhanced_image = np.power(normalized_image, gamma)
#     enhanced_image = enhanced_image * 255
#     cdf = get_cdf_hist(enhanced_image)
#     return enhanced_image, cdf


# def log_enhancement(image, gain):
#     """
#     Non-linear transformation function to enhance brightness and contrast
#     :param image: input image
#     :param gain: contrast enhancement factor
#     :return: enhanced image
#     """
#     normalized_image = image / np.max(image)
#     enhanced_image = gain*np.log1p(normalized_image)
#     enhanced_image = enhanced_image * 255
#     cdf = get_cdf_hist(enhanced_image)
#     return enhanced_image, cdf


# def gauss_enhancement(image, gain):
#     """
#     Non-linear transformation function to enhance brightness and contrast
#     :param image: input image
#     :param gain: contrast enhancement factor
#     :return: enhanced image
#     """
#     normalized_image = image / np.max(image)
#     enhanced_image = 1 - np.exp(-normalized_image**2/gain)
#     enhanced_image = enhanced_image*255
#     cdf = get_cdf_hist(enhanced_image)
#     return enhanced_image, cdf


# def hist_enhancement(image):
#     """
#     Histogram equalization to enhance the input image
#     :param image: input image
#     :return: enhanced image
#     """
#     enhanced_image = cv2.equalizeHist(image)
#     cdf = get_cdf_hist(enhanced_image)
#     return enhanced_image, cdf


# def clahe_enhancement(image, threshold, grid_size=(16, 16)):
#     """
#     Adaptive histogram equalization to enhance the input image
#     :param image: input image
#     :param threshold: clipping threshold
#     :param grid_size: local neighbourhood
#     :return: enhanced image
#         """
#     clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=grid_size)
#     enhanced_image = clahe.apply(image)
#     cdf = get_cdf_hist(enhanced_image)
#     return enhanced_image, cdf


# def image_enhancement_spatial():
#     """
#     Main method to change the pixels spatially
#     :return: image grid
#     """
#     image_retina = plt.imread("slo_input.jpg")
#     image_slo = plt.imread("retina_input.jpg")

#     cdf_input = get_cdf_hist(image_slo)

#     fig, axs = plt.subplots(5, 2, figsize=(8, 15))
#     axs[0, 0].imshow(image_slo, cmap='gray', vmin=0, vmax=255)
#     axs[0, 0].set_title("Input")
#     axs[0, 1].hist(image_slo.flatten(), 256, [0, 256], color='r')
#     axs[0, 1].plot(cdf_input)

#     enhanced_cb, cdf_cb = contrast_brightness(image_slo, 1.6, 20)
#     axs[1, 0].imshow(enhanced_cb, cmap='gray', vmin=0, vmax=255)
#     axs[1, 0].set_title("Linear")
#     axs[1, 1].hist(enhanced_cb.flatten(), 256, [0, 256], color='r')
#     axs[1, 1].plot(cdf_cb)

#     enhanced_gamma, cdf_gamma = gamma_enhancement(image_slo, 0.55)
#     axs[2, 0].imshow(enhanced_gamma, cmap='gray', vmin=0, vmax=255)
#     axs[2, 0].set_title("Non-linear (Gamma)")
#     axs[2, 1].hist(enhanced_gamma.flatten(), 256, [0, 256], color='r')
#     axs[2, 1].plot(cdf_gamma)

#     enhanced_log, cdf_log = log_enhancement(image_slo, 1.65)
#     axs[3, 0].imshow(enhanced_log, cmap='gray', vmin=0, vmax=255)
#     axs[3, 0].set_title("Non-linear (Log)")
#     axs[3, 1].hist(enhanced_log.flatten(), 256, [0, 256], color='r')
#     axs[3, 1].plot(cdf_log)

#     enhanced_gauss, cdf_gauss = gauss_enhancement(image_slo, 0.15)
#     axs[4, 0].imshow(enhanced_gauss, cmap='gray', vmin=0, vmax=255)
#     axs[4, 0].set_title("Non-linear (Inverse Gauss)")
#     axs[4, 1].hist(enhanced_gauss.flatten(), 256, [0, 256], color='r')
#     axs[4, 1].plot(cdf_gauss)

#     for i in range(5):
#         for j in range(2):
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])
#     plt.tight_layout()
#     plt.savefig("retina.png")
#     plt.show()


# def image_enhancement_spectral():
#     """
#     Main method to change the pixels based on distribution spectrum
#     :return: image grid
#     """

#     image_slo = cv2.imread("/home/sasi/Desktop/brighten.jpeg")
#     image_retina = cv2.imread("/home/sasi/Desktop/normal.jpeg")

#     cdf_input = get_cdf_hist(image_slo)

#     fig, axs = plt.subplots(2, 3, figsize=(8, 6))
#     axs[0, 0].imshow(image_slo, cmap='gray', vmin=0, vmax=255)
#     axs[0, 0].set_title("Input")
#     axs[1, 0].hist(image_slo.flatten(), 256, [0, 256], color='r')
#     axs[1, 0].plot(cdf_input)

#     enhanced_hist, cdf_hist = hist_enhancement(image_slo)
#     axs[0, 1].imshow(enhanced_hist, cmap='gray', vmin=0, vmax=255)
#     axs[0, 1].set_title("Histogram Equalization")
#     axs[1, 1].hist(enhanced_hist.flatten(), 256, [0, 256], color='r')
#     axs[1, 1].plot(cdf_hist)

#     enhanced_clahe, cdf_clahe = clahe_enhancement(image_slo, 10)
#     axs[0, 2].imshow(enhanced_clahe, cmap='gray', vmin=0, vmax=255)
#     axs[0, 2].set_title("CLAHE")
#     axs[1, 2].hist(enhanced_clahe.flatten(), 256, [0, 256], color='r')
#     axs[1, 2].plot(cdf_clahe)

#     for i in range(2):
#         for j in range(3):
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])
#     plt.tight_layout()
#     plt.show()


# image_enhancement_spectral()

# import cv2
# import numpy as np
# from PIL import Image, ImageEnhance

# def identify_parameters(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Compute average brightness
#     average_brightness = np.mean(gray_image)

#     # Compute average contrast
#     average_contrast = np.std(gray_image)

#     # Compute vibrance (average saturation)
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     average_saturation = np.mean(hsv_image[:, :, 1])

#     # Compute sharpness using Laplacian
#     sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()

#     return average_brightness, average_contrast, average_saturation, sharpness

# def adjust_parameters(image, brightness_factor, contrast_factor, vibrance_factor, sharpness_factor):
#     # Adjust brightness
#     enhanced_image = change_brightness(image, value=brightness_factor)

#     # Adjust contrast
#     enhanced_image = contrast(enhanced_image, beta=contrast_factor)

#     # Adjust vibrance
#     enhanced_image = adjust_vibrance(enhanced_image, amount=vibrance_factor)

#     # Adjust sharpness
#     enhanced_image = adjust_sharpness(enhanced_image, sigma=sharpness_factor)

#     return enhanced_image

def enhance_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply adaptive histogram equalization for enhancing contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    return enhanced_image

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def adjust_contrast(image, beta=1.0):
    img = convert_from_cv2_to_image(image)
    enhancer = ImageEnhance.Contrast(img)
    im_output = enhancer.enhance(beta)
    im_output.save('/home/sasi/Documents/pillowimg/temp_image.jpg')
    pil_image = Image.open('/home/sasi/Documents/pillowimg/temp_image.jpg')
    cv_image = convert_from_image_to_cv2(pil_image)
    return cv_image

def adjust_vibrance(image, amount=1.0):
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the saturation channel
    saturation = hsv[:, :, 1]

    # Calculate the average saturation
    avg_saturation = np.mean(saturation)

    # Calculate the target saturation
    target_saturation = avg_saturation * amount

    # Adjust the saturation channel
    hsv[:, :, 1] = np.clip(saturation * (target_saturation / avg_saturation), 0, 255)

    # Convert the image back to BGR
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image

def adjust_sharpness(image, sigma=1):
    ksize = 9  # Ensure it's odd

    # Adjust sharpness using GaussianBlur
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp

def adjust_exposure(image, gamma=1.0):
    # Adjust exposure using cv2.LUT gamma[0.5 to 2]
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted_image = cv2.LUT(image, table)
    return adjusted_image

def adjust_warmth(image, temperature=0):
    # Define the warming and cooling LUTs
    warming_lut = np.array([min(i + temperature, 255) for i in range(256)], dtype=np.uint8)
    cooling_lut = np.array([max(i - temperature, 0) for i in range(256)], dtype=np.uint8)

    # Split the image into channels
    b, g, r = cv2.split(image)

    # Apply the warming or cooling LUT to the red channel
    r_adjusted = cv2.LUT(r, warming_lut) if temperature > 0 else cv2.LUT(r, cooling_lut)

    # Merge the channels back together
    adjusted_image = cv2.merge([b, g, r_adjusted])

    return adjusted_image

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# def display_identified_parameters(image_path):
#     # Read the image
#     original_image = cv2.imread(image_path)

#     # Identify parameters for the image
#     brightness, contrast, vibrance, sharpness = identify_parameters(original_image)

#     # Display identified parameters
#     print(f"Image: {image_path}")
#     print(f"Identified Parameters:")
#     print(f"  - Brightness: {brightness}")
#     print(f"  - Contrast: {contrast}")
#     print(f"  - Vibrance: {vibrance}")
#     print(f"  - Sharpness: {sharpness}")
#     print("\n")



# def preprocess_for_text_extraction(image_path):
#     # Read the image
#     original_image = cv2.imread(image_path)
#     display_identified_parameters(image_path)

#     # Identify parameters for the image
#     brightness, contrast, vibrance, sharpness = identify_parameters(original_image)

#     # Adjust parameters based on identified values
#     adjusted_image = adjust_parameters(original_image, brightness, contrast, vibrance, sharpness)

#     # Display the original and preprocessed images (for visualization purposes)
#     cv2.imshow('Original Image', original_image)
#     cv2.imshow('Preprocessed Image', adjusted_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return adjusted_image

# # Example usage for multiple images
# image_paths = ['/home/sasi/Documents/pyphotoshop-main/input/cast.lang.exp0.png', 
#     '/home/sasi/Documents/pyphotoshop-main/input/cast.lang.exp11.png', 
#     '/home/sasi/Documents/pyphotoshop-main/input/cast.lang.exp12.png']

# for image_path in image_paths:
#     preprocess_for_text_extraction(image_path)




def estimate_noise(image, neighborhood_size=3):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the local standard deviation using a neighborhood
    noise_map = cv2.boxFilter(gray_image, -1, (neighborhood_size, neighborhood_size), normalize=False)

    # Normalize the noise map to the range [0, 255]
    cv2.normalize(noise_map, noise_map, 0, 255, cv2.NORM_MINMAX)

    # Calculate the average noise level
    average_noise = np.mean(noise_map)

    # Return the average noise level
    return average_noise

def estimate_image_parameters(image, file_info, **kwargs):
    # Convert the image to LAB color space for better analysis
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Calculate mean and standard deviation for LAB channels
    mean_lab, std_lab = cv2.meanStdDev(lab_image)

    # Calculate image sharpness using Laplacian operator
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = laplacian.var()

    # Calculate image temperature as a ratio of the A channel mean
    temperature = float(mean_lab[1] / 128.0)
    print(f"mean lab\n{mean_lab}\n")
    print(f"std_lab\n{std_lab}\n")
    cv2.imshow('lab_image', lab_image)
    for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab_image)):
	    cv2.imshow(name, chan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Estimate parameters
    brightness = mean_lab[0][0]
    contrast = std_lab[0][0]
    vibrance = (std_lab[1][0] + std_lab[2][0]) / 2
    exposure = np.mean(image)

    # Estimate noise
    noise_level = estimate_noise(image)

    parameters = {
        "Serial No": kwargs.get("serial_no"),
        "File Path": file_info['path'],
        "File Name": file_info['name'],
        "Brightness": brightness,
        "Contrast": contrast,
        "Vibrance": vibrance,
        "Sharpness": sharpness,
        "Temperature": temperature,
        "Exposure": exposure,
        "Noise Level": noise_level
    }

    return parameters

def auto_adjust_parameters(brightness, contrast, vibrance):
    # Dynamic adjustment logic for brightness
    if brightness < 110:
        brightness_factor = 1.0 + (128 - brightness) / 128.0
    else:
        brightness_factor = 1.0 - (brightness - 128) / 128.0

    # Dynamic adjustment logic for contrast
    if contrast < 80:
        contrast_factor = 1.0 + (128 - contrast) / 128.0
    else:
        contrast_factor = 1.0 - (contrast - 128) / 128.0

    # Dynamic adjustment logic for vibrance
    if vibrance < 7:
        vibrance_factor = 1.0 + (128 - vibrance) / 128.0
    else:
        vibrance_factor = 1.0 - (vibrance - 128) / 128.0

    return brightness_factor, contrast_factor, vibrance_factor

# def auto_adjust_parameters(brightness, contrast, vibrance, sharpness, temperature, exposure):
#     # #Define adjustment factors
#     # brightness_factor = calculate_factor(brightness, 0, 255)
#     # contrast_factor = calculate_factor(contrast, 0, 255)
#     # vibrance_factor = calculate_factor(vibrance, 0, 255)
#     # sharpness_factor = calculate_factor(sharpness, 0, 1000)
#     # temperature_factor = calculate_factor(temperature, 0, 255)
#     # exposure_factor = calculate_factor(exposure, 0, 255)
#     # # noise_level_factor = calculate_factor(noise_level, 0, 255)
#     # if brightness < 95:
#     #     brightness_factor = 1.0 + (128 - brightness) / 128.0
#     # else:
#     #     brightness_factor = 1.0 - (brightness - 128) / 128.0

#     # # # Dynamic adjustment logic for contrast
#     # if contrast < 59:
#     #     contrast_factor = 1.0 + (128 - contrast) / 128.0
#     # else:
#     #     contrast_factor = 1.0 - (contrast - 128) / 128.0

#     # # # Dynamic adjustment logic for vibrance
#     # if vibrance < 4:
#     #     vibrance_factor = 1.0 + (128 - vibrance) / 128.0
#     # else:
#     #     vibrance_factor = 1.0 - (vibrance - 128) / 128.0

#     # # # Dynamic adjustment logic for sharpness
#     # if sharpness < 550:
#     #     sharpness_factor = 1.0 + (400 - sharpness) / 400.0
#     # else:
#     #     sharpness_factor = 1.0 - (sharpness - 400) / 400.0

#     # # # Dynamic adjustment logic for temperature
#     # if exposure < 91:
#     #     exposure_factor = 1.0 + (128 - exposure) / 128.0
#     # else:
#     #     exposure_factor = 1.0 - (exposure - 128) / 128.0

#     # # Dynamic adjustment logic for exposure
#     # temperature_factor = 1.0 + (128 - temperature) / 128.0

#     # # Dynamic adjustment logic for brightness
#     brightness_factor = 1.0 + (128 - brightness) / 128.0 if brightness < 95 else 1.0 - (brightness - 128) / 128.0

#     # # Dynamic adjustment logic for contrast
#     contrast_factor = 1.0 + (128 - contrast) / 128.0 if contrast < 59 else 1.0 - (contrast - 128) / 128.0

#     # # Dynamic adjustment logic for vibrance
#     vibrance_factor = 1.0 + (128 - vibrance) / 128.0 if vibrance < 4 else 1.0 - (vibrance - 128) / 128.0

#     # # Dynamic adjustment logic for sharpness
#     sharpness_factor = 1.0 + (400 - sharpness) / 400.0 if sharpness < 550 else 1.0 - (sharpness - 400) / 400.0

#     # # Dynamic adjustment logic for temperature
#     temperature_factor = 1.0 + (128 - temperature) / 128.0 if temperature < 91 else 1.0 - (temperature - 128) / 128.0

#     # # Dynamic adjustment logic for exposure
#     exposure_factor = 1.0 + (128 - exposure) / 128.0

#     return brightness_factor, contrast_factor, vibrance_factor, sharpness_factor, temperature_factor, exposure_factor

def calculate_factor(value, min_value, max_value):
    # Normalize the value to the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)

    # Fine-tune the adjustment based on different conditions
    if normalized_value < 0.5:
        factor = 1.0 + normalized_value
    elif normalized_value > 0.5:
        factor = 1.0 - (1.0 - normalized_value)
    else:
        factor = 1.0

    return factor

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def save_parameters_to_excel(data, file_path):
    # Create a DataFrame from the parameters
    df = pd.DataFrame(data)

    parameter_columns = df.columns[3:]  # Assuming the parameters start from the 4th column

    # Identify the minimum and maximum values for each parameter
    min_values = df[parameter_columns].min()
    max_values = df[parameter_columns].max()

    min_row = pd.DataFrame({'Serial No': 'Min Values', 'File Path': '', 'File Name': '', **min_values}, index=[0])
    df = pd.concat([df, min_row], ignore_index=True)

    # Append a new row with maximum values
    max_row = pd.DataFrame({'Serial No': 'Max Values', 'File Path': '', 'File Name': '', **max_values}, index=[0])
    df = pd.concat([df, max_row], ignore_index=True)
    # Save the DataFrame to an Excel file
    df.to_excel(file_path, index=False)
    print(f"Parameters saved to {file_path}")

def adjust_image(image, brightness, contrast, vibrance): # sharpness=None, temperature=None, exposure=None):
    original_image = image
    # Adjust brightness
    # brightness_factor = calculate_brightness_factor(brightness)
    bimage = change_brightness(image, value=brightness)

    # Adjust contrast
    # contrast_factor = calculate_contrast_factor(contrast)
    cimage = adjust_contrast(bimage, beta=contrast)

    # Adjust vibrance
    # vibrance_factor = calculate_vibrance_factor(vibrance)
    vimage = adjust_vibrance(cimage, amount=vibrance)
    # b = image.imread(vimage)

    # gimage = rgb2gray(np.array(vimage))
    # img_train = np.expand_dims(gimage, axis=-1)
    # gimage = img_train.astype('float32') / 255

    # simage = adjust_sharpness(vimage, sigma=sharpness)

    # timage = adjust_exposure(simage, gamma=exposure)

    # eimage = adjust_warmth(timage, temperature=temperature)
    enhanced_image = enhance_image(vimage)

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1.4)

    # Perform thresholding to create a binary image with different parameters
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    # Invert the binary image
    inverted_image = cv2.bitwise_not(thresh)

    # Apply morphological transformations with different kernel sizes and iterations
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.erode(thresh, kernel, iterations=2)
    morph_image = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Experiment with different parameters for denoising
    noiseless_image_bw = cv2.fastNlMeansDenoising(morph_image, None, 30, 10, 21)

    # Resize all images to a common size
    common_size = (original_image.shape[1], original_image.shape[0])  # Width, height
    enhanced_image_resized = cv2.resize(enhanced_image, common_size)
    inverted_image_resized = cv2.resize(inverted_image, common_size)
    noiseless_image_bw_resized = cv2.resize(noiseless_image_bw, common_size)

    # Convert to RGB for compatibility with concatenate
    enhanced_image_resized = cv2.cvtColor(enhanced_image_resized, cv2.COLOR_GRAY2RGB)
    inverted_image_resized = cv2.cvtColor(inverted_image_resized, cv2.COLOR_BGR2RGB)
    noiseless_image_bw_resized = cv2.cvtColor(noiseless_image_bw_resized, cv2.COLOR_BGR2RGB)

    # Create a composite image for better clarity
    composite_image = np.concatenate((original_image, enhanced_image_resized, inverted_image_resized, noiseless_image_bw_resized),
                                     axis=1)
    
    # cv2.imshow('Original Image', original_image)
    # cv2.imshow('Auto-Adjusted Image', image)
    # cv2.imshow( 'brightness Image', bimage)
    # cv2.imshow('contrast Image', cimage )
    # cv2.imshow('vibrance Image', vimage)
    # cv2.imshow('enchanced Image', enhanced_image)
    # cv2.imshow('blurred Image', blurred_image)
    # cv2.imshow('thresh Image', thresh)
    # cv2.imshow('inverted Image', inverted_image)
    # cv2.imshow('enhanced_image_resizedImage', inverted_image_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return vimage


def display_image_and_parameters(image, brightness, contrast, vibrance):#, sharpness=None, temperature=None, exposure=None):
    adjusted_image = adjust_image(image, brightness, contrast, vibrance)#, sharpness, temperature, exposure)

    # cv2.imshow('Original Image', image)
    # cv2.imshow('Auto-Adjusted Image', adjusted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return adjusted_image

# List of image paths
# image_paths = ['/home/sasi/Documents/pyphotoshop-main/input/cast.lang.exp0.png',
#                 '/home/sasi/Documents/pyphotoshop-main/input/cast.lang.exp11.png',
#                 '/home/sasi/Documents/pyphotoshop-main/input/cast.lang.exp12.png',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp16.jpeg',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp18.jpeg',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp19.jpeg',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp20.jpeg',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp28.jpeg',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp29.jpeg',
#                 '/home/sasi/intern/tesseract/casting_image/cast.lang.exp31.jpeg',
#                 '/home/sasi/Documents/pyphotoshop-main/input/cameralw.jpeg',
#                 '/home/sasi/Documents/pyphotoshop-main/input/cameralb.jpeg',
#                 '/home/sasi/Documents/pyphotoshop-main/input/cameranb.jpeg',
#                 '/home/sasi/Documents/pyphotoshop-main/input/cameranw.jpeg']


output_excel_path = '/home/sasi/Documents/output.xlsx'
output_excel_path1 = '/home/sasi/Documents/adjoutput.xlsx'

# Initialize data list to store parameters for all images
all_parameters = []
adj_parameters = []
# Process each image
input_folder = '/home/sasi/Pictures/Camera Roll'
output_folder = '/home/sasi/Music/textfolder/adj2'

# Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
idx = 0
# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        idx += 1
        # Build the full path to the input image
        image_path = os.path.join(input_folder, filename)

        # Preprocess the image
        # preprocessed_img = preprocess_for_text_extraction(image_path)

        # Optional: Extract text from the preprocessed image
        

    # for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)

        # Extract file information
        file_info = {'path': image_path, 'name': image_path.split("/")[-1]}

        # Estimate image parameters
        parameters = estimate_image_parameters(image, file_info, serial_no=idx)

        # Append parameters to the list
        all_parameters.append(parameters)

        brightness_value = parameters['Brightness']
        contrast_value = parameters['Contrast']
        vibrance_value = parameters['Vibrance']
        sharpness_value = parameters['Sharpness']
        temperature_value = parameters['Temperature']
        exposure_value = parameters['Exposure']
        noise_value = parameters['Noise Level']
        brightness_factor, contrast_factor, vibrance_factor = auto_adjust_parameters(brightness_value, 
                                                                contrast_value, 
                                                                vibrance_value)
        # brightness, contrast, vibrance, sharpness, temperature, exposure = auto_adjust_parameters(brightness_value, 
        #                                                                                             contrast_value, 
        #                                                                                             vibrance_value, 
        #                                                                                             sharpness_value, 
        #                                                                                             temperature_value, 
        #                                                                                             exposure_value)
        adj = display_image_and_parameters(image, brightness_factor, contrast_factor, vibrance_factor)#, sharpness, temperature, exposure)
        adjparameters = estimate_image_parameters(adj, file_info, serial_no=idx)
        adj_parameters.append(adjparameters)
        grey_img = cv2.cvtColor(adj, cv2.COLOR_BGR2GRAY)

        # Apply a colormap for thermal effect (e.g., COLORMAP_JET)
        thermal_image = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)
        # Save the preprocessed image to the output folder cv2.cvtColor(adj, cv2.COLOR_HSV2BGR)
        output_path = os.path.join(output_folder, filename)
        # cv2.imshow("output image", cv2.cvtColor(adj, cv2.COLOR_HSV2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(output_path,adj)
        text = pt.image_to_string(adj, lang='eng', config=custom_config)
        print(f'Text from {filename}:\n{text}')

# Save parameters to an Excel file
save_parameters_to_excel(all_parameters, output_excel_path)
save_parameters_to_excel(adj_parameters, output_excel_path1)