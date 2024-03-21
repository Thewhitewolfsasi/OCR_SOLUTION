import cv2
import os
import numpy as np
import pandas as pd
import pytesseract as pt
from PIL import Image, ImageEnhance
import matplotlib.image as image

# Process each image
input_folder = '.'
output_folder = '.'

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



output_excel_path = './output.xlsx'
output_excel_path1 = './adjoutput.xlsx'

# Initialize data list to store parameters for all images
all_parameters = []
adj_parameters = []


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

# Save parameters to an Excel file
save_parameters_to_excel(all_parameters, output_excel_path)
save_parameters_to_excel(adj_parameters, output_excel_path1)
