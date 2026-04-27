import cv2
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from os.path import join


def show_tensor_img(tensor):
    plt.imshow(tensor.permute(1, 2, 0))
    plt.show()
    
    
def get_value(string):
        try:
            return float(string)
        except ValueError:
            return string


def show_random_tensors(dataset, num_images=16):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    indices = random.sample(range(len(dataset)), num_images)
    for ax, idx in zip(axes.flatten(), indices):
        tensor = dataset[idx]['image']
        ax.imshow(tensor.permute(1, 2, 0))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def batch_logging_tensorboard(writer, metrics, epoch, prefix='Eval/'):
    for key, value in metrics.items():
        writer.add_scalar(prefix + key, value, epoch)


def generate_confusion_matrix(acc_labels, acc_predictions, writer, epoch, res_path):
    cm = confusion_matrix(acc_labels, acc_predictions, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    disp.plot(cmap='Blues')

    plt.savefig(join(res_path, 'confusion_matrix.png'))
    if writer is not None:
        cm_img = plt.imread(join(res_path, 'confusion_matrix.png'))
        transform = transforms.ToTensor()
        cm_img_tensor = transform(cm_img)
        writer.add_image('Confusion Matrix Normalized', cm_img_tensor, epoch)


def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    return accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())


def log_hyperparameters(writer, config, eval_metrics):
    hyperparameters = {key: get_value(config[key]) for key in config.keys()}
    results = {f'hparam/{key}': value for key, value in eval_metrics.items()}
    writer.add_hparams(hyperparameters, results)

    
def create_results_excel(name="paxos_results.xlsx"):
    df = pd.DataFrame()
    df.to_excel(name, index=False)


def write_results_to_overview(timestamp, dataset_name, config, metrics, excel_path="paxos_results.xlsx"):
    df = pd.read_excel(excel_path)
    hyperparameters = {key: get_value(config[key]) for key in config.keys()}
    
    new_index = 0 if df.empty else df.index[-1] + 1
    df.at[new_index, "timestamp"] = timestamp
    df.at[new_index, "dataset"] = dataset_name
    for key, value in hyperparameters.items():
        df.at[new_index, key] = value
    for key, value in metrics.items():
        df.at[new_index, key] = value
    
    df.to_excel(excel_path, index=False)


def adjust_hough_parameters(frame, base_param1=130, base_param2=50):
    """
    Adjust Hough Circle Transform parameters based on frame brightness and sharpness.
    """
    h, w = frame.shape[:2]
    center_patch = frame[h//5:h*3//5, w//5:w*3//5]
    gray_frame = cv2.cvtColor(center_patch, cv2.COLOR_BGR2GRAY)
    brightness = gray_frame.mean()
    sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

    if brightness < 100:  # Dim lighting
        param1 = max(base_param1 - 50, 50)
    elif brightness > 150:  # Bright lighting
        param1 = min(base_param1 + 50, 250)
    else:  # Normal lighting
        param1 = base_param1

    if sharpness < 50:  # Low sharpness
        param2 = max(base_param2 - 10, 20)
    elif sharpness > 150:  # High sharpness
        param2 = min(base_param2 + 10, 100)
    else:  # Normal sharpness
        param2 = base_param2
    # print(f"Brightness: {brightness}, Sharpness: {sharpness}, Params: {param1}, {param2}")
    return param1, param2


def get_retina_mask(img: np.ndarray, radius_reduction: int = 20, min_radius = 300, speedup = 4):
    h, w = img.shape[0] // speedup, img.shape[1] // speedup
    
    red_channel = img[:, :, 2]
    green_channel = img[:, :, 1]
    gray = cv2.addWeighted(red_channel, 0.4, green_channel, 0.6, 0) ## Combine channels using weights
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dsize=(w, h))
    img_blur = cv2.medianBlur(gray, 5)
    img_blur = img_blur //32 * 32 + 32 // 2
    
    min_radius = min_radius // speedup
    param1, param2 = adjust_hough_parameters(img)
    circle = None
    
    # detect lens
    detected_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1.5, gray.shape[0] / 8, minRadius=min_radius, maxRadius=min_radius+(300 // speedup), param1=param1, param2=param2) # 75, 50
    
    circle = None
    if detected_circles is not None and len(detected_circles) > 0:
        detected_circles = np.round(detected_circles[0, :]).astype("int")
        detected_circles = [(x, y, r) for (x, y, r) in detected_circles if
                         gray.shape[0] / 3 < y < gray.shape[0] / 3 * 2 and gray.shape[1] / 3 < x < gray.shape[1] / 3 * 2]
        if len(detected_circles) > 2:
            x_avg = int(np.mean([c[0] for c in detected_circles[:3]])) 
            y_avg = int(np.mean([c[1] for c in detected_circles[:3]]))
            r_avg = int(np.mean([c[2] for c in detected_circles[:3]]))
            circle = (x_avg, y_avg, r_avg)
        elif len(detected_circles) > 0:
            circle = detected_circles[0]
            
        #circle = sorted(detected_circles, key=lambda xyr: xyr[2], reverse=False)[0] if len(detected_circles) > 0 else None

    if circle is not None:
        (x, y, r) = circle
        r -= radius_reduction
        return (x*speedup, y*speedup, r*speedup)
    else:
        print('UTIL> No mask found')
        return None


def crop_to_circle(img: np.ndarray, circle) -> np.ndarray:
    x, y, r = circle
    if x - r < 0 or y - r < 0 or x + r > img.shape[1] or y + r > img.shape[0]:
        x = img.shape[1] // 2
        y = img.shape[0] // 2
    return img[y - r:y + r, x - r:x + r, :]


def calculate_average_circle(average_circle, circle, step, memory=90):
    if circle is None:
        return average_circle
    step = min(step, memory)
    
    for i in range(3):
        average_circle[i] = int((circle[i] + step * average_circle[i]) / (step + 1))

    return average_circle


def center_image(input_image, output_size):
    """
    Centers an image into a new canvas of the specified size or center-crops it if larger.
    Args:
        input_image (numpy.ndarray): The input image as a NumPy array (OpenCV format).
        output_size (tuple): Desired output size as (width, height).
    Returns:
        numpy.ndarray: The resulting centered or cropped image.
    """
    input_height, input_width = input_image.shape[:2]
    output_width, output_height = output_size
    
    if input_width > output_width or input_height > output_height: # If the input image is larger, crop it
        left = max((input_width - output_width) // 2, 0)
        top = max((input_height - output_height) // 2, 0)
        right = left + output_width
        bottom = top + output_height
        cropped_image = input_image[top:bottom, left:right]
        return cropped_image

    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)  # Otherwise, center it on a new canvas
    offset_x = (output_width - input_width) // 2
    offset_y = (output_height - input_height) // 2
    canvas[offset_y:offset_y + input_height, offset_x:offset_x + input_width] = input_image
    return canvas
