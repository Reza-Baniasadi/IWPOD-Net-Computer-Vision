import numpy as np
import cv2
from src.keras_utils import detect_lp
from src.utils import im2single, nms_darkflow, adjust_pts
from dr_utils import draw_losangle


# ---------------- Vehicle Detection -----------------
def detect_cars(yolo_model, image):
    """
    Detect cars and buses in an image using YOLO
    """
    detections = yolo_model.return_predict(image)
    cars = [det for det in detections if det['label'] in ['car']]
    return cars


# ---------------- License Plate Extraction -----------------
def extract_license_plates(cars, image, wpod_model, lp_threshold):
    """
    Crop license plates from detected vehicles
    """
    plates, plate_images = [], []

    if len(cars) == 0:
        cars = [{'label': 'car', 'confidence': 1,
                 'topleft': {'x': 1, 'y': 1},
                 'bottomright': {'x': image.shape[1], 'y': image.shape[0]}}]

    for car in cars:
        x1 = car['topleft']['x']
        y1 = car['topleft']['y']
        x2 = car['bottomright']['x']
        y2 = car['bottomright']['y']

        vehicle_img = image[y1:y2, x1:x2]

        WPOD_RES = 416
        ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
        side = int(ratio * 288)
        bound_dim = min(side + (side % (2 ** 4)), WPOD_RES)

        Llp, LlpImgs, _ = detect_lp(
            wpod_model, im2single(vehicle_img), bound_dim, 2 ** 4, (240, 80), lp_threshold
        )

        plates.append(Llp)
        plate_images.append(LlpImgs)

    return plates, plate_images, cars


# ---------------- OCR Recognition -----------------
def recognize_plates(ocr_model, cars, image, plate_list, plate_imgs_list):
    recognized_texts = []
    processed_imgs = []
    plate_counter = 0

    for lp_imgs in plate_imgs_list:
        if len(lp_imgs):
            plate = plate_list[plate_counter]
            lp_img = lp_imgs[0]
            lp_img = cv2.cvtColor(lp_img, cv2.COLOR_BGR2GRAY)
            lp_img = cv2.cvtColor(lp_img, cv2.COLOR_GRAY2BGR)

            points = adjust_pts(plate[0].pts, cars[plate_counter])
            draw_losangle(image, points, (0, 0, 255), 3)

            ocr_results = ocr_model.return_predict(lp_img * 255.)
            ocr_results = nms_darkflow(ocr_results)
            ocr_results.sort(key=lambda x: x['topleft']['x'])

            lp_str = ''.join([r['label'] for r in ocr_results])
            recognized_texts.append(lp_str)
            processed_imgs.append(lp_img)
            plate_counter += 1

    return recognized_texts, processed_imgs


# ---------------- Save Results -----------------
def save_plate_outputs(recognized_texts, processed_imgs, output_dir, root_name, save_txt=True, save_images=True):
    for i, (lp_str, lp_img) in enumerate(zip(recognized_texts, processed_imgs)):
        if save_txt:
            with open(f"{output_dir}{root_name}_str_{i+1}.txt", 'w') as f:
                f.write(lp_str + '\n')
        if save_images:
            cv2.imwrite(f"{output_dir}{root_name}_plate_{i+1}_ocr.png", lp_img * 255.)


# ---------------- Complete Pipeline -----------------
def process_image_pipeline(yolo_model, image, wpod_model, lp_threshold, ocr_model, output_dir, root_name):
    cars = detect_cars(yolo_model, image)
    plates, plate_imgs, cars = extract_license_plates(cars, image, wpod_model, lp_threshold)
    recognized_texts, processed_imgs = recognize_plates(ocr_model, cars, image, plates, plate_imgs)
    save_plate_outputs(recognized_texts, processed_imgs, output_dir, root_name)
    return recognized_texts, processed_imgs


# ---------------- Plate Character Formatting -----------------
def format_plate_mercosul(lp_str):
    out = list(lp_str)
    if len(lp_str) == 7:
        for i in range(0, 3):
            out[i] = replace_with_letter(out[i])
        out[3] = replace_with_digit(out[3])
        out[4] = replace_with_letter(out[4])
        for i in range(5, 7):
            out[i] = replace_with_digit(out[i])
    return ''.join(out)


def format_plate_brazilian(lp_str):
    out = list(lp_str)
    if len(lp_str) == 7:
        for i in range(0, 3):
            out[i] = replace_with_letter(out[i])
        for i in range(3, 7):
            out[i] = replace_with_digit(out[i])
    return ''.join(out)


def format_plate_chinese(lp_str):
    out = list(lp_str)
    if len(out) == 7:
        out = out[1:]
    if len(out) == 6:
        out[0] = replace_with_letter(out[0])
    return ''.join(out)


def impose_letters(lp_str):
    return ''.join([replace_with_letter(c) for c in lp_str])


def replace_with_letter(c):
    digits = '0123456789'
    letters = 'OIZBASETBS'
    return letters[digits.index(c)] if c.isdigit() else c


def replace_with_digit(c):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digits = '48006661113191080651011017'
    return digits[letters.index(c)] if c.isalpha() else c


# ---------------- Plate Classification -----------------
def classify_plate_type(image, ocr_boxes):
    """
    Determine the type of license plate (Mercosul / Iranian) based on intensity analysis
    """
    offset = 4
    min_y, max_y, min_x, max_x, heights = [], [], [], [], []

    for box in ocr_boxes:
        min_y.append(box['topleft']['y'])
        max_y.append(box['bottomright']['y'])
        min_x.append(box['topleft']['x'])
        max_x.append(box['bottomright']['x'])
        heights.append(max_y[-1] - min_y[-1])

    min_y_val = max(offset, min(min_y))
    min_x_val = max(offset, min(min_x))
    max_x_val = min(239 - offset, max(max_x))
    avg_height = sum(heights) / len(heights)

    channel = 2
    img_channel = image[:, :, channel].copy()
    u_height = int(max(1, min(min_y_val, avg_height / 2)))
    l_height = int(max(1, min(u_height / 4, 79 - max(max_y))))

    up_intensity = np.median(img_channel[(min_y_val - u_height):min_y_val, min_x_val:max_x_val])
    middle_intensity = np.median(img_channel[min_y_val:max(max_y), min_x_val:max_x_val])
    median_intensity = np.median(img_channel[:, min_x_val:max_x_val])

    if up_intensity < 0.6 * middle_intensity:
        if median_intensity > 1.4 * up_intensity:
            return 'Mercosul'
    else:
        return 'Iranian'
