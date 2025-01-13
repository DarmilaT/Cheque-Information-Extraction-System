import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours
from tensorflow.keras.models import load_model
from easyocr import Reader
from PIL import Image
import io
import re


# Load pre-trained model
model = load_model(r"D:\7th Semester\Computer Vision\Mini-Project\new\handwriting_model.h5")  # Replace with the correct path to your model


# Initialize EasyOCR reader
reader = Reader(['en'], gpu=True)


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def cropping(image):
    # Use contours to crop out regions of interest (ROI)
    return image[100:510,200:2200]


# def cleanup_text(text):
#     return text.strip().replace("\n", " ").replace("\r", "")

# Function to clean text and detect valid payee names
def cleanup_text(text):
    # Remove any unwanted characters (e.g., punctuation, extra spaces)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)  # Keep letters, numbers, and underscores
    cleaned_text = cleaned_text.strip().lower()  # Clean and convert to lowercase
    return cleaned_text

def process_and_ocr(image, model, reader):
    """
    Processes the input image, performs OCR using the handwriting model and EasyOCR,
    and returns the detected results.
    """
    # Step 1: Preprocessing
    gray = get_grayscale(image)
    denoised = remove_noise(gray)
    thresh = thresholding(denoised)
    cropped = cropping(thresh)

    # Step 2: Contour Detection and Handwriting OCR
    edged = cv2.Canny(cropped, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    chars = []
    boxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)

        # Filter bounding boxes based on size, aspect ratio, and area
        if 50 <= area <= 5000 and 0.2 <= aspect_ratio <= 1.5 and (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y:y + h, x:x + w]
            roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = roi_thresh.shape

            # Resize the ROI to maintain aspect ratio
            if tW > tH:
                roi_thresh = imutils.resize(roi_thresh, width=32)
            else:
                roi_thresh = imutils.resize(roi_thresh, height=32)

            (tH, tW) = roi_thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            padded = cv2.copyMakeBorder(roi_thresh, top=dY, bottom=dY, left=dX, right=dX,
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            chars.append((padded, (x, y, w, h)))

    # amounts = []
    payees = []

    if chars:
        boxes = [b[1] for b in chars]
        chars = np.array([c[0] for c in chars], dtype="float32")
        preds = model.predict(chars)
        labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        labelNames = [l for l in labelNames]

        # for (pred, (x, y, w, h)) in zip(preds, boxes):
        #     i = np.argmax(pred)
        #     prob = pred[i]
        #     label = labelNames[i]
        #     if prob > 0.5:  # Confidence threshold
        #         cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         cv2.putText(cropped, label, (x - 10, y - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        #         amounts.append(label)  # Assuming amount is detected in this section

    # EasyOCR for printed text recognition
    results = reader.readtext(cropped)
    for (bbox, text, prob) in results:
        cleaned_text = cleanup_text(text)
        if prob > 0.5:  # Confidence threshold
            if re.match(r'^[a-zA-Z_]+$', text):  # Match alphanumeric text with underscores
                payees.append(cleaned_text) # Payee detection
                cv2.rectangle(cropped, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (255, 0, 0), 2)
                cv2.putText(cropped, text, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                break

    return payees, cropped


# Streamlit interface
st.title("Cheque Information Extraction System")
st.markdown("Upload an image of a cheque for OCR detection.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    img = Image.open(uploaded_file)
    img = np.array(img)
    st.image(img, caption='Uploaded Cheque', use_container_width=True)

    # Process the image
    payees, processed_img = process_and_ocr(img, model, reader)

    # Display the OCR results
    # st.subheader("Detected Amounts:")
    # st.write(amounts)

    st.subheader("Detected Payee Names:")
    st.write(payees)

    # Display the processed image
    st.subheader("Processed Image with OCR Results:")
    st.image(processed_img, channels="RGB")
