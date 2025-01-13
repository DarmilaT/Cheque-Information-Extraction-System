import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from tensorflow.keras.models import load_model
from easyocr import Reader
import imutils
from imutils.contours import sort_contours

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load pre-trained model
model = load_model(r"D:\7th Semester\Computer Vision\Mini-Project\new\handwriting_model.h5")  # Replace with the correct path to your model

# Initialize EasyOCR reader
reader = Reader(['en'], gpu=True)

# Function to extract cheque details
def extract_cheque_details(micr_string):
    """
    Extract cheque number, city code, bank code, and branch code from MICR string.
    """
    # Preprocess the string to retain only numeric characters
    cleaned_string = re.sub(r'\D', '', micr_string)
    
    # Ensure the cleaned string has sufficient length
    if len(cleaned_string) < 15:  # Minimum length for all codes
        return {"Error": "Invalid MICR string length"}
    
    # Extract details based on position
    cheque_number = cleaned_string[:6]
    city_code = cleaned_string[6:9]
    bank_code = cleaned_string[9:12]
    branch_code = cleaned_string[12:15]
    
    # Return extracted details
    return {
        "Cheque Number": cheque_number,
        "City Code": city_code,
        "Bank Code": bank_code,
        "Branch Code": branch_code
    }

# Preprocessing functions for handwriting OCR
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def cropping(image):
    # Use contours to crop out regions of interest (ROI)
    return image[100:510, 200:2200]

def cleanup_text(text):
    # Remove unwanted characters and clean up text
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)  # Keep letters, numbers, and underscores
    cleaned_text = cleaned_text.strip().lower()  # Clean and convert to lowercase
    return cleaned_text

# Process image and OCR
def process_and_ocr(image, model, reader):
    """
    Processes the input image, performs OCR using the handwriting model and EasyOCR,
    and returns the detected results.
    """
    # Step 1: Preprocessing for handwriting detection
    gray = get_grayscale(image)
    denoised = remove_noise(gray)
    thresh = thresholding(denoised)
    cropped = cropping(thresh)

    # Step 2: EasyOCR for printed text recognition (Payee Name detection)
    results = reader.readtext(cropped)
    payees = []

    for (bbox, text, prob) in results:
        cleaned_text = cleanup_text(text)
        if prob > 0.5:  # Confidence threshold
            if re.match(r'^[a-zA-Z_]+$', text):  # Match alphanumeric text with underscores
                payees.append(cleaned_text)  # Payee detection
                cv2.rectangle(cropped, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (255, 0, 0), 2)
                cv2.putText(cropped, text, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                break

    return payees, cropped

# Streamlit App
st.title("Cheque Information Extraction System")
st.markdown("Build by Darmila.T and Darshika.P")

st.write("Upload a cheque image to extract the details.")

# File uploader for cheque image
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    img = Image.open(uploaded_file)
    img = np.array(img)
    st.image(img, caption="Uploaded Cheque Image", use_container_width=True)

    # Extract the MICR region (bottom 17% of the image)
    cheque_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = cheque_img.shape[:2]
    delta = int(h - (h * 0.17))
    micr_img = cheque_img[delta:h, 0:w]
    
    # Convert MICR region to grayscale
    micr_img_display = cv2.cvtColor(micr_img, cv2.COLOR_BGR2RGB)
    st.image(micr_img_display, caption="Extracted MICR Region", use_container_width=True)

    # Extract text from MICR region using Tesseract
    micr_string = pytesseract.image_to_string(micr_img, lang='mcr')
    st.write("**Extracted MICR String:**", micr_string)

    # Extract cheque details from the MICR string
    cheque_details = extract_cheque_details(micr_string)
    if "Error" in cheque_details:
        st.error(cheque_details["Error"])
    else:
        st.success("Cheque Details Extracted Successfully!")
        st.write("**Cheque Number:**", cheque_details["Cheque Number"])
        st.write("**City Code:**", cheque_details["City Code"])
        st.write("**Bank Code:**", cheque_details["Bank Code"])
        st.write("**Branch Code:**", cheque_details["Branch Code"])

    # Process the image for OCR
    payees, processed_img = process_and_ocr(img, model, reader)

    # Display the OCR results
    st.subheader("Payee Name:")
    st.write(payees[0])

    # Display the processed image with OCR results
    st.subheader("Processed Image with OCR Results:")
    st.image(processed_img, channels="RGB", use_container_width=True)
