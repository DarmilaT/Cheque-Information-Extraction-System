# Import necessary libraries
import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

# Streamlit App
st.title("Cheque MICR Code Extraction")
st.write("Upload a cheque image to extract the cheque details from its MICR code.")

# File uploader for cheque image
uploaded_file = st.file_uploader("Upload Cheque Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Cheque Image", use_container_width =True)
    
    # Open the image and process it
    image = Image.open(uploaded_file)
    cheque_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Extract the MICR region (bottom 17% of the image)
    h, w = cheque_img.shape[:2]
    delta = int(h - (h * 0.17))
    micr_img = cheque_img[delta:h, 0:w]
    
    # Convert MICR region to grayscale and display it
    micr_img_display = cv2.cvtColor(micr_img, cv2.COLOR_BGR2RGB)
    st.image(micr_img_display, caption="Extracted MICR Region", use_container_width =True)
    
    # Extract text from MICR region using Tesseract
    micr_string = pytesseract.image_to_string(micr_img, lang='mcr')
    st.write("**Extracted MICR String:**", micr_string)
    
    # Extract details from the MICR string
    details = extract_cheque_details(micr_string)
    if "Error" in details:
        st.error(details["Error"])
    else:
        st.success("Cheque Details Extracted Successfully!")
        st.write("**Cheque Number:**", details["Cheque Number"])
        st.write("**City Code:**", details["City Code"])
        st.write("**Bank Code:**", details["Bank Code"])
        st.write("**Branch Code:**", details["Branch Code"])