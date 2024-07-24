import cv2
import pytesseract
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pdf2image import convert_from_path
import pymupdf  # PyMuPDF
import difflib
import re
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define dynamic field patterns to exclude from text comparison
dynamic_fields_patterns = [
    r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',  # Match numbers (e.g., amounts)
    r'\b\d{4}-\d{2}-\d{2}\b',               # Match dates (e.g., YYYY-MM-DD)
    r'\b\d{2}/\d{2}/\d{4}\b',               # Match dates (e.g., MM/DD/YYYY)
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', # Match email addresses
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b', # Match IP addresses
    r'\b\w+@\w+\.\w+\b',  # Simplified match for email addresses
    r'\b\d{2}:\d{2}:\d{2}\b'  # Match time (HH:MM:SS)
]

def remove_dynamic_fields(text, patterns):
    """Removes dynamic fields from text based on given patterns."""
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text

def resize_image(image, target_size):
    """Resizes the image to the target size."""
    return cv2.resize(np.array(image), target_size)

def compare_color_statistics(image1, image2):
    """Compares mean values and standard deviations of color channels of two images."""
    def compute_statistics(image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean = np.mean(hsv_image, axis=(0, 1))
        std_dev = np.std(hsv_image, axis=(0, 1))
        return mean, std_dev
    
    target_size = (np.array(image1).shape[1], np.array(image1).shape[0])
    image2_resized = resize_image(image2, target_size)

    mean1, std_dev1 = compute_statistics(np.array(image1))
    mean2, std_dev2 = compute_statistics(image2_resized)
    
    mean_diff = np.linalg.norm(mean1 - mean2)
    std_dev_diff = np.linalg.norm(std_dev1 - std_dev2)
    
    logging.debug(f"Mean value difference: {mean_diff}")
    logging.debug(f"Standard deviation difference: {std_dev_diff}")
    
    return mean_diff, std_dev_diff

def compare_average_colors(image1, image2):
    """Compares the average colors of two images."""
    def compute_average_color(image):
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return avg_color
    
    target_size = (np.array(image1).shape[1], np.array(image1).shape[0])
    image2_resized = resize_image(image2, target_size)

    avg_color1 = compute_average_color(np.array(image1))
    avg_color2 = compute_average_color(image2_resized)
    
    color_diff = np.linalg.norm(avg_color1 - avg_color2)
    
    logging.debug(f"Average color difference: {color_diff}")
    
    return color_diff

def segment_and_compare_color(image1, image2, lower_bound, upper_bound):
    """Segments color areas and compares them between two images."""
    def segment_color(image, lower_bound, upper_bound):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        return mask
    
    target_size = (np.array(image1).shape[1], np.array(image1).shape[0])
    image2_resized = resize_image(image2, target_size)

    mask1 = segment_color(np.array(image1), lower_bound, upper_bound)
    mask2 = segment_color(np.array(image2_resized), lower_bound, upper_bound)
    
    similarity = np.sum(mask1 == mask2) / mask1.size
    logging.debug(f"Segmented area similarity: {similarity}")
    
    return similarity

def compare_color_clusters(image1, image2, k=5):
    """Compares color clusters of two images using K-Means."""
    def get_dominant_colors(image, k):
        pixels = np.float32(image.reshape(-1, 3))
        _, labels, centers = cv2.kmeans(pixels, k, None, 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                                        10, cv2.KMEANS_RANDOM_CENTERS)
        return centers
    
    target_size = (np.array(image1).shape[1], np.array(image1).shape[0])
    image2_resized = resize_image(image2, target_size)

    colors1 = get_dominant_colors(np.array(image1), k)
    colors2 = get_dominant_colors(np.array(image2_resized), k)
    
    color_diff = np.linalg.norm(np.mean(colors1 - colors2, axis=0))
    
    logging.debug(f"Color cluster difference: {color_diff}")
    
    return color_diff

def compare_nd_histograms(image1, image2, bins=[32, 32, 32]):
    """Compares the color histograms of two images in N-D space."""
    def compute_nd_histogram(image, bins):
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    
    hist1 = compute_nd_histogram(image1, bins)
    hist2 = compute_nd_histogram(image2, bins)
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    logging.debug(f"N-D histogram similarity: {similarity}")
    
    return similarity

def get_pdf_metadata(pdf_path):
    """Gets metadata from a PDF file."""
    try:
        pdf_document = pymupdf.open(pdf_path)
        metadata = pdf_document.metadata
        pdf_document.close()
        return metadata
    except Exception as e:
        logging.error(f"Error getting metadata from {pdf_path}: {e}")
        return {}

def compare_metadata(metadata1, metadata2):
    """Compares metadata of two PDF files and returns discrepancies."""
    discrepancies = []
    keys_to_check = ['title', 'author', 'producer']
    for key in keys_to_check:
        if metadata1.get(key) != metadata2.get(key):
            discrepancies.append(f"❌ Metadata discrepancy for key '{key}': '{metadata1.get(key)}' != '{metadata2.get(key)}'")
    return discrepancies

def get_pdf_fonts(pdf_path):
    """Gets fonts from a PDF file."""
    try:
        pdf_document = pymupdf.open(pdf_path)
        fonts = set()
        for page in pdf_document:
            font_dict = page.get_fonts(full=True)
            for font in font_dict:
                fonts.add(font[3])
        pdf_document.close()
        return fonts
    except Exception as e:
        logging.error(f"Error getting fonts from {pdf_path}: {e}")
        return set()

def compare_fonts(fonts1, fonts2):
    """Compares fonts of two PDF files and returns discrepancies."""
    discrepancies = []
    if fonts1 != fonts2:
        original_fonts = "\n".join(sorted(fonts1))
        fake_fonts = "\n".join(sorted(fonts2))
        discrepancies.append(f"🟢 Original:\n{original_fonts}\n\n⚫️ Your Check:\n{fake_fonts}")
    return discrepancies

def pdf_to_images(pdf_path):
    """Converts a PDF file to images."""
    try:
        images = convert_from_path(pdf_path)
        return images
    except Exception as e:
        logging.error(f"Error converting PDF to images for {pdf_path}: {e}")
        return []

def extract_text_from_image(image):
    """Extracts text from an image."""
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image, lang='rus+eng')
    return text

def extract_image_features(image):
    """Extracts image features for comparison."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

def compare_texts(text1, text2, patterns):
    """Compares two texts, excluding dynamic fields."""
    text1 = remove_dynamic_fields(text1.lower(), patterns)
    text2 = remove_dynamic_fields(text2.lower(), patterns)
    
    seq = difflib.SequenceMatcher(None, text1, text2)
    match_ratio = seq.ratio()
    logging.debug(f"Text match ratio: {match_ratio}")
    return match_ratio

def compare_images_ssim(image1, image2):
    """Compares two images using SSIM."""
    image1_gray = extract_image_features(image1)
    image2_gray = extract_image_features(image2)

    if image1_gray.shape != image2_gray.shape:
        # Resize image2_gray to match image1_gray
        image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))

    score, _ = ssim(image1_gray, image2_gray, full=True)
    logging.debug(f"SSIM score: {score}")
    return score

def compare_images_orb(image1, image2):
    """Compares two images using ORB."""
    orb = cv2.ORB_create()
    
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2GRAY)

    # Resize second image
    if gray_image1.shape != gray_image2.shape:
        gray_image2 = cv2.resize(gray_image2, (gray_image1.shape[1], gray_image1.shape[0]))

    kp1, des1 = orb.detectAndCompute(gray_image1, None)
    kp2, des2 = orb.detectAndCompute(gray_image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 50]
    logging.debug(f"ORB good matches: {len(good_matches)}")
    return len(good_matches) / min(len(kp1), len(kp2))

def compare_color_histograms(image1, image2):
    """Compares the color histograms of two images."""
    if np.array(image1).size == 0 or np.array(image2).size == 0:
        logging.error("One of the images is empty or contains an error.")
        return 0
    
    try:
        hsv_image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2HSV)
        hsv_image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2HSV)
    except Exception as e:
        logging.error(f"Error converting images to HSV: {e}")
        return 0

    if hsv_image1.shape[0] == 0 or hsv_image1.shape[1] == 0 or hsv_image2.shape[0] == 0 or hsv_image2.shape[1] == 0:
        logging.error("One of the images has incorrect dimensions.")
        return 0

    hist1 = cv2.calcHist([hsv_image1], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv_image2], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    logging.debug(f"Histogram 1: {hist1.flatten()[:10]}")  
    logging.debug(f"Histogram 2: {hist2.flatten()[:10]}")  

    # Compare histograms
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    logging.debug(f"Color histogram similarity: {similarity}")
    
    return similarity

def convert_to_percentage(value, max_value=255):
    return (value / max_value) * 100

def visualize_image(image, title="Image"):
    """Function to visualize an image."""
    if image is not None and np.array(image).size > 0:
        cv2.imshow(title, np.array(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logging.error(f"{title} is empty or contains an error.")

def check_authenticity(original_pdf_path, fake_pdf_path):
    """Compares an original PDF with a fake PDF and generates a report."""
    original_images = pdf_to_images(original_pdf_path)
    fake_images = pdf_to_images(fake_pdf_path)

    original_metadata = get_pdf_metadata(original_pdf_path)
    fake_metadata = get_pdf_metadata(fake_pdf_path)
    metadata_discrepancies = compare_metadata(original_metadata, fake_metadata)

    original_fonts = get_pdf_fonts(original_pdf_path)
    fake_fonts = get_pdf_fonts(fake_pdf_path)
    font_discrepancies = compare_fonts(original_fonts, fake_fonts)

    if len(original_images) != 1 or len(fake_images) != 1:
        return "❗ Only single-page PDF files are supported."

    original_image = original_images[0]
    fake_image = fake_images[0]

    original_text = extract_text_from_image(original_image)
    fake_text = extract_text_from_image(fake_image)
    text_match_ratio = compare_texts(original_text, fake_text, dynamic_fields_patterns) * 100  # Convert to percentage

    image_similarity_ssim = compare_images_ssim(original_image, fake_image) * 100  
    image_similarity_orb = compare_images_orb(original_image, fake_image) * 100  

    mean_diff, std_dev_diff = compare_color_statistics(original_image, fake_image)
    avg_color_diff = compare_average_colors(original_image, fake_image)
    cluster_diff = compare_color_clusters(original_image, fake_image)

    mean_diff_percentage = convert_to_percentage(mean_diff)
    std_dev_diff_percentage = convert_to_percentage(std_dev_diff)
    avg_color_diff_percentage = convert_to_percentage(avg_color_diff)
    cluster_diff_percentage = convert_to_percentage(cluster_diff)

    report = []

    # Formatting report based on metadata
    if metadata_discrepancies:
        report.append("❌ Metadata Discrepancies:")
        for discrepancy in metadata_discrepancies:
            report.append(f"   {discrepancy}")
    else:
        report.append("✅ Metadata Matches.")

    # Formatting report based on fonts
    if font_discrepancies:
        report.append("🔍 Font Discrepancies:")
        report.append("🟢 Original:")
        for font in sorted(original_fonts):
            report.append(f"   {font}")
        report.append("🔴 Your Check:")
        for font in sorted(fake_fonts):
            report.append(f"   {font}")
    else:
        report.append("✅ Fonts Match.")

    # Outputting coefficients
    report.append(f"\n📄 Text Match Ratio: {text_match_ratio:.2f}%")
    if text_match_ratio < 70:
        report.append("🔍 Texts Differ. Check the content.")

    report.append(f"📈 Image Similarity (SSIM): {image_similarity_ssim:.2f}%")
    if image_similarity_ssim < 90:
        report.append("🔍 Images Differ. Check visual elements.")

    report.append(f"📊 Image Similarity (ORB): {image_similarity_orb:.2f}%")
    if image_similarity_orb < 20:
        report.append("🔍 Image key points differ significantly.")

    report.append(f"🎨 Average Color Difference: {avg_color_diff_percentage:.2f}%")
    if avg_color_diff_percentage > 4:
        report.append("🔍 Average colors of images differ significantly.")

    report.append(f"🎨 Differences in Mean Values and Standard Deviations: {mean_diff_percentage:.2f}%, {std_dev_diff_percentage:.2f}%")
    if mean_diff_percentage > 4 or std_dev_diff_percentage > 4:
        report.append("🔍 Mean values or standard deviations of colors differ significantly.")

    report.append(f"🎨 Color Cluster Difference: {cluster_diff_percentage:.2f}%")
    if cluster_diff_percentage > 10:
        report.append("🔍 Color clusters of images differ significantly.")

    return "\n".join(report)


original_pdf_path = 'path/to/original_pdf.pdf'
fake_pdf_path = 'path/to/fake_pdf.pdf'

report = check_authenticity(original_pdf_path, fake_pdf_path)
print(report)


