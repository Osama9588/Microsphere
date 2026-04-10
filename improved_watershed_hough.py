import streamlit as st
import cv2
import numpy as np
import requests
import tempfile

# ==============================================================================
# CONFIG
# ==============================================================================

API_KEY = "H7vGKbWcGLGBtLqLHCDNWeSf"
BG_REMOVED_IMAGE = "no_bg.png"

# ==============================================================================
# CALIBRATION CONSTANTS
# ==============================================================================

MAGNIFICATION = 2.0
BASE_MICRONS_PER_PIXEL = 22.5
MICRONS_PER_PIXEL = BASE_MICRONS_PER_PIXEL / MAGNIFICATION


# ==============================================================================
# BACKGROUND REMOVAL
# ==============================================================================

def remove_background(image_path):

    try:
        with open(image_path, 'rb') as img_file:

            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files={'image_file': img_file},
                data={'size': 'auto'},
                headers={'X-Api-Key': API_KEY},
                timeout=30
            )

        if response.status_code == requests.codes.ok:

            with open(BG_REMOVED_IMAGE, 'wb') as out:
                out.write(response.content)

            st.success("✓ Background removed successfully")
            return BG_REMOVED_IMAGE

        elif response.status_code == 402:
            st.warning("⚠ API quota exceeded. Using original image.")
            return image_path

        elif response.status_code == 403:
            st.warning("⚠ Invalid API key. Using original image.")
            return image_path

        else:
            st.warning(f"⚠ remove.bg error {response.status_code}")
            return image_path

    except requests.exceptions.RequestException as e:
        st.warning("⚠ Network/API error. Using original image.")
        return image_path


# ==============================================================================
# SPHERE VALIDATION FUNCTIONS
# ==============================================================================

def calculate_circularity(mask, x, y, r):

    h, w = mask.shape
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return 0

    roi = mask[y-r:y+r, x-r:x+r]

    circle_mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (r, r), r, 255, -1)

    overlap = np.sum((roi > 0) & (circle_mask > 0))
    circle_area = np.pi * r * r

    return overlap / circle_area if circle_area > 0 else 0


def calculate_intensity_uniformity(gray, x, y, r):

    h, w = gray.shape
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return 0

    roi = gray[y-r:y+r, x-r:x+r]

    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)

    masked_pixels = roi[mask > 0]

    if len(masked_pixels) == 0:
        return 0

    std = np.std(masked_pixels)
    mean = np.mean(masked_pixels)

    cv_val = std / mean if mean > 0 else 1.0

    return 1.0 - min(cv_val, 1.0)


def validate_sphere(gray, mask, x, y, r):

    score = 0

    circularity = calculate_circularity(mask, x, y, r)

    if circularity > 0.60:
        score += 3
    elif circularity > 0.45:
        score += 1

    uniformity = calculate_intensity_uniformity(gray, x, y, r)

    if uniformity > 0.70:
        score += 3
    elif uniformity > 0.45:
        score += 1

    if r > 120:
        score -= 2

    h, w = gray.shape

    if 0 <= x < w and 0 <= y < h:

        y_start, y_end = max(0, y - 2), min(h, y + 3)
        x_start, x_end = max(0, x - 2), min(w, x + 3)

        center_patch = gray[y_start:y_end, x_start:x_end]

        if center_patch.size > 0:

            center_intensity = np.max(center_patch)

            if center_intensity > 130:
                score += 2
            elif center_intensity > 80:
                score += 1

    return score >= 2, score


def detect_overlapping_circles(circles):

    if circles is None or len(circles) == 0:
        return []

    circles_list = circles[0].tolist()

    circles_list.sort(key=lambda x: x[2], reverse=True)

    keep = []

    for c1 in circles_list:

        x1, y1, r1 = c1
        duplicate = False

        for c2 in keep:

            x2, y2, r2 = c2

            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

            if dist < max(r1, r2) * 0.5:
                duplicate = True
                break

        if not duplicate:
            keep.append(c1)

    return np.array([keep]) if keep else None


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("Microsphere Detection with Background Removal")

uploaded_file = st.file_uploader("Upload Microsphere Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    st.image(temp_path, caption="Uploaded Image", width="stretch")

    # Step 1: Remove background
    processed_image_path = remove_background(temp_path)

    # Step 2: Load image
    img = cv2.imread(processed_image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    final_output = np.zeros_like(img)

    # Step 3: Preprocessing
    blur = cv2.GaussianBlur(gray, (7,7), 1.5)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 4: Circle Detection
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=20,
        param1=50,
        param2=25,
        minRadius=12,
        maxRadius=75
    )

    st.write("Initial detections:", len(circles[0]) if circles is not None else 0)

    if circles is not None:
        circles = detect_overlapping_circles(circles)

    st.write("After overlap merging:", len(circles[0]) if circles is not None else 0)

    # Step 5: Validation
    validated = []

    if circles is not None:

        for (x,y,r) in circles[0]:

            x,y,r = int(x), int(y), int(r)

            is_valid, score = validate_sphere(gray, opening, x, y, r)

            if is_valid:
                validated.append((x,y,r))

    st.write("Final validated spheres:", len(validated))

    # Step 6: Draw Results
    for (x,y,r) in validated:

        mask = np.zeros(gray.shape, np.uint8)

        cv2.circle(mask, (x,y), r, 255, -1)

        pixel_extract = cv2.bitwise_and(img, img, mask=mask)

        final_output = cv2.add(final_output, pixel_extract)

        diameter_px = 2*r
        diameter_um = diameter_px * MICRONS_PER_PIXEL

        # label = f"{diameter_um:.0f} um"
        label = f"{diameter_um:.2f} um"
        
        cv2.circle(final_output, (x,y), r, (0,255,0), 2)

        # cv2.line(final_output, (x-r,y), (x+r,y), (0,0,255), 2)
        cv2.line(final_output, (x, y-r), (x, y+r), (0,0,255), 2)

        cv2.putText(final_output, label,
                    (x-25,y-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,0,255),
                    1)

    st.image(final_output, caption="Detected Microspheres", channels="BGR", width="stretch")

    st.success("Processing complete")
