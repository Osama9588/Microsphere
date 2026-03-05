import streamlit as st
import cv2
import numpy as np

st.title("Microsphere Detection & Diameter Measurement")

# ==============================================================================
# CALIBRATION CONSTANTS
# ==============================================================================
MAGNIFICATION = 2.0
BASE_MICRONS_PER_PIXEL = 7.5
MICRONS_PER_PIXEL = BASE_MICRONS_PER_PIXEL / MAGNIFICATION


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

    pixels = roi[mask > 0]

    if len(pixels) == 0:
        return 0

    std = np.std(pixels)
    mean = np.mean(pixels)

    cv_val = std / mean if mean > 0 else 1

    return 1 - min(cv_val, 1)


def validate_sphere(gray, mask, x, y, r):

    score = 0

    circularity = calculate_circularity(mask, x, y, r)

    if circularity > 0.60:
        score += 3
    elif circularity > 0.50:
        score += 1

    uniformity = calculate_intensity_uniformity(gray, x, y, r)

    if uniformity > 0.70:
        score += 3
    elif uniformity > 0.50:
        score += 1

    if r > 120:
        score -= 2

    h, w = gray.shape

    if 0 <= x < w and 0 <= y < h:
        center_intensity = gray[y, x]

        if center_intensity > 140:
            score += 2
        elif center_intensity > 90:
            score += 1

    return score >= 2


def detect_overlapping_circles(circles):

    if circles is None or len(circles) == 0:
        return None

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

uploaded = st.file_uploader("Upload Microsphere Image", type=["png","jpg","jpeg"])

if uploaded is not None:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Original Image", channels="BGR")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 1.5)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2
    )

    opening = cv2.morphologyEx(
        opening, cv2.MORPH_CLOSE, kernel, iterations=1
    )

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

    st.write("Initial detections:",
             len(circles[0]) if circles is not None else 0)

    if circles is not None:
        circles = detect_overlapping_circles(circles)

        st.write("After NMS:",
                 len(circles[0]) if circles is not None else 0)

    validated = []

    if circles is not None:

        for (x,y,r) in circles[0]:

            x,y,r = int(x), int(y), int(r)

            if validate_sphere(gray, opening, x, y, r):
                validated.append((x,y,r))

    st.write("Final validated spheres:", len(validated))

    final_output = np.zeros_like(img)

    for (x,y,r) in validated:

        mask = np.zeros(gray.shape, np.uint8)

        cv2.circle(mask, (x,y), r, 255, -1)

        pixel_extract = cv2.bitwise_and(img, img, mask=mask)

        final_output = cv2.add(final_output, pixel_extract)

        diameter_px = 2*r
        diameter_um = diameter_px * MICRONS_PER_PIXEL

        label = f"{diameter_um:.0f} um"

        cv2.circle(final_output, (x,y), r, (0,255,0), 2)

        cv2.line(final_output,
                 (x-r,y),
                 (x+r,y),
                 (0,0,255),2)

        cv2.putText(
            final_output,
            label,
            (x-25, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    st.image(final_output, caption="Detected Microspheres", channels="BGR")
