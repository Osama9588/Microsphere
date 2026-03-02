import streamlit as st
import cv2
import numpy as np

# ==============================================================================
# CALIBRATION CONSTANTS (UNCHANGED)
# ==============================================================================
MICRONS_PER_PIXEL = 3.97  # <-- EXACT SAME VALUE


# ==============================================================================
# ORIGINAL FUNCTIONS (UNCHANGED)
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

    is_valid = score >= 2
    return is_valid, score


def detect_overlapping_circles(circles):
    if circles is None or len(circles) == 0:
        return []

    circles_list = circles[0].tolist()
    circles_list.sort(key=lambda x: x[2], reverse=True)

    keep = []
    for c1 in circles_list:
        x1, y1, r1 = c1
        is_duplicate = False

        for c2 in keep:
            x2, y2, r2 = c2
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            overlap_limit = max(r1, r2) * 0.5

            if dist < overlap_limit:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(c1)

    return np.array([keep]) if keep else None


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("🔬 Microsphere Detection with Micron Measurement")

uploaded_file = st.file_uploader("Upload sphere image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final_output = np.zeros_like(img)

    # Preprocessing (UNCHANGED)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Hough (UNCHANGED)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=30,
        param1=50,
        param2=18,
        minRadius=20,
        maxRadius=60
    )

    st.write(f"Initial detections: {len(circles[0]) if circles is not None else 0}")

    if circles is not None:
        circles = detect_overlapping_circles(circles)
        st.write(f"After NMS merging: {len(circles[0]) if circles is not None else 0}")

    validated = []

    if circles is not None:
        for (x, y, r) in circles[0]:
            x, y, r = int(x), int(y), int(r)
            is_valid, quality = validate_sphere(gray, opening, x, y, r)

            if is_valid:
                validated.append((x, y, r))

    st.success(f"Final validated spheres: {len(validated)}")

    # Render Output (UNCHANGED LOGIC)
    for (x, y, r) in validated:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        pixel_extract = cv2.bitwise_and(img, img, mask=mask)
        final_output = cv2.add(final_output, pixel_extract)

        diameter_px = 2 * r
        diameter_um = diameter_px * MICRONS_PER_PIXEL
        um_label = f"{diameter_um:.0f} um"

        cv2.circle(final_output, (x, y), r, (0, 255, 0), 2)
        cv2.line(final_output, (x, y - r), (x, y + r), (0, 0, 255), 2)

        cv2.putText(final_output, um_label, (x - 20, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    st.subheader("Detected Spheres with Micron Measurements")
    st.image(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB), width="stretch")

    # Download button
    _, buffer = cv2.imencode(".png", final_output)
    st.download_button(
        label="📥 Download Result",
        data=buffer.tobytes(),
        file_name="detected_microns.png",
        mime="image/png"
    )

# ---------------------------------------------------------------------------
# import cv2
# import numpy as np

# input_path = "background-removed\sphere5.png"
# img = cv2.imread(input_path)
# output = img.copy()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (7,7), 1.5)

# # =========================
# # EDGE MAP (NEW — KEY STEP)
# # =========================
# edges = cv2.Canny(blur, 50, 150)

# # --- Step 1: Binary mask ---
# _, thresh = cv2.threshold(blur, 0, 255,
#                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
#                            kernel, iterations=2)

# # --- Step 2: Watershed ---
# dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
# unknown = cv2.subtract(sure_bg, sure_fg)

# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0
# markers = cv2.watershed(img, markers)

# # =========================
# # EDGE SUPPORT FUNCTION (NEW)
# # =========================
# h, w = gray.shape

# def edge_support_ratio(cx, cy, r, edge_img):
#     """How much of the circle perimeter has edge support."""
#     pts = 0
#     hits = 0

#     for angle in range(0, 360, 5):
#         x = int(cx + r * np.cos(np.deg2rad(angle)))
#         y = int(cy + r * np.sin(np.deg2rad(angle)))

#         if 0 <= x < w and 0 <= y < h:
#             pts += 1
#             if edge_img[y, x] > 0:
#                 hits += 1

#     return hits / pts if pts > 0 else 0

# # --- Step 3: Hough Circle ---
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1.2,
#     minDist=25,
#     param1=100,
#     param2=18,
#     minRadius=8,
#     maxRadius=120
# )

# count = 0

# if circles is not None:
#     circles = np.uint16(np.around(circles))

#     for (x, y, r) in circles[0]:

#         # bounds check
#         if not (0 <= x < w and 0 <= y < h):
#             continue

#         # must lie in foreground
#         if opening[y, x] == 0:
#             continue

#         # =========================
#         # NEW: EDGE VALIDATION
#         # =========================
#         support = edge_support_ratio(x, y, r, edges)

#         #  MAIN FILTER — tune if needed
#         if support < 0.11:   # try 0.35–0.50
#             continue

#         count += 1
#         cv2.circle(output, (x, y), r, (0,255,0), 2)
#         cv2.putText(output, str(count), (x-10, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

# out_path = "3_edge_filtered.png"
# cv2.imwrite(out_path, output)

# print("Final count:", count)
# ---------------------------------------------------
# now adding diamter check
# import cv2
# import numpy as np

# input_path = "background-removed\sphere5.png"
# PIXEL_TO_UM = 1.0  #  change if you have calibration

# img = cv2.imread(input_path)
# output = img.copy()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (7,7), 1.5)

# # =========================
# # EDGE MAP
# # =========================
# edges = cv2.Canny(blur, 50, 150)

# # --- Binary mask ---
# _, thresh = cv2.threshold(
#     blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )

# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(
#     thresh, cv2.MORPH_OPEN, kernel, iterations=2
# )

# # --- Watershed ---
# dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
# unknown = cv2.subtract(sure_bg, sure_fg)

# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0
# markers = cv2.watershed(img, markers)

# # =========================
# # EDGE SUPPORT FUNCTION
# # =========================
# h, w = gray.shape

# def edge_support_ratio(cx, cy, r, edge_img):
#     pts = 0
#     hits = 0
#     for angle in range(0, 360, 5):
#         x = int(cx + r * np.cos(np.deg2rad(angle)))
#         y = int(cy + r * np.sin(np.deg2rad(angle)))
#         if 0 <= x < w and 0 <= y < h:
#             pts += 1
#             if edge_img[y, x] > 0:
#                 hits += 1
#     return hits / pts if pts > 0 else 0

# # --- Hough Circle ---
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1.2,
#     minDist=25,
#     param1=100,
#     param2=18,
#     minRadius=8,
#     maxRadius=120
# )

# count = 0
# diameters = []

# if circles is not None:
#     circles = np.uint16(np.around(circles))

#     for (x, y, r) in circles[0]:

#         if not (0 <= x < w and 0 <= y < h):
#             continue

#         if opening[y, x] == 0:
#             continue

#         #  Edge validation
#         support = edge_support_ratio(x, y, r, edges)
#         if support < 0.12:  # tune 0.35–0.50
#             continue

#         count += 1

#         # =========================
#         # DIAMETER CALCULATION
#         # =========================
#         diameter_px = 2 * r
#         diameter_um = diameter_px * PIXEL_TO_UM
#         diameters.append(diameter_um)

#         # --- Draw outer circle (green)
#         cv2.circle(output, (x, y), r, (0,255,0), 2)

#         # =========================
#         #  DRAW DIAMETER LINE
#         # =========================
#         # horizontal
#         # pt1 = (int(x - r), int(y))
#         # pt2 = (int(x + r), int(y))   
#         # cv2.line(output, pt1, pt2, (0,0,255), 2)
#         # vertical
#         pt1 = (int(x), int(y - r))
#         pt2 = (int(x), int(y + r))
#         cv2.line(output, pt1, pt2, (0,0,255), 2)

#         # --- Label diameter
#         cv2.putText(
#             output,
#             f"{diameter_um:.1f}",
#             (x - 20, y - r - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.4,
#             (0,255,0),
#             1
#         )

# out_path = "output/3_watershed_hough_with_diameter.png"
# cv2.imwrite(out_path, output)

# print("Final count:", count)
# print("Diameters:", diameters)
# ----------------------------------------------------
# import cv2
# import numpy as np

# input_path = "background-removed\sphere9.png"
# PIXEL_TO_UM = 1.0  #  change if you have calibration

# img = cv2.imread(input_path)
# output = img.copy()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (7,7), 1.5)

# # =========================
# # EDGE MAP
# # =========================
# edges = cv2.Canny(blur, 50, 150)

# # --- Binary mask ---
# _, thresh = cv2.threshold(
#     blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )

# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(
#     thresh, cv2.MORPH_OPEN, kernel, iterations=2
# )

# # --- Watershed ---
# dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
# unknown = cv2.subtract(sure_bg, sure_fg)

# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0
# markers = cv2.watershed(img, markers)

# # =========================
# # EDGE SUPPORT FUNCTION
# # =========================
# h, w = gray.shape

# def edge_support_ratio(cx, cy, r, edge_img):
#     pts = 0
#     hits = 0
#     for angle in range(0, 360, 5):
#         x = int(cx + r * np.cos(np.deg2rad(angle)))
#         y = int(cy + r * np.sin(np.deg2rad(angle)))
#         if 0 <= x < w and 0 <= y < h:
#             pts += 1
#             if edge_img[y, x] > 0:
#                 hits += 1
#     return hits / pts if pts > 0 else 0


# def calculate_sphere_coverage(cx, cy, r, mask_img):
#     """
#     Calculate what percentage of the circle is actually visible
#     (i.e., overlaps with white pixels in the mask)
#     """
#     pts = 0
#     visible_pts = 0
    
#     # Sample points around and inside the circle
#     for angle in range(0, 360, 10):
#         for radius in range(0, r+1, max(1, r//5)):
#             x = int(cx + radius * np.cos(np.deg2rad(angle)))
#             y = int(cy + radius * np.sin(np.deg2rad(angle)))
            
#             if 0 <= x < w and 0 <= y < h:
#                 pts += 1
#                 if mask_img[y, x] > 0:
#                     visible_pts += 1
    
#     coverage = visible_pts / pts if pts > 0 else 0
#     return coverage


# def find_visible_diameter_endpoints(cx, cy, r, mask_img, vertical=True):
#     """
#     Find the actual visible endpoints of the diameter line
#     by scanning from center outward until hitting background
#     """
#     if vertical:
#         # Scan upward from center
#         top_y = cy
#         for dy in range(0, r + 1):
#             y = cy - dy
#             if y < 0 or y >= h:
#                 break
#             if mask_img[y, cx] == 0:  # Hit background
#                 break
#             top_y = y
        
#         # Scan downward from center
#         bottom_y = cy
#         for dy in range(0, r + 1):
#             y = cy + dy
#             if y < 0 or y >= h:
#                 break
#             if mask_img[y, cx] == 0:  # Hit background
#                 break
#             bottom_y = y
        
#         return (cx, top_y), (cx, bottom_y)
    
#     else:  # horizontal
#         # Scan left from center
#         left_x = cx
#         for dx in range(0, r + 1):
#             x = cx - dx
#             if x < 0 or x >= w:
#                 break
#             if mask_img[cy, x] == 0:
#                 break
#             left_x = x
        
#         # Scan right from center
#         right_x = cx
#         for dx in range(0, r + 1):
#             x = cx + dx
#             if x < 0 or x >= w:
#                 break
#             if mask_img[cy, x] == 0:
#                 break
#             right_x = x
        
#         return (left_x, cy), (right_x, cy)


# # --- Hough Circle ---
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1.2,
#     minDist=25,
#     param1=100,
#     param2=18,
#     minRadius=8,
#     maxRadius=120
# )

# count = 0
# diameters = []

# if circles is not None:
#     circles = np.uint16(np.around(circles))

#     for (x, y, r) in circles[0]:

#         if not (0 <= x < w and 0 <= y < h):
#             continue

#         if opening[y, x] == 0:
#             continue

#         #  Edge validation
#         support = edge_support_ratio(x, y, r, edges)
#         if support < 0.12:  # tune 0.35–0.50
#             continue

#         # =========================
#         # CHECK SPHERE COVERAGE
#         # =========================
#         coverage = calculate_sphere_coverage(x, y, r, opening)
        
#         # Skip spheres that are mostly cut off (less than 50% visible)
#         if coverage < 0.5:
#             continue

#         count += 1

#         # =========================
#         # DIAMETER CALCULATION
#         # =========================
#         diameter_px = 2 * r
#         diameter_um = diameter_px * PIXEL_TO_UM
#         diameters.append(diameter_um)

#         # --- Draw outer circle (green)
#         cv2.circle(output, (x, y), r, (0,255,0), 2)

#         # =========================
#         # DRAW CLIPPED DIAMETER LINE
#         # =========================
#         # Find actual visible endpoints
#         pt1, pt2 = find_visible_diameter_endpoints(x, y, r, opening, vertical=True)
        
#         # Only draw if line is reasonably long (not just a tiny stub)
#         line_length = abs(pt2[1] - pt1[1])  # for vertical
#         if line_length > r * 0.4:  # At least 40% of expected diameter
#             cv2.line(output, pt1, pt2, (0,0,255), 2)
            
#             # Add small endpoint markers
#             cv2.circle(output, pt1, 3, (0,0,255), -1)
#             cv2.circle(output, pt2, 3, (0,0,255), -1)

#         # --- Label diameter
#         cv2.putText(
#             output,
#             f"{diameter_um:.1f}",
#             (x - 20, y - r - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.4,
#             (0,255,0),
#             1
#         )

# out_path = "output/9_watershed_hough_with_diameter.png"
# cv2.imwrite(out_path, output)

# print("Final count:", count)
# print("Diameters:", diameters)
# print(f"\nNote: Excluded {len(circles[0]) - count} partially visible spheres")
# -------------------------------------------------------
# working code with restrictions
# import cv2
# import numpy as np

# def calculate_circularity(mask, x, y, r):
#     """Check if detected region is actually circular"""
#     h, w = mask.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return 0
    
#     roi = mask[y-r:y+r, x-r:x+r]
#     circle_mask = np.zeros(roi.shape, dtype=np.uint8)
#     cv2.circle(circle_mask, (r, r), r, 255, -1)
    
#     overlap = np.sum((roi > 0) & (circle_mask > 0))
#     circle_area = np.pi * r * r
    
#     return overlap / circle_area if circle_area > 0 else 0


# def check_multiple_spheres_inside(mask, x, y, r):
#     """Detect if circle contains multiple separate spheres"""
#     h, w = mask.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return False
    
#     roi = mask[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
#     num_labels, labels = cv2.connectedComponents(roi)
    
#     # RELAXED: Allow up to 4 components (was 3)
#     if num_labels > 4:
#         return True
    
#     return False


# def calculate_intensity_uniformity(gray, x, y, r):
#     """Real spheres have relatively uniform brightness"""
#     h, w = gray.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return 0
    
#     roi = gray[y-r:y+r, x-r:x+r]
#     mask = np.zeros(roi.shape, dtype=np.uint8)
#     cv2.circle(mask, (r, r), r, 255, -1)
    
#     masked_pixels = roi[mask > 0]
#     if len(masked_pixels) == 0:
#         return 0
    
#     std = np.std(masked_pixels)
#     mean = np.mean(masked_pixels)
#     cv_val = std / mean if mean > 0 else 1.0
    
#     return 1.0 - min(cv_val, 1.0)


# def validate_sphere(gray, mask, x, y, r):
#     """Multi-criteria validation - RELAXED THRESHOLDS"""
#     score = 0
    
#     # Criterion 1: Circularity (RELAXED)
#     circularity = calculate_circularity(mask, x, y, r)
#     if circularity > 0.75:      # Was 0.80
#         score += 3
#     elif circularity > 0.65:    # Was 0.70
#         score += 2
#     elif circularity > 0.55:    # Was 0.60
#         score += 1
#     else:
#         return False, 0
    
#     # Criterion 2: NOT multiple spheres
#     if check_multiple_spheres_inside(mask, x, y, r):
#         return False, 0
    
#     # Criterion 3: Uniformity (RELAXED)
#     uniformity = calculate_intensity_uniformity(gray, x, y, r)
#     if uniformity > 0.70:       # Was 0.75
#         score += 3
#     elif uniformity > 0.60:     # Was 0.65
#         score += 2
#     elif uniformity > 0.50:     # Was 0.55
#         score += 1
    
#     # Criterion 4: Size check (RELAXED)
#     if r > 120:                 # Was 100
#         score -= 2
    
#     # Criterion 5: Brightness (RELAXED)
#     h, w = gray.shape
#     if 0 <= x < w and 0 <= y < h:
#         center_intensity = gray[y, x]
#         if center_intensity > 140:    # Was 150
#             score += 2
#         elif center_intensity > 90:   # Was 100
#             score += 1
    
#     # LOWERED THRESHOLD: 4 points instead of 5
#     is_valid = score >= 4
#     return is_valid, score


# def detect_overlapping_circles(circles):
#     """Remove large circles containing smaller ones"""
#     if circles is None or len(circles) == 0:
#         return []
    
#     circles_list = circles[0].tolist()
#     n = len(circles_list)
    
#     sorted_indices = sorted(range(n), key=lambda i: circles_list[i][2], reverse=True)
    
#     keep = []
#     removed = set()
    
#     for i in sorted_indices:
#         if i in removed:
#             continue
        
#         x1, y1, r1 = circles_list[i]
        
#         contains_others = False
#         for j in range(n):
#             if i == j or j in removed:
#                 continue
            
#             x2, y2, r2 = circles_list[j]
#             dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
#             # RELAXED: 0.85 instead of 0.90
#             if dist + r2 < r1 * 0.85:
#                 contains_others = True
#                 break
        
#         if not contains_others:
#             keep.append(circles_list[i])
#         else:
#             removed.add(i)
    
#     return np.array([keep]) if keep else None


# # ═══════════════════════════════════════════════════════════
# # MAIN PROCESSING
# # ═══════════════════════════════════════════════════════════

# input_path = 'background-removed\sphere3.png'
# PIXEL_TO_UM = 1.0

# img = cv2.imread(input_path)
# output = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # IMPROVED PREPROCESSING
# blur = cv2.GaussianBlur(gray, (5, 5), 1.2)  # Slightly less blur

# _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

# h, w = gray.shape

# # MORE AGGRESSIVE HOUGH PARAMETERS
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1.0,              # Was 1.1 - even higher precision
#     minDist=12,          # Was 15 - catch closer spheres
#     param1=70,           # Was 80 - more lenient edge detection
#     param2=11,           # Was 15 - MUCH more sensitive
#     minRadius=5,
#     maxRadius=90         # Was 80 - allow slightly larger
# )

# print(f"Initial Hough detections: {len(circles[0]) if circles is not None else 0}")

# if circles is not None:
#     circles = detect_overlapping_circles(circles)
#     print(f"After overlap removal: {len(circles[0]) if circles is not None else 0}")

# validated = []
# rejected = []

# if circles is not None:
#     for (x, y, r) in circles[0]:
#         x, y, r = int(x), int(y), int(r)
#         is_valid, quality = validate_sphere(gray, opening, x, y, r)
        
#         if is_valid:
#             validated.append((x, y, r, quality))
#         else:
#             rejected.append((x, y, r))

# print(f"Final validated: {len(validated)}")
# print(f"Rejected: {len(rejected)}")

# # Draw results
# count = 0
# for (x, y, r, quality) in validated:
#     count += 1
#     cv2.circle(output, (x, y), r, (0, 255, 0), 2)
#     pt1, pt2 = (x, y - r), (x, y + r)
#     cv2.line(output, pt1, pt2, (0, 0, 255), 2)
#     cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
    
#     diameter_um = 2 * r * PIXEL_TO_UM
#     cv2.putText(output, f"{diameter_um:.1f}", (x - 15, y - r - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# cv2.putText(output, f"Detected: {count} spheres", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# out_path = 'output/tuned_sphere_7_detection.png'
# cv2.imwrite(out_path, output)

# print(f"\n✓ Saved: {out_path}")
# print(f"✓ Total: {count}")
# ---------------------------------------
# 95% working code
# import cv2
# import numpy as np

# def calculate_circularity(mask, x, y, r):
#     """Check if detected region is actually circular"""
#     h, w = mask.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return 0
    
#     roi = mask[y-r:y+r, x-r:x+r]
#     circle_mask = np.zeros(roi.shape, dtype=np.uint8)
#     cv2.circle(circle_mask, (r, r), r, 255, -1)
    
#     overlap = np.sum((roi > 0) & (circle_mask > 0))
#     circle_area = np.pi * r * r
    
#     return overlap / circle_area if circle_area > 0 else 0


# def calculate_intensity_uniformity(gray, x, y, r):
#     """Real spheres have relatively uniform brightness"""
#     h, w = gray.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return 0
    
#     roi = gray[y-r:y+r, x-r:x+r]
#     mask = np.zeros(roi.shape, dtype=np.uint8)
#     cv2.circle(mask, (r, r), r, 255, -1)
    
#     masked_pixels = roi[mask > 0]
#     if len(masked_pixels) == 0:
#         return 0
    
#     std = np.std(masked_pixels)
#     mean = np.mean(masked_pixels)
#     cv_val = std / mean if mean > 0 else 1.0
    
#     return 1.0 - min(cv_val, 1.0)


# def validate_sphere(gray, mask, x, y, r):
#     """Multi-criteria validation - MORE PERMISSIVE"""
#     score = 0
    
#     # Criterion 1: Circularity
#     circularity = calculate_circularity(mask, x, y, r)
#     if circularity > 0.75:
#         score += 3
#     elif circularity > 0.65:
#         score += 2
#     elif circularity > 0.55:
#         score += 1
#     # Removing the hard 'else: return False' to be more lenient. 
#     # A low circularity just gets 0 points now.
    
#     # --- CHANGE 1: Disable the cluster check ---
#     # This check was rejecting spheres situated in the middle of a pack.
#     # We'll rely on circularity and uniformity to filter out bad blobs.
#     # if check_multiple_spheres_inside(mask, x, y, r):
#     #     return False, 0
    
#     # Criterion 2: Uniformity
#     uniformity = calculate_intensity_uniformity(gray, x, y, r)
#     if uniformity > 0.70:
#         score += 3
#     elif uniformity > 0.60:
#         score += 2
#     elif uniformity > 0.50:
#         score += 1
    
#     # Criterion 3: Size check
#     if r > 120:
#         score -= 2
    
#     # Criterion 4: Center Brightness (Strong feature of these spheres)
#     h, w = gray.shape
#     if 0 <= x < w and 0 <= y < h:
#         center_intensity = gray[y, x]
#         if center_intensity > 140:
#             score += 2
#         elif center_intensity > 90:
#             score += 1
    
#     # --- CHANGE 2: Lower the passing score ---
#     # Was 4, now 3. This allows slightly less perfect spheres to pass.
#     is_valid = score >= 3
#     return is_valid, score


# def detect_overlapping_circles(circles):
#     """Remove large circles containing smaller ones"""
#     if circles is None or len(circles) == 0:
#         return []
    
#     circles_list = circles[0].tolist()
#     n = len(circles_list)
    
#     # Sort by radius descending
#     sorted_indices = sorted(range(n), key=lambda i: circles_list[i][2], reverse=True)
    
#     keep = []
#     removed = set()
    
#     for i in sorted_indices:
#         if i in removed:
#             continue
        
#         x1, y1, r1 = circles_list[i]
        
#         contains_others = False
#         for j in range(n):
#             if i == j or j in removed:
#                 continue
            
#             x2, y2, r2 = circles_list[j]
#             dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
#             if dist + r2 < r1 * 0.85:
#                 contains_others = True
#                 break
        
#         if not contains_others:
#             keep.append(circles_list[i])
#         else:
#             removed.add(i)
    
#     return np.array([keep]) if keep else None


# # ═══════════════════════════════════════════════════════════
# # MAIN PROCESSING
# # ═══════════════════════════════════════════════════════════

# input_path = 'background-removed\sphere9.png' # Make sure this matches your filename
# PIXEL_TO_UM = 1.0

# img = cv2.imread(input_path)
# if img is None:
#     print(f"Error: Could not read image from {input_path}")
#     exit()

# output = img.copy()
# # Create a black background for the final output, as in your example
# final_output = np.zeros_like(img)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Preprocessing
# blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
# _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

# # --- CHANGE 3: Tune Hough Parameters ---
# # --- THE FIX: Tuned Hough Parameters ---
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1.0,
#     minDist=35,          # INCREASED: Centers must be at least ~1 radius apart
#     param1=50,           # Slightly lower edge threshold to catch faint edges
#     param2=15,           # Balanced: Less strict than 18, but stops the chaos of 11
#     minRadius=28,        # CRITICAL FIX: Ignore tiny internal reflections
#     maxRadius=50         # Restrict max size so it doesn't circle a whole cluster
# )

# print(f"Initial Hough detections: {len(circles[0]) if circles is not None else 0}")

# if circles is not None:
#     circles = detect_overlapping_circles(circles)
#     print(f"After overlap removal: {len(circles[0]) if circles is not None else 0}")

# validated = []

# if circles is not None:
#     for (x, y, r) in circles[0]:
#         x, y, r = int(x), int(y), int(r)
#         is_valid, quality = validate_sphere(gray, opening, x, y, r)
        
#         if is_valid:
#             validated.append((x, y, r))

# print(f"Final validated: {len(validated)}")

# # Draw results on the black background
# for (x, y, r) in validated:
#     # Draw the sphere from the original image onto the black background
#     mask = np.zeros(gray.shape, np.uint8)
#     cv2.circle(mask, (x, y), r, 255, -1)
#     pixel_s = cv2.bitwise_and(img, img, mask=mask)
#     final_output = cv2.add(final_output, pixel_s)

#     # Draw graphics
#     cv2.circle(final_output, (x, y), r, (0, 255, 0), 2)
#     pt1, pt2 = (x, y - r), (x, y + r)
#     cv2.line(final_output, pt1, pt2, (0, 0, 255), 2)
    
#     diameter_px = 2 * r
#     cv2.putText(final_output, f"{diameter_px:.1f}", (x - 15, y - r - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# out_path = 'improved_sphere_detection_6.png'
# cv2.imwrite(out_path, final_output)
# print(f"✓ Saved result to: {out_path}")
# --------------------------------------------------
# now with streamlit
# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile

# # ═══════════════════════════════════════════════════════════
# # ORIGINAL FUNCTIONS (UNCHANGED)
# # ═══════════════════════════════════════════════════════════

# def calculate_circularity(mask, x, y, r):
#     """Check if detected region is actually circular"""
#     h, w = mask.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return 0
    
#     roi = mask[y-r:y+r, x-r:x+r]
#     circle_mask = np.zeros(roi.shape, dtype=np.uint8)
#     cv2.circle(circle_mask, (r, r), r, 255, -1)
    
#     overlap = np.sum((roi > 0) & (circle_mask > 0))
#     circle_area = np.pi * r * r
    
#     return overlap / circle_area if circle_area > 0 else 0


# def calculate_intensity_uniformity(gray, x, y, r):
#     """Real spheres have relatively uniform brightness"""
#     h, w = gray.shape
#     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#         return 0
    
#     roi = gray[y-r:y+r, x-r:x+r]
#     mask = np.zeros(roi.shape, dtype=np.uint8)
#     cv2.circle(mask, (r, r), r, 255, -1)
    
#     masked_pixels = roi[mask > 0]
#     if len(masked_pixels) == 0:
#         return 0
    
#     std = np.std(masked_pixels)
#     mean = np.mean(masked_pixels)
#     cv_val = std / mean if mean > 0 else 1.0
    
#     return 1.0 - min(cv_val, 1.0)


# def validate_sphere(gray, mask, x, y, r):
#     """Multi-criteria validation - MORE PERMISSIVE"""
#     score = 0
    
#     circularity = calculate_circularity(mask, x, y, r)
#     if circularity > 0.75:
#         score += 3
#     elif circularity > 0.65:
#         score += 2
#     elif circularity > 0.55:
#         score += 1
    
#     uniformity = calculate_intensity_uniformity(gray, x, y, r)
#     if uniformity > 0.70:
#         score += 3
#     elif uniformity > 0.60:
#         score += 2
#     elif uniformity > 0.50:
#         score += 1
    
#     if r > 120:
#         score -= 2
    
#     h, w = gray.shape
#     if 0 <= x < w and 0 <= y < h:
#         center_intensity = gray[y, x]
#         if center_intensity > 140:
#             score += 2
#         elif center_intensity > 90:
#             score += 1
    
#     is_valid = score >= 3
#     return is_valid, score


# def detect_overlapping_circles(circles):
#     """Remove large circles containing smaller ones"""
#     if circles is None or len(circles) == 0:
#         return []
    
#     circles_list = circles[0].tolist()
#     n = len(circles_list)
    
#     sorted_indices = sorted(range(n), key=lambda i: circles_list[i][2], reverse=True)
    
#     keep = []
#     removed = set()
    
#     for i in sorted_indices:
#         if i in removed:
#             continue
        
#         x1, y1, r1 = circles_list[i]
        
#         contains_others = False
#         for j in range(n):
#             if i == j or j in removed:
#                 continue
            
#             x2, y2, r2 = circles_list[j]
#             dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
#             if dist + r2 < r1 * 0.85:
#                 contains_others = True
#                 break
        
#         if not contains_others:
#             keep.append(circles_list[i])
#         else:
#             removed.add(i)
    
#     return np.array([keep]) if keep else None


# # ═══════════════════════════════════════════════════════════
# # STREAMLIT UI
# # ═══════════════════════════════════════════════════════════

# st.title("🔬 Microsphere Detection")

# uploaded_file = st.file_uploader("Upload sphere image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Read image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     st.subheader("Original Image")
#     st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")

#     # ═══════════════════════════════════════════════════════════
#     # MAIN PROCESSING (UNCHANGED)
#     # ═══════════════════════════════════════════════════════════

#     PIXEL_TO_UM = 1.0
#     output = img.copy()
#     final_output = np.zeros_like(img)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

#     circles = cv2.HoughCircles(
#         blur,
#         cv2.HOUGH_GRADIENT,
#         dp=1.0,
#         minDist=35,
#         param1=50,
#         param2=15,
#         minRadius=28,
#         maxRadius=50
#     )

#     st.write(f"Initial Hough detections: {len(circles[0]) if circles is not None else 0}")

#     if circles is not None:
#         circles = detect_overlapping_circles(circles)
#         st.write(f"After overlap removal: {len(circles[0]) if circles is not None else 0}")

#     validated = []

#     if circles is not None:
#         for (x, y, r) in circles[0]:
#             x, y, r = int(x), int(y), int(r)
#             is_valid, quality = validate_sphere(gray, opening, x, y, r)
            
#             if is_valid:
#                 validated.append((x, y, r))

#     st.success(f"Final validated: {len(validated)}")

#     for (x, y, r) in validated:
#         mask = np.zeros(gray.shape, np.uint8)
#         cv2.circle(mask, (x, y), r, 255, -1)
#         pixel_s = cv2.bitwise_and(img, img, mask=mask)
#         final_output = cv2.add(final_output, pixel_s)

#         cv2.circle(final_output, (x, y), r, (0, 255, 0), 2)
#         pt1, pt2 = (x, y - r), (x, y + r)
#         cv2.line(final_output, pt1, pt2, (0, 0, 255), 2)
        
#         diameter_px = 2 * r
#         cv2.putText(final_output, f"{diameter_px:.1f}", (x - 15, y - r - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

#     # Show result
#     st.subheader("Detected Spheres")
#     st.image(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB), width="stretch")

#     # Download button
#     _, buffer = cv2.imencode(".png", final_output)
#     st.download_button(
#         label="📥 Download Result",
#         data=buffer.tobytes(),
#         file_name="improved_sphere_detection.png",
#         mime="image/png"
#     )
