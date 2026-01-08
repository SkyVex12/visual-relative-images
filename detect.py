import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def reflection_overlap_ratio(x, w, reflect_ranges, pad=8):
    """
    Fraction of the candidate (x..x+w) that overlaps any reflective x-interval.
    0.0 = no overlap, 1.0 = fully inside reflection.
    """
    rect_l = x - pad
    rect_r = x + w + pad
    total_w = max(1, rect_r - rect_l)

    overlap = 0
    for rl, rr in reflect_ranges:
        inter_l = max(rect_l, rl)
        inter_r = min(rect_r, rr)
        if inter_r > inter_l:
            overlap += inter_r - inter_l

    return overlap / total_w


def merge_contiguous_columns(cols):
    """Turn sorted column indices into contiguous (start, end) ranges."""
    ranges = []
    if len(cols) == 0:
        return ranges

    start = cols[0]
    prev = cols[0]
    for v in cols[1:]:
        if v == prev + 1:
            prev = v
        else:
            ranges.append((start, prev))
            start = v
            prev = v
    ranges.append((start, prev))
    return ranges


# -------------------- CONFIG --------------------
measure_path = r"D:\visual_relative_images\inspection\test"

# ROI crop (fraction of original height)
ROI_Y0 = 0.39
ROI_Y1 = 0.57

# Brightness adjustment (optional)
ALPHA = 1.0
BETA = 40

# Binary threshold
BIN_THR = 118

# Morphology
KERNEL_W, KERNEL_H = 7, 3
MORPH_ITERS = 2

# Reflection detection threshold on x-projection (tune per camera)
REFLECT_PROJ_THR = 78

# Reflection handling
REFLECT_PAD = 8  # was 100: too aggressive
REFLECT_OVERLAP_THR = 0.30

# Defect filtering
MIN_BOX_AREA = 90  # ignore tiny noise
SMALL_FAKE_AREA = 200  # ignore small blobs inside reflection
EDGE_IGNORE = 6  # ignore border artifacts

# Output
OUT_DIR = measure_path  # change if you want a separate output folder
# ------------------------------------------------


os.makedirs(OUT_DIR, exist_ok=True)

for filename in os.listdir(measure_path):
    if not filename.lower().endswith(
        (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ):
        continue

    start_time = time.time()
    full_path = os.path.join(measure_path, filename)
    name, ext = os.path.splitext(filename)

    measure = cv2.imread(full_path)
    if measure is None:
        print(f"Skip (cannot read): {filename}")
        continue

    # Brighter (optional)
    measure = cv2.convertScaleAbs(measure, alpha=ALPHA, beta=BETA)

    # Save debug: lightened
    cv2.imwrite(os.path.join(OUT_DIR, f"lightened_{filename}"), measure)

    gray_raw = cv2.cvtColor(measure, cv2.COLOR_BGR2GRAY)
    h_image, w_image = gray_raw.shape

    y0 = int(ROI_Y0 * h_image)
    y1 = int(ROI_Y1 * h_image)

    roi = gray_raw[y0:y1, :]
    vis = measure[y0:y1, :].copy()

    # Threshold segmentation
    _, binary = cv2.threshold(roi, BIN_THR, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours: {filename}")
        cv2.imwrite(os.path.join(OUT_DIR, f"result_{filename}"), vis)
        continue

    # Biggest area contour
    max_contour = max(contours, key=cv2.contourArea)

    # Mask biggest area
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    max_image = cv2.bitwise_and(binary, binary, mask=mask)

    # Morphology (use correct structuring element type!)
    kernel_link = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_W, KERNEL_H))
    test_image = cv2.morphologyEx(
        max_image, cv2.MORPH_CLOSE, kernel_link, iterations=MORPH_ITERS
    )

    cv2.imwrite(os.path.join(OUT_DIR, f"processed_{filename}"), test_image)

    # X-axis area projection (reflection detection)
    x_projection = np.sum(test_image == 255, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(x_projection)), x_projection, linewidth=1)
    plt.xlabel("X (image column)")
    plt.ylabel("White pixel count")
    plt.title("Horizontal Projection")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Horizontal_Projection_{name}.png"))
    plt.close()

    # Reflective columns -> ranges
    valid_x = np.where(x_projection > REFLECT_PROJ_THR)[0]
    reflect_ranges = merge_contiguous_columns(valid_x)

    # Find holes/sub-contours inside biggest blob using hierarchy
    contours_f, hierarchy_f = cv2.findContours(
        test_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy_f is None or len(contours_f) == 0:
        cv2.imwrite(os.path.join(OUT_DIR, f"result_{filename}"), vis)
        continue

    hierarchy_f = hierarchy_f[0]

    # Find largest parent contour (hierarchy parent == -1)
    max_area = 0
    max_idx = -1
    for i, cnt in enumerate(contours_f):
        if hierarchy_f[i][3] == -1:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_idx = i

    if max_idx == -1:
        cv2.imwrite(os.path.join(OUT_DIR, f"result_{filename}"), vis)
        continue

    # Draw candidate "defects": children of the largest parent contour
    for i, cnt in enumerate(contours_f):
        if hierarchy_f[i][3] != max_idx:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h

        # Basic noise filter
        if box_area < MIN_BOX_AREA:
            continue

        # Border artifact filter (fixes "left border defect")
        if x < EDGE_IGNORE or (x + w) > (roi.shape[1] - EDGE_IGNORE):
            continue

        # Reflection-aware skip: ignore only SMALL blobs in reflection
        overlap = reflection_overlap_ratio(x, w, reflect_ranges, pad=REFLECT_PAD)
        if overlap > REFLECT_OVERLAP_THR and box_area < SMALL_FAKE_AREA:
            continue

        # Keep large candidates even if in reflection (this is the key change)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    diff_time = time.time() - start_time
    print(f"{filename}: {diff_time:.4f}s")

    cv2.imwrite(os.path.join(OUT_DIR, f"result_{filename}"), vis)
