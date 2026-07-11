# TODO: Update the main function to your needs or remove it.

import cv2 as cv
import pygerber
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme, Project, FileTypeEnum, COLOR_MAP_T, DEFAULT_COLOR_MAP
import numpy as np
import time
from pygerber.common.rgba import RGBA
import math
from skimage.exposure import match_histograms
import re

import matplotlib.pyplot as plt

import serial

COLOR_MAP = {
    FileTypeEnum.COPPER: RGBA(r = 184, g = 115, b = 51, a = 255),
    FileTypeEnum.MASK: RGBA(r = 0, g = 100, b = 0, a = 180),
    FileTypeEnum.SILK:RGBA(r = 255, g = 255, b = 255, a = 255),
    FileTypeEnum.PASTE: RGBA(r = 200, g = 200, b = 200, a = 220)
}
COPPER = ColorScheme(
    background_color=RGBA.from_rgba(0, 0, 0, 0),
    clear_color=RGBA.from_rgba(0, 0, 0, 255),
    solid_color=RGBA.from_rgba(255, 255, 255, 255),
    clear_region_color=RGBA.from_rgba(0, 0, 0, 255),
    solid_region_color=RGBA.from_rgba(255, 255, 255, 255),
)

PASTE_MASK = ColorScheme(
    background_color=RGBA.from_rgba(0, 0, 0, 0),
    clear_color=RGBA.from_rgba(0, 0, 0, 0),
    solid_color=RGBA.from_rgba(200, 219, 235, 255),
    clear_region_color=RGBA.from_rgba(0, 0, 0, 0),
    solid_region_color=RGBA.from_rgba(200, 219, 235, 255),
)
SOLDER_MASK = ColorScheme(
    background_color=RGBA.from_rgba(0, 0, 0, 0),
    clear_color=RGBA.from_rgba(0, 0, 0, 0),
    solid_color=RGBA.from_rgba(200, 219, 235, 255),
    clear_region_color=RGBA.from_rgba(0, 0, 0, 0),
    solid_region_color=RGBA.from_rgba(200, 219, 235, 255),
)
SILK = ColorScheme(
    background_color=RGBA.from_hex("#00000000"),
    clear_color=RGBA.from_hex("#00000000"),
    solid_color=RGBA.from_hex("#FFFFFFFF"),
    clear_region_color=RGBA.from_hex("#00000000"),
    solid_region_color=RGBA.from_hex("#FFFFFFFF"),
)
COLOR_MAP: COLOR_MAP_T = {
    FileTypeEnum.COPPER: COPPER,
    FileTypeEnum.MASK: SOLDER_MASK,
    FileTypeEnum.PASTE: PASTE_MASK,
    FileTypeEnum.SILK: SILK,
    FileTypeEnum.EDGE: SILK,
    FileTypeEnum.OTHER: ColorScheme.DEBUG_1_ALPHA,
    FileTypeEnum.UNDEFINED: ColorScheme.DEBUG_1_ALPHA,
    FileTypeEnum.PLATED: SOLDER_MASK,
    FileTypeEnum.NON_PLATED: PASTE_MASK,
    FileTypeEnum.PROFILE: SILK,
    FileTypeEnum.SOLDERMASK: SOLDER_MASK,
    FileTypeEnum.LEGEND: SILK,
    FileTypeEnum.COMPONENT: PASTE_MASK,
    FileTypeEnum.GLUE: PASTE_MASK,
    FileTypeEnum.CARBONMASK: SOLDER_MASK,
    FileTypeEnum.GOLDMASK: SOLDER_MASK,
    FileTypeEnum.HEATSINKMASK: SOLDER_MASK,
    FileTypeEnum.PEELABLEMASK: SOLDER_MASK,
    FileTypeEnum.SILVERMASK: SOLDER_MASK,
    FileTypeEnum.TINMASK: SOLDER_MASK,
    FileTypeEnum.DEPTHROUT: PASTE_MASK,
    FileTypeEnum.VCUT: PASTE_MASK,
    FileTypeEnum.VIAFILL: PASTE_MASK,
    FileTypeEnum.PADS: PASTE_MASK,
}

def extract_coords(gbr_path: str, offset_x: float = None, offset_y: float = None):
    """Extract X/Y pad coordinates with aperture sizes."""
    gerber_file = GerberFile.from_file(gbr_path)
    source = gerber_file.source_code
    scale = 25.4 if "%MOIN*%" in source else 1.0
    fmt_match = re.search(r'%FSLA[XY](\d)(\d)', source)
    if fmt_match:
        decimal_places = int(fmt_match.group(2))
        divisor = 10 ** decimal_places
    else:
        divisor = 1_000_000

    aperture_sizes  = {}
    aperture_shapes = {}
    for match in re.finditer(r'%ADD(\d+)([A-Za-z]+),([^*]+)\*%', source):
        apt_id   = match.group(1)
        apt_type = match.group(2)
        params   = match.group(3).split('X')

        if apt_type == 'C':
            size  = float(params[0]) * scale
            shape = 'C'
        elif apt_type == 'R':
            width  = float(params[0]) * scale
            height = float(params[1]) * scale if len(params) > 1 else width
            size   = width
            shape  = f'RR:{height}'
        elif apt_type == 'RoundRect':
            try:
                r      = abs(float(params[0]))  # corner radius extends beyond corner points
                dxs    = [abs(float(params[i])) for i in range(1, len(params) - 1, 2)]
                dys    = [abs(float(params[i])) for i in range(2, len(params) - 1, 2)]
                width  = (max(dxs) + r) * 2 * scale
                height = (max(dys) + r) * 2 * scale
                size  = width
                shape = f'RR:{height}'
            except:
                size  = 0.6
                shape = 'C'
        else:
            size  = float(params[0]) if params else 0.2
            shape = 'C'

        aperture_sizes[apt_id]  = size
        aperture_shapes[apt_id] = shape

    for match in re.finditer(r'G04:AMPARAMS\|DCode=(\d+)\|XSize=([\d.]+)mm\|YSize=([\d.]+)mm[^*]*Shape=(\w+)', source):
        apt_id = match.group(1)
        width  = float(match.group(2))
        height = float(match.group(3))
        shape  = match.group(4)
        if apt_id not in aperture_sizes:
            aperture_sizes[apt_id]  = width
            aperture_shapes[apt_id] = f'RR:{height}' if shape == 'RoundedRectangle' else 'C'

    for match in re.finditer(r'%ADD(\d+)([A-Za-z]\w+)\*%', source):
        apt_id = match.group(1)
        if apt_id not in aperture_sizes:
            aperture_sizes[apt_id]  = 1.4
            aperture_shapes[apt_id] = 'C'

    raw = []
    current_aperture = None
    current_x = 0.0
    current_y = 0.0

    for line in source.split('\n'):
        line = line.strip()
        apt_match = re.match(r'D(\d+)\*', line)
        if apt_match and int(apt_match.group(1)) >= 10:
            current_aperture = apt_match.group(1)

        coord_move = re.match(r'X(-?\d+)Y(-?\d+)D02', line)
        if coord_move:
            current_x = int(coord_move.group(1)) / divisor * scale
            current_y = int(coord_move.group(2)) / divisor * scale

        x_only = re.match(r'X(-?\d+)D0[23]\*', line)
        if x_only:
            current_x = int(x_only.group(1)) / divisor * scale

        y_only = re.match(r'Y(-?\d+)D0[23]\*', line)
        if y_only:
            current_y = int(y_only.group(1)) / divisor * scale

        d03_match = re.match(r'X(-?\d+)Y(-?\d+)D03', line)
        if d03_match:
            x = int(d03_match.group(1)) / divisor * scale
            y = int(d03_match.group(2)) / divisor * scale
            size  = aperture_sizes.get(current_aperture, 0.2)
            shape = aperture_shapes.get(current_aperture, 'C')
            raw.append((x, y, size, shape))
        elif re.match(r'D03\*', line):
            size  = aperture_sizes.get(current_aperture, 0.2)
            shape = aperture_shapes.get(current_aperture, 'C')
            raw.append((current_x, current_y, size, shape))

    # --- G36/G37 filled regions → treat as rectangular pads ---
    for region_match in re.finditer(r'G36\*(.*?)G37\*', source, re.DOTALL):
        block = region_match.group(1)
        pts = [(int(m.group(1)) / divisor * scale, int(m.group(2)) / divisor * scale)
               for m in re.finditer(r'X(-?\d+)Y(-?\d+)D0[12]\*', block)]
        if len(pts) >= 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w  = max(xs) - min(xs)
            h  = max(ys) - min(ys)
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            if w > 0 and h > 0:
                raw.append((cx, cy, w, f'RR:{h}'))

    if not raw:
        return [], 0, 0

    min_x = offset_x if offset_x is not None else min(c[0] for c in raw)
    min_y = offset_y if offset_y is not None else min(c[1] for c in raw)
    max_y = max(c[1] for c in raw)
    return [(x - offset_x, - y + offset_y, size, shape) for x, y, size, shape in raw], min_x, min_y


def main() -> None:

    # Render reference image
    # Project(
    #     [
    #         GerberFile.from_file(
    #             'GerberFiles/copper_top.gbr',
    #             FileTypeEnum.COPPER,
    #         ),
    #         GerberFile.from_file(
    #             'GerberFiles/soldermask_top.gbr',
    #             FileTypeEnum.MASK,
    #         ),
    #         GerberFile.from_file(
    #             'GerberFiles/solderpaste_top.gbr',
    #             FileTypeEnum.PASTE,
    #         ),
    #         GerberFile.from_file(
    #             'GerberFiles/silkscreen_top.gbr',
    #             FileTypeEnum.SILK,
    #         ),
    #     ],
    # ).parse().render_raster("ref.png", color_map = COLOR_MAP, dpmm=40)
    Project(
        [
            GerberFile.from_file(
                'GerberFiles2/copper_top.gbr',
                FileTypeEnum.COPPER,
            ),
        #     GerberFile.from_file(
        #         'GerberFiles/soldermask_top.gbr',
        #         FileTypeEnum.MASK,
        #     ),
        #     GerberFile.from_file(
        #         'GerberFiles/solderpaste_top.gbr',
        #         FileTypeEnum.PASTE,
        #     ),
        #     GerberFile.from_file(
        #         'GerberFiles/silkscreen_top.gbr',
        #         FileTypeEnum.SILK,
        #     ),
        ],
    ).parse().render_raster("OLDREF.png", color_map = COLOR_MAP, dpmm=58)

    video = cv.VideoCapture(0)

    i = 1

    ### Set up video
    while (not video.isOpened()):
        video = cv.VideoCapture(i)
        i = i + 1

    video.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if (not video.isOpened()):
        return

    # Clear out bad first captures 
    for i in range(0, 20):
        _, _ = video.read()

    ### Set up serial
    ser = serial.Serial('/dev/ttyAMA0', 9600)
    ser.flush()

    # video.set(cv.CAP_PROP_SETTINGS, 1)

    # video.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    # time.sleep(1)

    # video.set(cv.CAP_PROP_EXPOSURE, 1000)

    # video.set(cv.CAP_PROP_SETTINGS, 1)
    n = 0

    # total_img = None

    # Open the image files.
    # gray_color = cv.imread("image" + str(n) + ".jpg")  # Image to be aligned.
    # img2_color = cv.imread("ref.png")    # Reference i 9mage.

    # img3 = img2_color.copy()

    ref = None

    # i = 0

    # stitcher = cv.Stitcher_create(cv.Stitcher_SCANS)

    # stitcher.setPanoConfidenceThresh(0.4)

    stitch_list = []

    # stitched_image = None

    # roi = None

    y_shift0 = 0
    x_shift0 = 0

    row_y_shift0 = 0
    row_y_shift = 0

    x = 283
    y = 0
    w = 691
    h = 567

    raw_image = None
    trace_image = None
    laplacian_image = None

    raw_row = np.zeros((h, w, 3), dtype = np.uint8)
    trace_row = np.zeros((720, 1280), dtype = np.uint8)
    prev_laplacian_row = None
    laplacian_row = np.zeros((h, w), dtype = np.float32)
    prev_first_in_row = None
    first_in_row = None

    # sum_y = 0
    # sum_x = 0
    # n_angles = 0

    check, current_frame = video.read()

    # roi = cv.selectROI("selectioj ROI", current_frame)
    # x, y, w, h = roi
    # print(str(x) + " " + str(y) + " " + str(w) + " " + str(h))

    ref_frame = None
    crop = None


    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    line_h, = axs[0].plot([], color='m')
    line_s, = axs[1].plot([], color='g')
    line_v, = axs[2].plot([], color='k')
    axs[0].set_xlim(0, 179); axs[0].set_title('Hue')
    axs[1].set_xlim(0, 255); axs[1].set_title('Saturation')
    axs[2].set_xlim(0, 255); axs[2].set_title('Value')

    first_row = True
    do_capture = True
    x_phase_sum = 0
    x_phase_n = 0
    x_phase_average = 0

    ref_gbr_file = None
    file_transfer_mode = False

    scale = 58

    ### Infinite loop waiting for serial and responding to commands
    while (1):

        while (ser.in_waiting == 0):
            check, current_frame = video.read()
            cv.imshow("Frame Feed", current_frame)
            
            if (crop is not None):
                crop_x, crop_y, crop_w, crop_h = crop
                cropped_frame = current_frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                cropped_ycrcb = cv.cvtColor(cropped_frame, cv.COLOR_BGR2YCrCb)
                y_channel, cr_channel, cb_channel = cv.split(cropped_ycrcb)
                clahe2 = cv.createCLAHE(clipLimit = 3.0, tileGridSize=(8, 8))
                y_channel_eq = cv.equalizeHist(y_channel)
                cropped_eq = cv.merge([y_channel_eq, cr_channel, cb_channel])
                cropped_eq_bgr = cv.cvtColor(cropped_eq, cv.COLOR_YCrCb2BGR)
                cv.imshow("Equalized", cropped_eq_bgr)

            # Convert to HSV
            current_hsv = cv.cvtColor(current_frame, cv.COLOR_BGR2HSV)

            # Split channels
            h_____, s, v = cv.split(current_hsv)
            # Per-channel histograms
            hist_h = cv.calcHist([h_____], [0], None, [180], [0, 180])
            hist_s = cv.calcHist([s], [0], None, [256], [0, 256])
            hist_v = cv.calcHist([v], [0], None, [256], [0, 256])

            # Normalize for display
            cv.normalize(hist_h, hist_h, 0, 1, cv.NORM_MINMAX)
            cv.normalize(hist_s, hist_s, 0, 1, cv.NORM_MINMAX)
            cv.normalize(hist_v, hist_v, 0, 1, cv.NORM_MINMAX)

            # Matplotlib live update (optional)
            line_h.set_data(np.arange(180), hist_h.flatten())
            line_s.set_data(np.arange(256), hist_s.flatten())
            line_v.set_data(np.arange(256), hist_v.flatten())
            for ax in axs:
                ax.relim(); ax.autoscale_view()
            plt.pause(0.001)

            # 2D HS histogram visualization
            hist_hs = cv.calcHist([h_____, s], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv.normalize(hist_hs, hist_hs, 0, 255, cv.NORM_MINMAX)
            hist_hs_u8 = hist_hs.astype(np.uint8)

            # Build HSV image: cols=H, rows=S, value=freq
            hs_img = np.zeros((256, 180, 3), dtype=np.uint8)
            # H: 0..h_bins-1 scaled to 0..179 if h_bins !=180
            h_values = np.linspace(0, 179, 180).astype(np.uint8)
            s_values = np.arange(256).astype(np.uint8)
            for i, hv in enumerate(h_values):
                hs_img[:, i, 0] = hv                   # Hue
            hs_img[:, :, 1] = s_values[:, None]       # Saturation (rows)
            hs_img[:, :, 2] = cv.transpose(hist_hs_u8)  # Value = frequency (note transpose)

            hs_bgr = cv.cvtColor(hs_img, cv.COLOR_HSV2BGR)

            # Display original frame and HS histogram
            hs_display = cv.resize(hs_bgr, (640, 360))
            cv.imshow('HS histogram (H horiz, S vert, V=freq)', hs_display)

            # time.sleep(0.04)
            cv.waitKey(80)

        line = ser.readline().decode('utf-8')

        line_stripped = line.strip()

        command = line_stripped.split()[0]

        print(command)

        if (command == "END_FILE_TRANSFER"):
            ref_gbr_file.close()
            file_transfer_mode = False


            Project(
                [
                    GerberFile.from_file(
                        'ref.gbr',
                        FileTypeEnum.COPPER,
                    ),
                #     GerberFile.from_file(
                #         'GerberFiles/soldermask_top.gbr',
                #         FileTypeEnum.MASK,
                #     ),
                #     GerberFile.from_file(
                #         'GerberFiles/solderpaste_top.gbr',
                #         FileTypeEnum.PASTE,
                #     ),
                #     GerberFile.from_file(
                #         'GerberFiles/silkscreen_top.gbr',
                #         FileTypeEnum.SILK,
                #     ),
                ],
            ).parse().render_raster("ref.png", color_map = COLOR_MAP, dpmm=scale)
            
            ref = cv.imread("ref.png")
        elif (file_transfer_mode):
            print("Written to file")
            ref_gbr_file.write(line)
        elif (command == "BEGIN_FILE_TRANSFER"):
            print("ENTERED")
            ref_gbr_file = open("ref.gbr", "w")
            file_transfer_mode = True
        elif (command == "START_LAYER"):
            print("STARTED")
        elif (command == "CAPTURE"):

            if (first_row or do_capture):
                # if (first_in_row is None):
                for j in range(10):
                    video.grab()
                    # cv.waitKey(40)
                img_and = None

                for i in range(1):
                    # if (first_in_row is None):
                    #     for j in range(5):
                    #         video.grab()
                    #         cv.waitKey(80)
                    check, current_frame = video.read()
                    cv.imshow("Frame Feed", current_frame)
                    bilateral_blur = cv.bilateralFilter(current_frame, 21, 50, 200)  # Image to be aligned.
                    gaussian_blur = cv.GaussianBlur(bilateral_blur, (5,5), 0)  # Image to be aligned.
                    # gray = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2GRAY)
                    # cv.imshow("Bilateral and Gaussian Blur", gaussian_blur)
                    # cv.imshow("Grayed Blur", gray)
                    # if (ref_frame is not None):
                    #     current_hsv = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2HSV)
                    #     ref_hsv = cv.cvtColor(ref_frame, cv.COLOR_BGR2HSV)
                    #     current_h, current_s, current_v = cv.split(current_hsv)
                    #     _, _, ref_v = cv.split(ref_hsv)
                    #     # v_hist_match = match_histograms(current_v, ref_v, channel_axis=None).astype(np.uint8)
                    #     current_hist = cv.calcHist([current_v], [0], None, [256], [0, 256])
                    #     ref_hist = cv.calcHist([ref_v], [0], None, [256], [0, 256])
                    #     current_cdf = current_hist.cumsum()
                    #     ref_cdf = ref_hist.cumsum()
                    #     current_cdf_norm = current_cdf / current_cdf[-1]
                    #     ref_cdf_norm = ref_cdf / ref_cdf[-1]
                    #     # 5. Create the Mapping Lookup Table (LUT)
                    #     lut = np.zeros(256, dtype=np.uint8)
                    #     ref_idx = 0
                        
                    #     for src_idx in range(256):
                    #         while ref_idx < 255 and ref_cdf_norm[ref_idx] < current_cdf_norm[src_idx]:
                    #             ref_idx += 1
                    #         lut[src_idx] = ref_idx
                            
                    #     # 6. Apply LUT using OpenCV's native function (keeps layout continuous)
                    #     matched_v = cv.LUT(current_v, lut)
                    #     current_hsv = cv.merge([current_h, current_s, matched_v])
                    #     current_hist_match = cv.cvtColor(current_hsv, cv.COLOR_HSV2BGR)

                    #     cv.imshow("Matched", current_hist_match)
                    # else:
                    #     current_hist_match = gaussian_blur

                    ### Extracting traces
                    img_hsv = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2HSV)

                    # cv.imshow("HSV", img_hsv)

                    trace_lower = np.array([90, 51, 204])
                    trace_higher = np.array([105, 255, 255])

                    img_threshold = cv.inRange(img_hsv, trace_lower, trace_higher)

                    empty_lower = np.array([0, 0, 0]) # Maybe reduce this range
                    empty_higher = np.array([45, 38, 153])

                    dark_threshold = cv.inRange(img_hsv, empty_lower, empty_higher)

                    img_diff = cv.subtract(img_threshold, dark_threshold)

                    # cv.imshow("Trace Threshold", img_threshold)

                    # cv.imshow("Dark Threshold", dark_threshold)
                        
                    # cv.namedWindow("Difference of Thresholds", cv.WINDOW_NORMAL)
                    # cv.resizeWindow("Difference of Thresholds", 1000, 1000)
                    # cv.imshow("Difference of Thresholds", img_diff)

                    # open = cv.morphologyEx(img_diff, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)))

                    # cv.imshow("Closed Difference", open)

                    if (img_and is None):
                        img_and = img_diff
                    else:
                        img_and = cv.bitwise_and(img_and, img_diff)

                    # cv.imshow("AND Operation", img_and)
                    # print("Hello")

                    # cv.waitKey(0)
                    # time.sleep(5)

                # for i in range(5):
                #     # if (first_in_row is None):
                #     #     for j in range(5):
                #     #         video.grab()
                #     #         cv.waitKey(80)
                #     check, current_frame = video.read()

                    
                

                print("Done!")

                # print(type(x))
                # print(type(y))
                # print(type(w))
                # print(type(h))

                # cropped_img = gaussian_blur[y:y+h, x:x+w]
                cropped_img_unblur = current_frame[y:y+h, x:x+w]

                # resized_img = cv.resize(cropped_img, None, fx = 0.44, fy = 0.44, interpolation = cv.INTER_AREA)

                # cv.imwrite("image" + str(n) + ".jpg", current_frame)

                # res = cv.matchTemplate(img2_color, resized_img, cv.TM_CCOEFF_NORMED)

                # _, max_val, _, max_loc = cv.minMaxLoc(res)

                # img3[max_loc[1]:max_loc[1] + resized_img.shape[0], max_loc[0]: max_loc[0] + resized_img.shape[1], :] = resized_img

                # cv.namedWindow("test", cv.WINDOW_NORMAL)
                # cv.resizeWindow("test", 1000, 1000)
                # cv.imshow("test", img3)

                stitch_list.append(cropped_img_unblur)

                clahe = cv.createCLAHE(clipLimit = 3.0, tileGridSize=(8, 8))

                current = stitch_list[-1]
                current_gray = cv.cvtColor(current, cv.COLOR_BGR2GRAY)
                current_clahe = clahe.apply(current_gray)
                current_enhanced = cv.Laplacian(current_clahe, cv.CV_32F, ksize=3)
                current_32 = current_enhanced.astype(np.float32)

                if (first_in_row is None):
                    first_in_row = current_32.copy()
                    
                # cv.imshow("Enhanced Current", current_enhanced)
                if (len(stitch_list) > 1):
                    print("Stitch list is larger than 1")

                    previous = stitch_list[-2]

                    previous_gray = cv.cvtColor(previous, cv.COLOR_BGR2GRAY)
                    
                    previous_clahe = clahe.apply(previous_gray)

                    previous_enhanced = cv.Laplacian(previous_clahe, cv.CV_32F, ksize=3)

                    previous_32 = previous_enhanced.astype(np.float32)

                    shift, response = cv.phaseCorrelate(previous_32, current_32)
                    x_shift = int(round(shift[0]))
                    # y_shift = int(round(shift[1]))
                    y_shift = 0

                    # if (first_row):
                    x_phase_sum += x_shift
                    x_phase_n += 1
                    x_phase_average = int(x_phase_sum / x_phase_n)

                    print(x_shift)
                    print(x_shift0)
                    print(x_phase_average)

                    raw_row = cv.copyMakeBorder(raw_row, 0, 0, 0,  abs(x_phase_average), borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                    trace_row = cv.copyMakeBorder(trace_row, 0, 0, 0, abs(x_phase_average), borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                    laplacian_row = cv.copyMakeBorder(laplacian_row, 0, 0, 0, abs(x_phase_average), borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])


                    y_shift0 += abs(y_shift)

                    x_shift0 += abs(x_phase_average)
                else:
                    raw_row = current.copy()
                    trace_row = img_and.copy()
                    laplacian_row = current_enhanced.copy()
                    x_shift = 0
                    y_shift = 0

                raw_row[y_shift0 + abs(y_shift):y_shift0 + abs(y_shift) + current.shape[0], x_shift0:x_shift0 + current.shape[1]] = current.copy()
                laplacian_row[y_shift0 + abs(y_shift):y_shift0 + abs(y_shift) + current.shape[0], x_shift0:x_shift0 + current.shape[1]] = current_enhanced.copy()

                print(type(img_and))

                trace_row[y_shift0 + abs(y_shift):y_shift0 + abs(y_shift) + img_and.shape[0], x_shift0:x_shift0 + img_and.shape[1]] = cv.bitwise_or(trace_row[y_shift0 + abs(y_shift):y_shift0 + abs(y_shift) + img_and.shape[0], x_shift0:x_shift0 + img_and.shape[1]], img_and)


                print(x_shift0)
                print("")

                # canvas[abs(y_shift):abs(y_shift) + current.shape[0], abs(x_shift):abs(x_shift) + current.shape[1]] = transparent

                cv.namedWindow("Phase correlated", cv.WINDOW_NORMAL)
                cv.resizeWindow("Phase correlated", 1000, 1000)
                cv.imshow("Phase correlated", raw_row)

                cv.namedWindow("Trace Canvas", cv.WINDOW_NORMAL)
                cv.resizeWindow("Trace Canvas", 1000, 1000)
                cv.imshow("Trace Canvas", trace_row)

                cv.namedWindow("Laplacian Canvas", cv.WINDOW_NORMAL)
                cv.resizeWindow("Laplacian Canvas", 1000, 1000)
                cv.imshow("Laplacian Canvas", laplacian_row)

                # if (prev_laplacian_row is not None):
                #     cv.imshow("Previous", prev_laplacian_row)
                # cv.imshow("Now", laplacian_row)

                if (prev_first_in_row is not None):
                    cv.imshow("Previous", prev_first_in_row)

                if (first_in_row is not None):
                    cv.imshow("Now", first_in_row)

            if (not first_row):
                do_capture = not do_capture

        elif (command == "NEW_ROW"):

            # if (first_row == True):
            #     x_phase_average *= 2
            #     x_phase_average = int(x_phase_average)
            #     first_row = False

            # clahe2 = cv.createCLAHE(clipLimit = 3.0, tileGridSize=(8, 8))

            # current_row_gray = cv.cvtColor(laplacian_row, cv.COLOR_BGR2GRAY)
            # current_row_clahe = clahe2.apply(current_row_gray)
            # current_row_enhanced = cv.Laplacian(current_row_clahe, cv.CV_32F, ksize=3)
            # current_row_32 = current_row_enhanced.astype(np.float32)
            # cv.imshow("Enhanced Current", current_enhanced)
            if (prev_first_in_row is not None):
                # previous_row_gray = cv.cvtColor(prev_laplacian_row, cv.COLOR_BGR2GRAY)
                
                # previous_row_clahe = clahe2.apply(previous_row_gray)

                # previous_row_enhanced = cv.Laplacian(previous_row_clahe, cv.CV_32F, ksize=3)

                # previous_row_32 = previous_row_enhanced.astype(np.float32)

                # print(prev_laplacian_row.size)
                # print(laplacian_row.size)


                shift_row, response = cv.phaseCorrelate(prev_first_in_row, first_in_row)
                # x_shift = int(round(shift[0]))
                # y_shift = int(round(shift[1]))
                # row_x_shift = int(round(shift_row[0]))
                row_y_shift = int(round(shift_row[1]))

                # raw_image = cv.copyMakeBorder(raw_image, 0, 0, abs(row_x_shift), 0, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                # trace_image = cv.copyMakeBorder(trace_image, 0, 0, abs(row_x_shift), 0, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                # laplacian_image = cv.copyMakeBorder(laplacian_image, 0, 0, abs(row_x_shift), 0, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

                prev_width = prev_laplacian_row.shape[1]
                current_width = laplacian_row.shape[1]

                if (prev_width < current_width):
                #     laplacian_row = cv.copyMakeBorder(laplacian_row, 0, 0, 0, prev_width - current_width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                # else:
                    prev_laplacian_row = cv.copyMakeBorder(prev_laplacian_row, 0, 0, 0, current_width - prev_width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                    raw_image = cv.copyMakeBorder(raw_image, 0, 0, 0, current_width - prev_width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                    trace_image = cv.copyMakeBorder(trace_image, 0, 0, 0, current_width - prev_width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                    laplacian_image = cv.copyMakeBorder(laplacian_image, 0, 0, 0, current_width - prev_width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

                raw_image = cv.copyMakeBorder(raw_image, abs(row_y_shift), 0, 0, 0, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                trace_image = cv.copyMakeBorder(trace_image, abs(row_y_shift), 0, 0, 0, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
                laplacian_image = cv.copyMakeBorder(laplacian_image, abs(row_y_shift), 0, 0, 0, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

            else:
                raw_image = raw_row.copy()
                trace_image = trace_row.copy()
                laplacian_image = laplacian_row.copy()

                # raw_row = np.zeros((h, w, 3), dtype = np.uint8)
                # trace_row = np.zeros((720, 1280), dtype = np.uint8)
                # laplacian_row = np.zeros((h, w), dtype = np.float32)
            x_shift = 0
            y_shift = 0
            y_shift0 = 0
            x_shift0 = 0

            # first_in_row = stitch_list[0].copy()

            stitch_list = []


            prev_laplacian_row = laplacian_row.copy()
            prev_first_in_row = first_in_row.copy()
            first_in_row = None

            raw_image[0:raw_row.shape[0], 0:raw_row.shape[1]] = raw_row.copy()
            laplacian_image[0:laplacian_row.shape[0], 0:laplacian_row.shape[1]] = laplacian_row.copy()

            trace_image[0:trace_row.shape[0], 0:trace_row.shape[1]] = cv.bitwise_or(trace_image[0:trace_row.shape[0], 0:trace_row.shape[1]], trace_row)

            row_y_shift0 += abs(row_y_shift)

            # x_shift0 += abs(x_shift)

            # canvas[abs(y_shift):abs(y_shift) + current.shape[0], abs(x_shift):abs(x_shift) + current.shape[1]] = transparent

            cv.namedWindow("Raw Image", cv.WINDOW_NORMAL)
            cv.resizeWindow("Raw Image", 1000, 1000)
            cv.imshow("Raw Image", raw_image)

            cv.namedWindow("Trace Image", cv.WINDOW_NORMAL)
            cv.resizeWindow("Trace Image", 1000, 1000)
            cv.imshow("Trace Image", trace_image)

            cv.namedWindow("Laplacian Image", cv.WINDOW_NORMAL)
            cv.resizeWindow("Laplacian Image", 1000, 1000)
            cv.imshow("Laplacian Image", laplacian_image)
        elif (command == "END_LAYER"):
            print("FINISHED")
            break # CHANGE THIS TO MAKE THE END LAYER ROUTINE NOT CANCEL THE RASPBERRY PI FOR FUTURE USE
        else:
            pass

        # ser.write(("hello from raspberry pi\0").encode("utf-8"))
        ser.write((command + "\0").encode("utf-8"))

        n = n + 1

    ref_grayscale = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)

    cv.imwrite("traceoutput.jpg", trace_image)
    cv.imwrite("rawoutput.jpg", raw_image)
        
    bilateral_blur_total = cv.bilateralFilter(raw_image, 21, 50, 200)  # Image to be aligned.
    gaussian_blur_total = cv.GaussianBlur(bilateral_blur_total, (5,5), 0)  # Image to be aligned.
    # gray = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2GRAY)
    # cv.imshow("Bilateral and Gaussian Blur", gaussian_blur)
    # cv.imshow("Grayed Blur", gray)
    # if (ref_frame is not None):
    #     current_hsv = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2HSV)
    #     ref_hsv = cv.cvtColor(ref_frame, cv.COLOR_BGR2HSV)
    #     current_h, current_s, current_v = cv.split(current_hsv)
    #     _, _, ref_v = cv.split(ref_hsv)
    #     # v_hist_match = match_histograms(current_v, ref_v, channel_axis=None).astype(np.uint8)
    #     current_hist = cv.calcHist([current_v], [0], None, [256], [0, 256])
    #     ref_hist = cv.calcHist([ref_v], [0], None, [256], [0, 256])
    #     current_cdf = current_hist.cumsum()
    #     ref_cdf = ref_hist.cumsum()
    #     current_cdf_norm = current_cdf / current_cdf[-1]
    #     ref_cdf_norm = ref_cdf / ref_cdf[-1]
    #     # 5. Create the Mapping Lookup Table (LUT)
    #     lut = np.zeros(256, dtype=np.uint8)
    #     ref_idx = 0
        
    #     for src_idx in range(256):
    #         while ref_idx < 255 and ref_cdf_norm[ref_idx] < current_cdf_norm[src_idx]:
    #             ref_idx += 1
    #         lut[src_idx] = ref_idx
            
    #     # 6. Apply LUT using OpenCV's native function (keeps layout continuous)
    #     matched_v = cv.LUT(current_v, lut)
    #     current_hsv = cv.merge([current_h, current_s, matched_v])
    #     current_hist_match = cv.cvtColor(current_hsv, cv.COLOR_HSV2BGR)

    #     cv.imshow("Matched", current_hist_match)
    # else:
    #     current_hist_match = gaussian_blur

    ### Extracting traces
    img_hsv_total = cv.cvtColor(gaussian_blur_total, cv.COLOR_BGR2HSV)

    # cv.imshow("HSV", img_hsv)

    trace_lower = np.array([90, 51, 204])
    trace_higher = np.array([105, 255, 255])

    img_threshold_total = cv.inRange(img_hsv_total, trace_lower, trace_higher)

    empty_lower = np.array([0, 0, 0]) # Maybe reduce this range
    empty_higher = np.array([45, 38, 153])

    dark_threshold_total = cv.inRange(img_hsv_total, empty_lower, empty_higher)

    true_trace_image = cv.subtract(img_threshold_total, dark_threshold_total)

    cv.namedWindow("TOTAL TRACE", cv.WINDOW_NORMAL)
    cv.resizeWindow("TOTAL TRACE", 1000, 1000)
    cv.imshow("TOTAL TRACE", true_trace_image)

    res = cv.matchTemplate(true_trace_image, ref_grayscale, cv.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv.minMaxLoc(res)

    trace_image_match = true_trace_image[max_loc[1]:max_loc[1] + ref_grayscale.shape[0], max_loc[0]: max_loc[0] + ref_grayscale.shape[1]].copy()

    cv.namedWindow("test", cv.WINDOW_NORMAL)
    cv.resizeWindow("test", 1000, 1000)
    cv.imshow("test", trace_image_match)

    # simple_contours, _ = cv.findContours(trace_image_match, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    white_points = cv.findNonZero(trace_image_match)

    # Get trace bounding rectangle measurements
    tbx, tby, tbw, tbh = cv.boundingRect(white_points)

    print(str(tbx) + " " + str(tby) + " " + str(tbw) + " " + str(tbh))

    # Get image dimensions
    imh, imw = trace_image_match.shape[:2]

    cropped_trace_image = trace_image_match[tby:tby+tbh, tbx:tbx+tbw].copy()
    
    cv.imshow("Cropped Trace Image", cropped_trace_image)

    adjusted_scale_float = scale * tbw / imw

    adjusted_scale = int(adjusted_scale_float)

    print(str(adjusted_scale))

    print("ADJUSTED SCALE " + str(adjusted_scale))

    Project(
        [
            GerberFile.from_file(
                'ref.gbr',
                FileTypeEnum.COPPER,
            ),
        #     GerberFile.from_file(
        #         'GerberFiles/soldermask_top.gbr',
        #         FileTypeEnum.MASK,
        #     ),
        #     GerberFile.from_file(
        #         'GerberFiles/solderpaste_top.gbr',
        #         FileTypeEnum.PASTE,
        #     ),
        #     GerberFile.from_file(
        #         'GerberFiles/silkscreen_top.gbr',
        #         FileTypeEnum.SILK,
        #     ),
        ],
    ).parse().render_raster("ref_adjusted.png", color_map = COLOR_MAP, dpmm=adjusted_scale)
    
    ref_adjusted = cv.imread("ref_adjusted.png")

    cv.imshow("Adjusted Reference", ref_adjusted)

    ref_scaled = cv.resize(ref_adjusted, (tbw, int(ref_adjusted.shape[0] * adjusted_scale_float / adjusted_scale)), interpolation=cv.INTER_CUBIC)

    cv.imshow("Scaled Reference", ref_scaled)

    ref_scaled_grayscale = cv.cvtColor(ref_scaled, cv.COLOR_BGR2GRAY)

    rsh, rsw = ref_scaled_grayscale.shape[:2]

    second_match = None

    if (rsh > tbh):
        matched_scaled = cv.matchTemplate(ref_scaled_grayscale, cropped_trace_image, cv.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv.minMaxLoc(matched_scaled)

        second_match = ref_scaled_grayscale[max_loc[1]:max_loc[1] + cropped_trace_image.shape[0], max_loc[0]: max_loc[0] + cropped_trace_image.shape[1]].copy()

        cv.imshow("Second Match", second_match)
    else:
        matched_scaled = cv.matchTemplate(cropped_trace_image, ref_scaled_grayscale, cv.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv.minMaxLoc(matched_scaled)

        second_match = cropped_trace_image[max_loc[1]:max_loc[1] + ref_scaled_grayscale.shape[0], max_loc[0]: max_loc[0] + ref_scaled_grayscale.shape[1]].copy()

        cv.imshow("Second Match", second_match)

    # simple_contours, _ = cv.findContours(trace_image_match, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    white_points = cv.findNonZero(second_match)

    # Get trace bounding rectangle measurements
    smx, smy, smw, smh = cv.boundingRect(white_points)

    second_crop = second_match[smy:smy+smh, smx:smx+smw].copy()
    
    cv.imshow("Second Crop", second_crop)

    adjusted_scale_float_2 = adjusted_scale_float * smw / tbw

    adjusted_scale_2 = int(adjusted_scale_float_2)

    print(str(adjusted_scale_2))

    print("ADJUSTED SCALE " + str(adjusted_scale_2))

    Project(
        [
            GerberFile.from_file(
                'ref.gbr',
                FileTypeEnum.COPPER,
            ),
        #     GerberFile.from_file(
        #         'GerberFiles/soldermask_top.gbr',
        #         FileTypeEnum.MASK,
        #     ),
        #     GerberFile.from_file(
        #         'GerberFiles/solderpaste_top.gbr',
        #         FileTypeEnum.PASTE,
        #     ),
        #     GerberFile.from_file(
        #         'GerberFiles/silkscreen_top.gbr',
        #         FileTypeEnum.SILK,
        #     ),
        ],
    ).parse().render_raster("ref_adjusted_2.png", color_map = COLOR_MAP, dpmm=adjusted_scale_2)
    
    ref_adjusted_2 = cv.imread("ref_adjusted_2.png")

    cv.imshow("Adjusted Reference 2", ref_adjusted_2)

    ref_scaled_2 = cv.resize(ref_adjusted_2, (smw, int(ref_adjusted_2.shape[0] * adjusted_scale_float_2 / adjusted_scale_2)), interpolation=cv.INTER_CUBIC)

    cv.imshow("Scaled Reference 2", ref_scaled_2)

    ref_scaled_grayscale_2 = cv.cvtColor(ref_scaled_2, cv.COLOR_BGR2GRAY)

    rsh2, rsw2 = ref_scaled_grayscale_2.shape[:2]

    third_match = None

    if (rsh2 > smh):
        matched_scaled = cv.matchTemplate(ref_scaled_grayscale_2, second_crop, cv.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv.minMaxLoc(matched_scaled)

        third_match = ref_scaled_grayscale_2[max_loc[1]:max_loc[1] + second_crop.shape[0], max_loc[0]: max_loc[0] + second_crop.shape[1]].copy()

        cv.imshow("Third Match", third_match)
    else:
        matched_scaled = cv.matchTemplate(second_crop, ref_scaled_grayscale_2, cv.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv.minMaxLoc(matched_scaled)

        third_match = second_crop[max_loc[1]:max_loc[1] + ref_scaled_grayscale_2.shape[0], max_loc[0]: max_loc[0] + ref_scaled_grayscale.shape[1]].copy()

        cv.imshow("Third Match", third_match)

    intended_contours_image = cv.cvtColor(ref_scaled_grayscale_2, cv.COLOR_GRAY2BGR)
    real_contours_image = cv.cvtColor(third_match, cv.COLOR_GRAY2BGR)

    intended_contours, _ = cv.findContours(ref_scaled_grayscale_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    real_contours, _ = cv.findContours(third_match, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.drawContours(intended_contours_image, intended_contours, -1, (0, 0, 255), 2)
    cv.drawContours(real_contours_image, real_contours, -1, (0, 0, 255), 2)

    unintended_ink = cv.subtract(third_match, ref_scaled_grayscale_2,)
    unintended_gaps = cv.subtract(ref_scaled_grayscale_2, third_match)

    cv.imshow("Unintended Ink", unintended_ink)
    cv.imshow("Unintended Gaps", unintended_gaps)

    # Scale project to be perfect pixel to pixel coordination

    # Get contours (FLAT AND NOT SIMPLIFYING)
    # Get paths for pads

    # detect scale and divisor from the actual gerber file
    gerber_file_raw = GerberFile.from_file("ref.gbr")
    source = gerber_file_raw.source_code
    scale = 25.4 if "%MOIN*%" in source else 1.0
    fmt_match = re.search(r'%FSLA[XY](\d)(\d)', source)
    divisor = 10 ** int(fmt_match.group(2)) if fmt_match else 1_000_000

    all_matches  = re.findall(r'X(-?\d+)Y(-?\d+)D0[123]', source)
    all_raw_x    = [int(x) / divisor * scale for x, y in all_matches]
    all_raw_y    = [int(y) / divisor * scale for x, y in all_matches]
    global_min_x = min(all_raw_x)
    global_min_y = min(all_raw_y)
    global_max_y = max(all_raw_y)

    pad_centers, _, _ = extract_coords("ref.gbr", global_min_x, global_max_y)

    test_points = []

    real_contours_image_copy = real_contours_image.copy()
    intended_contours_image_copy = intended_contours_image.copy()

    for center_x, center_y, size, shape in pad_centers:
        print(str(center_x) + " " + str(center_y))

        pixel_coord = (int(center_x * adjusted_scale_float_2), int(center_y * adjusted_scale_float_2))
        test_points.append(pixel_coord)

        print(pixel_coord)
    
        cv.circle(real_contours_image_copy, pixel_coord, radius=3, color=(0, 255, 0), thickness=4)
        cv.circle(intended_contours_image_copy, pixel_coord, radius=3, color=(0, 255, 0), thickness=4)

    cv.imshow("Intended Contours", intended_contours_image_copy)
    cv.imshow("Real Contours", real_contours_image_copy)
    
    n = len(test_points)



    for i in range(n):
        tp1 = test_points[i]
        tp1_x, tp1_y = tp1

        if (real_contours_image_copy[tp1_y][tp1_x] == 0):
            print("Test point at " + str(tp1) + " is not connected to the circuit")
            continue

        for j in range(i + 1, n):
            tp2 = test_points[j]
            tp2_x, tp2_y = tp2

            if (real_contours_image_copy[tp2_y][tp2_x] == 0):
                print("Test point at " + str(tp2) + " is not connected to the circuit")
                continue

    print(str(rsh2) + " " + str(rsw2))

    cv.waitKey(0)
    cv.waitKey(0)
    cv.waitKey(0)
    cv.waitKey(0)
    cv.waitKey(0)
    cv.waitKey(0)


        # pass
    # while (1):


        # if pressed == ord('r'):
        #     ref_frame = current_frame.copy()
        # elif pressed == ord('q'):
        #     crop = cv.selectROI("selectioj ROI", current_frame)
        # elif pressed == ord('n'):




        # elif pressed == ord('c'):



            # if (y_shift > 0):
            #     sum_y += y_shift
            # sum_x += x_shift

            # print(sum_y)
            # print(sum_x)

            # # sum_angles += angle
            # n_angles += 1

            # average_angle = 0

            # if (n_angles > 0):
            #     angle = 180 * math.atan(sum_y / sum_x) / math.pi
            #     # average_angle = sum_angles / n_angles

            

            # black = np.array([0, 0, 0])

            # white_background = cv.inRange(raw_row, black, black)

            # black_background = cv.bitwise_not(white_background)

            # contours, _ = cv.findContours(black_background, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # largest_contour = max(contours, key=cv.contourArea)

            # rect = cv.minAreaRect(largest_contour)

            # draw = raw_row.copy()

            # box = cv.boxPoints(rect)
            # box = np.int64(box)

            # cv.drawContours(draw, [box], 0, (255, 0, 0), 2)

            # cv.namedWindow("Boxed", cv.WINDOW_NORMAL)
            # cv.resizeWindow("Boxed", 1000, 1000)
            # cv.imshow("Boxed", draw)

            # draw2 = raw_row.copy()

            # cv.drawContours(draw2, largest_contour, -1, (255, 0, 0), 2)

            # cv.namedWindow("Contoured", cv.WINDOW_NORMAL)
            # cv.resizeWindow("Contoured", 1000, 1000)
            # cv.imshow("Contoured", draw2)

            # (center_x, center_y), (canvas_w, canvas_h), _ = rect

            # # print(angle)

            # # if angle < -45:
            # #     angle = 90 + angle

            # # print(average_angle)
            # # print("")

            # # (canvas_h, canvas_w) = canvas.shape[:2]
            # # center = (canvas_w // 2, canvas_h // 2)
            # # center = (center_x, center_y)
            # # M = cv.getRotationMatrix2D(center, - 2 * angle, 1.0)
            # # rotated = cv.warpAffine(canvas, M, (int(canvas_h), int(canvas_w)), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)

            # # cv.namedWindow("Rotated", cv.WINDOW_NORMAL)
            # # cv.resizeWindow("Rotated", 1000, 1000)
            # # cv.imshow("Rotated", rotated)
            
            # center = (center_x, center_y)
            # M = cv.getRotationMatrix2D(center, 0, 1.0)
            # rotated = cv.warpAffine(raw_row, M, (int(canvas_h), int(canvas_w)), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)

            # cv.namedWindow("Zoomed", cv.WINDOW_NORMAL)
            # cv.resizeWindow("Zoomed", 1000, 1000)
            # cv.imshow("Zoomed", rotated)

if __name__ == "__main__":
    main()
