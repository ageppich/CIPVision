# TODO: Update the main function to your needs or remove it.

import cv2 as cv
import pygerber
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme, Project, FileTypeEnum, COLOR_MAP_T, DEFAULT_COLOR_MAP
import numpy as np
import time
from pygerber.common.rgba import RGBA
import math
from skimage.exposure import match_histograms

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
    clear_color=RGBA.from_rgba(1, 109, 90, 255),
    solid_color=RGBA.from_rgba(1, 150, 109, 255),
    clear_region_color=RGBA.from_rgba(1, 109, 90, 255),
    solid_region_color=RGBA.from_rgba(1, 150, 109, 255),
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

def main() -> None:

    # Render reference image
    Project(
        [
            GerberFile.from_file(
                'GerberFiles/copper_top.gbr',
                FileTypeEnum.COPPER,
            ),
            GerberFile.from_file(
                'GerberFiles/soldermask_top.gbr',
                FileTypeEnum.MASK,
            ),
            GerberFile.from_file(
                'GerberFiles/solderpaste_top.gbr',
                FileTypeEnum.PASTE,
            ),
            GerberFile.from_file(
                'GerberFiles/silkscreen_top.gbr',
                FileTypeEnum.SILK,
            ),
        ],
    ).parse().render_raster("ref.png", color_map = COLOR_MAP, dpmm=40)

    ### Set up video
    video = cv.VideoCapture(0)

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
    img2_color = cv.imread("ref.png")    # Reference image.

    img3 = img2_color.copy()

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

        line = ser.readline().decode('utf-8').strip()

        command = line.split()[0]

        print(command)

        if (command == "START_LAYER"):
            print("STARTED")
        elif (command == "CAPTURE"):

            if (first_row or do_capture):
                # if (first_in_row is None):
                for j in range(10):
                    video.grab()
                    # cv.waitKey(40)
                img_and = None

                for i in range(1):
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

                    trace_lower = np.array([76, 180, 140])
                    trace_higher = np.array([84, 255, 255])

                    img_threshold = cv.inRange(img_hsv, trace_lower, trace_higher)

                    empty_lower = np.array([82, 0, 102]) # Maybe reduce this range
                    empty_higher = np.array([87, 255, 150])

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

                # _, max_val, __dict__, max_loc = cv.minMaxLoc(res)

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

                    if (first_row):
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

            if (first_row == True):
                x_phase_average *= 2
                x_phase_average = int(x_phase_average)
                first_row = False

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
        else:
            pass

        # ser.write(("hello from raspberry pi\0").encode("utf-8"))
        ser.write((command + "\0").encode("utf-8"))

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

        n = n + 1

if __name__ == "__main__":
    main()
