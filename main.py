#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
6.819 Advances in Computer Vision
Bill Freeman, Antonio Torralba

Final Project - Optical Music Recognition Program
cadenCV
"""
__author__ = "Afika Nyati"
__email__ = "anyati@mit.edu"
__status__ = "Prototype"

# cv2.imshow("Input", no_staff_img)
# cv2.waitKey(0)

#-------------------------------------------------------------------------------
# Import Statements
#-------------------------------------------------------------------------------

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from copy import deepcopy
from PIL import Image

from best_match import match
from box import BoundingBox
from staff import Staff
from primitive import Primitive
from bar import Bar

#-------------------------------------------------------------------------------
# Template Paths
#-------------------------------------------------------------------------------


clef_paths = {
    "treble": [
        "resources/template/clef/treble_1.jpg",
        "resources/template/clef/treble_2.jpg"
    ],
    "bass": [
        "resources/template/clef/bass_1.jpg"
    ]
}

accidental_paths = {
    "sharp": [
        "resources/template/sharp-line.png",
        "resources/template/sharp-space.png"
    ],
    "flat": [
        "resources/template/flat-line.png",
        "resources/template/flat-space.png"
    ]
}

note_paths = {
    "quarter": [
        "resources/template/note/quarter.png",
        "resources/template/note/solid-note.png"
    ],
    "half": [
        "resources/template/note/half-space.png",
        "resources/template/note/half-note-line.png",
        "resources/template/note/half-line.png",
        "resources/template/note/half-note-space.png"
    ],
    "whole": [
        "resources/template/note/whole-space.png",
        "resources/template/note/whole-note-line.png",
        "resources/template/note/whole-line.png",
        "resources/template/note/whole-note-space.png"
    ]
}

rest_paths = {
    "eighth": ["resources/template/rest/eighth_rest.jpg"],
    "quarter": ["resources/template/rest/quarter_rest.jpg"],
    "half": ["resources/template/rest/half_rest_1.jpg",
            "resources/template/rest/half_rest_2.jpg"],
    "whole": ["resources/template/rest/whole_rest.jpg"]
}

barline_paths = ["resources/template/barline_1.jpg",
                 "resources/template/barline_2.jpg",
                 "resources/template/barline_3.jpg"]

#-------------------------------------------------------------------------------
# Template Images
#-------------------------------------------------------------------------------

# Clefs
clef_imgs = {
    "treble": [cv2.imread(clef_file, 0) for clef_file in clef_paths["treble"]],
    "bass": [cv2.imread(clef_file, 0) for clef_file in clef_paths["bass"]]
}

# Time Signatures
time_imgs = {
    "common": [cv2.imread(time, 0) for time in ["resources/template/time/common.jpg"]],
    "44": [cv2.imread(time, 0) for time in ["resources/template/time/44.jpg"]],
    "34": [cv2.imread(time, 0) for time in ["resources/template/time/34.jpg"]],
    "24": [cv2.imread(time, 0) for time in ["resources/template/time/24.jpg"]],
    "68": [cv2.imread(time, 0) for time in ["resources/template/time/68.jpg"]]
}

# Accidentals
sharp_imgs = [cv2.imread(sharp_files, 0) for sharp_files in accidental_paths["sharp"]]
flat_imgs = [cv2.imread(flat_file, 0) for flat_file in accidental_paths["flat"]]

# Notes
quarter_note_imgs = [cv2.imread(quarter, 0) for quarter in note_paths["quarter"]]
half_note_imgs = [cv2.imread(half, 0) for half in note_paths["half"]]
whole_note_imgs = [cv2.imread(whole, 0) for whole in note_paths['whole']]

# Rests
eighth_rest_imgs = [cv2.imread(eighth, 0) for eighth in rest_paths["eighth"]]
quarter_rest_imgs = [cv2.imread(quarter, 0) for quarter in rest_paths["quarter"]]
half_rest_imgs = [cv2.imread(half, 0) for half in rest_paths["half"]]
whole_rest_imgs = [cv2.imread(whole, 0) for whole in rest_paths['whole']]

# Bar line
bar_imgs = [cv2.imread(barline, 0) for barline in barline_paths]


#-------------------------------------------------------------------------------
# Template Thresholds
#-------------------------------------------------------------------------------

# Clefs
clef_lower, clef_upper, clef_thresh = 50, 150, 0.88

# Time
time_lower, time_upper, time_thresh = 50, 150, 0.88

# Accidentals
sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
flat_lower, flat_upper, flat_thresh = 50, 150, 0.77

# Notes
quarter_note_lower, quarter_note_upper, quarter_note_thresh = 50, 150, 0.70
half_note_lower, half_note_upper, half_note_thresh = 50, 150, 0.70
whole_note_lower, whole_note_upper, whole_note_thresh = 50, 150, 0.70

# Rests
eighth_rest_lower, eighth_rest_upper, eighth_rest_thresh = 50, 150, 0.70
quarter_rest_lower, quarter_rest_upper, quarter_rest_thresh = 50, 150, 0.70
half_rest_lower, half_rest_upper, half_rest_thresh = 50, 150, 0.80
whole_rest_lower, whole_rest_upper, whole_rest_thresh = 50, 150, 0.80

# Bar line
bar_lower, bar_upper, bar_thresh = 50, 150, 0.85

#-------------------------------------------------------------------------------
# General Functions
#-------------------------------------------------------------------------------

pitch_to_MIDI = {
    "C8": 108,
    "B7": 107,
    "A#7": 106,
    "A7": 105,
    "G#7": 104,
    "G7": 103,
    "F#7": 102,
    "F7": 101,
    "E7": 100,
    "D#7": 99,
    "D7": 98,
    "C#7": 97,
    "C7": 96,
    "B6": 95,
    "A#6": 94,
    "A6": 93,
    "G#6": 92,
    "G6": 91,
    "F#6": 90,
    "F6": 89,
    "E6": 88,
    "D#6": 87,
    "D6": 86,
    "C#6": 85,
    "C6": 84,
    "B5": 83,
    "A#5": 82,
    "A5": 81,
    "G#5": 80,
    "G5": 79,
    "F#5": 78,
    "F5": 77,
    "E5": 76,
    "D#5": 75,
    "D5": 74,
    "C#5": 73,
    "C5": 72,
    "B4": 71,
    "A#4": 70,
    "A4": 69,
    "G#4": 68,
    "G4": 67,
    "F#4": 66,
    "F4": 65,
    "E4": 64,
    "D#4": 63,
    "D4": 62,
    "C#4": 61,
    "C4": 60,
    "B3": 59,
    "A#3": 58,
    "A3": 57,
    "G#3": 56,
    "G3": 55,
    "F#3": 54,
    "F3": 53,
    "E3": 52,
    "D#3": 51,
    "D3": 50,
    "C#3": 49,
    "C3": 48,
    "B2": 47,
    "A#2": 46,
    "A2": 45,
    "G#2": 44,
    "G2": 43,
    "F#2": 42,
    "F2": 41,
    "E2": 40,
    "D#2": 39,
    "D2": 38,
    "C#2": 37,
    "C2": 36,
    "B1": 35,
    "A#1": 34,
    "A1": 33,
    "G#1": 32,
    "G1": 31,
    "F#1": 30,
    "F1": 29,
    "E1": 28,
    "D#1": 27,
    "D1": 26,
    "C#1": 25,
    "C1": 24,
    "B0": 23,
    "A#0": 22,
    "A0": 21
}

#-------------------------------------------------------------------------------
# General Functions
#-------------------------------------------------------------------------------

def deskew(img):
    skew_img = cv2.bitwise_not(img)  # Invert image

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(skew_img > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return angle, rotated


def get_ref_lengths(img):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    rle_image_white_runs = []  # Cumulative white run list
    rle_image_black_runs = []  # Cumulative black run list
    sum_all_consec_runs = []  # Cumulative consecutive black white runs

    for i in range(num_cols):
        col = img[:, i]
        rle_col = []
        rle_white_runs = []
        rle_black_runs = []
        run_val = 0  # (The number of consecutive pixels of same value)
        run_type = col[0]  # Should be 255 (white) initially
        for j in range(num_rows):
            if (col[j] == run_type):
                # increment run length
                run_val += 1
            else:
                # add previous run length to rle encoding
                rle_col.append(run_val)
                if (run_type == 0):
                    rle_black_runs.append(run_val)
                else:
                    rle_white_runs.append(run_val)

                # alternate run type
                run_type = col[j]
                # increment run_val for new value
                run_val = 1

        # add final run length to encoding
        rle_col.append(run_val)
        if (run_type == 0):
            rle_black_runs.append(run_val)
        else:
            rle_white_runs.append(run_val)

        # Calculate sum of consecutive vertical runs
        sum_rle_col = [sum(rle_col[i: i + 2]) for i in range(len(rle_col))]

        # Add to column accumulation list
        rle_image_white_runs.extend(rle_white_runs)
        rle_image_black_runs.extend(rle_black_runs)
        sum_all_consec_runs.extend(sum_rle_col)

    white_runs = Counter(rle_image_white_runs)
    black_runs = Counter(rle_image_black_runs)
    black_white_sum = Counter(sum_all_consec_runs)

    line_spacing = white_runs.most_common(1)[0][0]
    line_width = black_runs.most_common(1)[0][0]
    width_spacing_sum = black_white_sum.most_common(1)[0][0]

    assert (line_spacing + line_width == width_spacing_sum), "Estimated Line Thickness + Spacing doesn't correspond with Most Common Sum "

    return line_width, line_spacing


def find_staffline_rows(img, line_width, line_spacing):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    row_black_pixel_histogram = []

    # Determine number of black pixels in each row
    for i in range(num_rows):
        row = img[i]
        num_black_pixels = 0
        for j in range(len(row)):
            if (row[j] == 0):
                num_black_pixels += 1

        row_black_pixel_histogram.append(num_black_pixels)

    # plt.bar(np.arange(num_rows), row_black_pixel_histogram)
    # plt.show()

    all_staff_row_indices = []
    num_stafflines = 5
    threshold = 0.4
    staff_length = num_stafflines * (line_width + line_spacing) - line_spacing
    iter_range = num_rows - staff_length + 1

    # Find stafflines by finding sum of rows that occur according to
    # staffline width and staffline space which contain as many black pixels
    # as a thresholded value (based of width of page)
    #
    # Filter out using condition that all lines in staff
    # should be above a threshold of black pixels
    current_row = 0
    while (current_row < iter_range):
        staff_lines = [row_black_pixel_histogram[j: j + line_width] for j in
                       range(current_row, current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
                             line_width + line_spacing)]
        pixel_avg = sum(sum(staff_lines, [])) / (num_stafflines * line_width)

        for line in staff_lines:
            if (sum(line) / line_width < threshold * num_cols):
                current_row += 1
                break
        else:
            staff_row_indices = [list(range(j, j + line_width)) for j in
                                 range(current_row,
                                       current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
                                       line_width + line_spacing)]
            all_staff_row_indices.append(staff_row_indices)
            current_row = current_row + staff_length

    return all_staff_row_indices


def find_staffline_columns(img, all_staffline_vertical_indices, line_width, line_spacing):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    # Create list of tuples of the form (column index, number of occurrences of width_spacing_sum)
    all_staff_extremes = []

    # Find start of staff for every staff in piece
    for i in range(len(all_staffline_vertical_indices)):
        begin_list = [] # Stores possible beginning column indices for staff
        end_list = []   # Stores possible end column indices for staff
        begin = 0
        end = num_cols - 1

        # Find staff beginning
        for j in range(num_cols // 2):
            first_staff_rows_isolated = img[all_staffline_vertical_indices[i][0][0]:all_staffline_vertical_indices[i][4][
                line_width - 1], j]
            num_black_pixels = len(list(filter(lambda x: x == 0, first_staff_rows_isolated)))

            if (num_black_pixels == 0):
                begin_list.append(j)

        # Find maximum column that has no black pixels in staff window
        list.sort(begin_list, reverse=True)
        begin = begin_list[0]

        # Find staff beginning
        for j in range(num_cols // 2, num_cols):
            first_staff_rows_isolated = img[all_staffline_vertical_indices[i][0][0]:all_staffline_vertical_indices[i][4][
                line_width - 1], j]
            num_black_pixels = len(list(filter(lambda x: x == 0, first_staff_rows_isolated)))

            if (num_black_pixels == 0):
                end_list.append(j)

        # Find maximum column that has no black pixels in staff window
        list.sort(end_list)
        end = end_list[0]

        staff_extremes = (begin, end)
        all_staff_extremes.append(staff_extremes)

    return all_staff_extremes


def remove_stafflines(img, all_staffline_vertical_indices):
    no_staff_img = deepcopy(img)
    for staff in all_staffline_vertical_indices:
        for line in staff:
            for row in line:
                # Remove top and bottom line to be sure
                no_staff_img[row - 1, :] = 255
                no_staff_img[row, :] = 255
                no_staff_img[row + 1, :] = 255

    return no_staff_img


def open_file(path):
    img = Image.open(path)
    img.show()


def locate_templates(img, templates, start, stop, threshold):
    locations, scale = match(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append([BoundingBox(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    return img_locations


def merge_boxes(boxes, threshold):
    filtered_boxes = []
    while len(boxes) > 0:
        r = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while (merged):
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(r) > threshold:
                    r = r.merge(boxes.pop(i))
                    merged = True
                elif boxes[i].distance(r) > r.w / 2 + boxes[i].w / 2:
                    break
                else:
                    i += 1
        filtered_boxes.append(r)
    return filtered_boxes


if __name__ == "__main__":

    #-------------------------------------------------------------------------------
    # Image Preprocessing (Blurring, Noise Removal, Binarization, Deskewing)
    #-------------------------------------------------------------------------------

    # Noise Removal: https://docs.opencv.org/3.3.1/d5/d69/tutorial_py_non_local_means.html
    # Deskewing: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # Binarization + Blurring (Otsu): https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html

    # ============ Read Image ============
    # img_file = sys.argv[1:][0]
    img_file ='resources/samples/dream.png'
    img = cv2.imread(img_file, 0)

    # ============ Noise Removal ============

    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # ============ Binarization ============

    # Global Thresholding
    # retval, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's Thresholding
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # ============ Deskewing ============

    # angle, img = deskew(img)
    # print("[INFO] Deskew Angle: {:.3f}".format(angle))
    # cv2.imshow("Input", img)
    # cv2.waitKey(0)

    # ============ Reference Lengths ============
    # Reference lengths staff line thickness (staffline_height)
    # and vertical line distance within the same staff (staffspace_height)
    # computed, providing the basic scale for relative size comparisons

    # Use run-length encoding on columns to estimate staffline height and staffspace height

    line_width, line_spacing = get_ref_lengths(img)

    print("[INFO] Staff line Width: ", line_width)
    print("[INFO] Staff line Spacing: ", line_spacing)

    #-------------------------------------------------------------------------------
    # Staff Line Detection
    #-------------------------------------------------------------------------------

    # In practice, several horizontal projections on images with slightly different
    # rotation angles are computed to deal with not completely horizontal staff lines.
    # The projection with the highest local maxima is then chosen.

    # ============ Find Staff Line Rows ============

    all_staffline_vertical_indices = find_staffline_rows(img, line_width, line_spacing)
    print("[INFO] Found ", len(all_staffline_vertical_indices), " sets of staff lines")

    # ============ Find Staff Line Columns ============

    # Find column with largest index that has no black pixels

    all_staffline_horizontal_indices = find_staffline_columns(img, all_staffline_vertical_indices, line_width, line_spacing)
    print("[INFO] Found all staff line horizontal extremes")

    # ============ Show Detected Staffs ============
    staffs = []
    half_dist_between_staffs = (all_staffline_vertical_indices[1][0][0] - all_staffline_vertical_indices[0][4][line_width - 1])//2

    for i in range(len(all_staffline_vertical_indices)):
        # Create Bounding Box
        x = all_staffline_horizontal_indices[i][0]
        y = all_staffline_vertical_indices[i][0][0]
        width = all_staffline_horizontal_indices[i][1] - x
        height = all_staffline_vertical_indices[i][4][line_width - 1] - y
        staff_box = BoundingBox(x, y, width, height)

        # Create Cropped Staff Image
        staff_img = img[max(0, y - half_dist_between_staffs): min(y+ height + half_dist_between_staffs, img.shape[0] - 1), x:x+width]
        staff = Staff(all_staffline_vertical_indices[i], staff_box, staff_img)
        staffs.append(staff)

    staff_boxes_img = img.copy()
    staff_boxes_img = cv2.cvtColor(staff_boxes_img, cv2.COLOR_GRAY2RGB)
    red = (0, 0, 255)
    box_thickness = 2
    for staff in staffs:
        box = staff.getBox()
        box.draw(staff_boxes_img, red, box_thickness)

    cv2.imwrite('staff_boxes_img.png', staff_boxes_img)
    open_file('staff_boxes_img.png')
    print("[INFO] Outputting image showing detected staffs")

    #-------------------------------------------------------------------------------
    # Symbol Segmentation, Object Recognition, and Semantic Reconstruction
    #-------------------------------------------------------------------------------

    # The score is then divided into regions of interest to localize and isolate the musical primitives.
    # Music score is analyzed and split by staves
    # Primitive symbols extracted

    # Find all primitives on each stave first
    # then move from left to right and create structure

    # ============ Determine Clef, Time Signature ============

    # for i in range(len(staffs)):
    #     red = (0, 0, 255)
    #     box_thickness = 2
    #
    #     # ------- Clef -------
    #     for clef in clef_imgs:
    #         print("[INFO] Matching {} clef template on staff".format(clef), i + 1)
    #         clef_boxes = locate_templates(staffs[i].getImage(), clef_imgs[clef], clef_lower, clef_upper, clef_thresh)
    #
    #         print("[INFO] Merging {} clef template results...".format(clef))
    #         clef_boxes = merge_boxes([j for i in clef_boxes for j in i], 0.5)
    #
    #         if (len(clef_boxes) == 1):
    #             print("[INFO] Clef Found")
    #             staffs[i].setClef(clef)
    #
    #             print("[INFO] Displaying Matching Results on staff", i + 1)
    #             clef_boxes_img = staffs[i].getImage()
    #             clef_boxes_img = clef_boxes_img.copy()
    #             clef_boxes_img = cv2.cvtColor(clef_boxes_img, cv2.COLOR_GRAY2RGB)
    #             for boxes in clef_boxes:
    #                 boxes.draw(clef_boxes_img, red, box_thickness)
    #             cv2.imwrite("{}_clef_boxes_img_{}.png".format(clef,i + 1), clef_boxes_img)
    #             open_file("{}_clef_boxes_img_{}.png".format(clef,i + 1))
    #             break
    #
    #         print("[INFO] {} clef not found on staff".format(clef), i+1)
    #
    #     else:
    #         # A clef should always be found
    #         print("[INFO] No clef found on staff", i+1)
    #
    #     # # ------- Time -------
    #     for time in time_imgs:
    #         print("[INFO] Matching {} time signature template on staff".format(time), i + 1)
    #         time_boxes = locate_templates(staffs[i].getImage(), time_imgs[time], time_lower, time_upper, time_thresh)
    #
    #         print("[INFO] Merging {} time signature template results...".format(time))
    #
    #         time_boxes = merge_boxes([j for i in time_boxes for j in i], 0.5)
    #
    #         if (len(time_boxes) == 1):
    #             print("[INFO] Time Signature Found")
    #             staffs[i].setTimeSignature(time)
    #
    #             print("[INFO] Displaying Matching Results on staff", i + 1)
    #             time_boxes_img = staffs[i].getImage()
    #             time_boxes_img = time_boxes_img.copy()
    #             time_boxes_img = cv2.cvtColor(time_boxes_img, cv2.COLOR_GRAY2RGB)
    #
    #             for boxes in time_boxes:
    #                 boxes.draw(time_boxes_img, red, box_thickness)
    #             cv2.imwrite("{}_time_boxes_img_{}.png".format(time, i + 1), time_boxes_img)
    #             open_file("{}_time_boxes_img_{}.png".format(time, i + 1))
    #             break
    #
    #         elif (len(time_boxes) == 0 and i > 0):
    #             # Take time signature of previous staff
    #             previousTime = staffs[i-1].getTimeSignature()
    #             staffs[i].setTimeSignature(previousTime)
    #             print("[INFO] No time signature found on staff", i + 1, ". Using time signature from previous staff line.")
    #             break
    #     else:
    #         print("[INFO] No time signature available for staff", i + 1)

    # ============ Find Primitives ============

    # always assert that notes in a bar equal duration dictated by time signature
    for i in range(len(staffs)):
        print("[INFO] Finding Primitives on Staff ", i+1)
        staff_primitives = []
        staff_img = staffs[i].getImage()
        red = (0, 0, 255)
        box_thickness = 2

        # print("[INFO] Matching sharp accidental template...")
        # sharp_boxes = locate_templates(staff_img, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)
        #
        # print("[INFO] Merging sharp accidental template results...")
        # sharp_boxes = merge_boxes([j for i in sharp_boxes for j in i], 0.5)
        # sharp_boxes_img = staffs[i].getImage()
        # sharp_boxes_img = sharp_boxes_img.copy()
        # sharp_boxes_img = cv2.cvtColor(sharp_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in sharp_boxes:
        #     box.draw(sharp_boxes_img, red, box_thickness)
        #     sharp = Primitive("sharp", 0, box)
        #     staff_primitives.append(sharp)
        # cv2.imwrite('sharp_boxes_img.png', sharp_boxes_img)
        # open_file('sharp_boxes_img.png')
        #
        # print("[INFO] Matching flat accidental template...")
        # flat_boxes = locate_templates(staff_img, flat_imgs, flat_lower, flat_upper, flat_thresh)
        #
        # print("[INFO] Merging flat accidental template results...")
        # flat_boxes = merge_boxes([j for i in flat_boxes for j in i], 0.5)
        # flat_boxes_img = staffs[i].getImage()
        # flat_boxes_img = flat_boxes_img.copy()
        # flat_boxes_img = cv2.cvtColor(flat_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in flat_boxes:
        #     box.draw(flat_boxes_img, red, box_thickness)
        #     flat = Primitive("flat", 0, box)
        #     staff_primitives.append(flat)
        # cv2.imwrite('flat_boxes_img.png', flat_boxes_img)
        # open_file('flat_boxes_img.png')
        #
        # print("[INFO] Matching quarter note template...")
        # quarter_boxes = locate_templates(staff_img, quarter_note_imgs, quarter_note_lower, quarter_note_upper, quarter_note_thresh)
        #
        # print("[INFO] Merging quarter note template results...")
        # quarter_boxes = merge_boxes([j for i in quarter_boxes for j in i], 0.5)
        # quarter_boxes_img = staffs[i].getImage()
        # quarter_boxes_img = quarter_boxes_img.copy()
        # quarter_boxes_img = cv2.cvtColor(quarter_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in quarter_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(quarter_boxes_img, red, box_thickness)
        #     quarter = Primitive("note", 1, box)     # Add pitch
        #     staff_primitives.append(quarter)
        # cv2.imwrite('quarter_boxes_img.png', quarter_boxes_img)
        # open_file('quarter_boxes_img.png')
        #
        # print("[INFO] Matching half note template...")
        # half_boxes = locate_templates(staff_img, half_note_imgs, half_note_lower, half_note_upper, half_note_thresh)
        #
        # print("[INFO] Merging half note template results...")
        # half_boxes = merge_boxes([j for i in half_boxes for j in i], 0.5)
        # half_boxes_img = staffs[i].getImage()
        # half_boxes_img = half_boxes_img.copy()
        # half_boxes_img = cv2.cvtColor(half_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in half_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(half_boxes_img, red, box_thickness)
        #     half = Primitive("note", 2, box)  # Add pitch
        #     staff_primitives.append(half)
        # cv2.imwrite('half_boxes_img.png', half_boxes_img)
        # open_file('half_boxes_img.png')
        #
        # print("[INFO] Matching whole note template...")
        # whole_boxes = locate_templates(staff_img, whole_note_imgs, whole_note_lower, whole_note_upper, whole_note_thresh)
        #
        # print("[INFO] Merging whole note template results...")
        # whole_boxes = merge_boxes([j for i in whole_boxes for j in i], 0.5)
        # whole_boxes_img = staffs[i].getImage()
        # whole_boxes_img = whole_boxes_img.copy()
        # whole_boxes_img = cv2.cvtColor(whole_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in whole_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(whole_boxes_img, red, box_thickness)
        #     whole = Primitive("note", 4, box)  # Add pitch
        #     staff_primitives.append(whole)
        # cv2.imwrite('whole_boxes_img.png', whole_boxes_img)
        # open_file('whole_boxes_img.png')

        # print("[INFO] Matching eighth rest template...")
        # eighth_boxes = locate_templates(staff_img, eighth_rest_imgs, eighth_rest_lower, eighth_rest_upper, eighth_rest_thresh)
        #
        # print("[INFO] Merging eighth rest template results...")
        # eighth_boxes = merge_boxes([j for i in eighth_boxes for j in i], 0.5)
        # eighth_boxes_img = staffs[i].getImage()
        # eighth_boxes_img = eighth_boxes_img.copy()
        # eighth_boxes_img = cv2.cvtColor(eighth_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in eighth_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(eighth_boxes_img, red, box_thickness)
        #     eighth = Primitive("rest", 0.5, box)  # Add pitch
        #     staff_primitives.append(eighth)
        # cv2.imwrite('eighth_boxes_img.png', eighth_boxes_img)
        # open_file('eighth_boxes_img.png')
        #
        # print("[INFO] Matching quarter rest template...")
        # quarter_boxes = locate_templates(staff_img, quarter_rest_imgs, quarter_rest_lower, quarter_rest_upper, quarter_rest_thresh)
        #
        # print("[INFO] Merging quarter rest template results...")
        # quarter_boxes = merge_boxes([j for i in quarter_boxes for j in i], 0.5)
        # quarter_boxes_img = staffs[i].getImage()
        # quarter_boxes_img = quarter_boxes_img.copy()
        # quarter_boxes_img = cv2.cvtColor(quarter_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in quarter_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(quarter_boxes_img, red, box_thickness)
        #     quarter = Primitive("rest", 1, box)  # Add pitch
        #     staff_primitives.append(quarter)
        # cv2.imwrite('quarter_boxes_img.png', quarter_boxes_img)
        # open_file('quarter_boxes_img.png')
        #
        # print("[INFO] Matching half rest template...")
        # half_boxes = locate_templates(staff_img, half_rest_imgs, half_rest_lower, half_rest_upper, half_rest_thresh)
        #
        # print("[INFO] Merging half rest template results...")
        # half_boxes = merge_boxes([j for i in half_boxes for j in i], 0.5)
        # half_boxes_img = staffs[i].getImage()
        # half_boxes_img = half_boxes_img.copy()
        # half_boxes_img = cv2.cvtColor(half_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in half_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(half_boxes_img, red, box_thickness)
        #     half = Primitive("rest", 2, box)  # Add pitch
        #     staff_primitives.append(half)
        # cv2.imwrite('half_boxes_img.jpg', half_boxes_img)
        # open_file('half_boxes_img.jpg')

        # print("[INFO] Matching whole rest template...")
        # whole_boxes = locate_templates(staff_img, whole_rest_imgs, whole_rest_lower, whole_rest_upper, whole_rest_thresh)
        #
        # print("[INFO] Merging whole rest template results...")
        # whole_boxes = merge_boxes([j for i in whole_boxes for j in i], 0.5)
        # whole_boxes_img = staffs[i].getImage()
        # whole_boxes_img = whole_boxes_img.copy()
        # whole_boxes_img = cv2.cvtColor(whole_boxes_img, cv2.COLOR_GRAY2RGB)
        # for box in whole_boxes:
        #     # Determine Pitch
        #     # Map Pitch to MIDI Note
        #     box.draw(whole_boxes_img, red, box_thickness)
        #     whole = Primitive("rest", 4, box)  # Add pitch
        #     staff_primitives.append(whole)
        # cv2.imwrite('whole_boxes_img.jpg', whole_boxes_img)
        # open_file('whole_boxes_img.jpg')

        print("[INFO] Matching bar line template...")
        bar_boxes = locate_templates(staff_img, bar_imgs, bar_lower, bar_upper,
                                       bar_thresh)

        print("[INFO] Merging bar line template results...")
        bar_boxes = merge_boxes([j for i in bar_boxes for j in i], 0.5)
        bar_boxes_img = staffs[i].getImage()
        bar_boxes_img = bar_boxes_img.copy()
        bar_boxes_img = cv2.cvtColor(bar_boxes_img, cv2.COLOR_GRAY2RGB)
        for box in bar_boxes:
            # Determine Pitch
            # Map Pitch to MIDI Note
            box.draw(bar_boxes_img, red, box_thickness)
            line = Primitive("line", 0, box)
            staff_primitives.append(line)
        cv2.imwrite('bar_boxes_img.jpg', bar_boxes_img)
        open_file('bar_boxes_img.jpg')






