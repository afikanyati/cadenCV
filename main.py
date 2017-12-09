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

from best_fit import fit
from rectangle import Rectangle
from staff import Staff

#-------------------------------------------------------------------------------
# File Paths
#-------------------------------------------------------------------------------

clef_paths = [
    "resources/template/clefs/treble.jpg",
    "resources/template/clefs/bass.jpg"
]

time_paths = [
    "resources/template/time/common.jpg",
    "resources/template/time/44.jpg",
    "resources/template/time/34.jpg",
    "resources/template/time/24.jpg",
    "resources/template/time/68.jpg"
]

key_paths = {
    "treble": [
        "resources/template/key/treble/a_flat_treble.jpg",
        "resources/template/key/treble/a_treble.jpg",
        "resources/template/key/treble/b_flat_treble.jpg",
        "resources/template/key/treble/b_treble.jpg",
        "resources/template/key/treble/c_flat_treble.jpg",
        "resources/template/key/treble/c_flat_treble.jpg"
        "resources/template/key/treble/c#_treble.jpg",
        "resources/template/key/treble/d_flat_treble.jpg",
        "resources/template/key/treble/d_treble.jpg",
        "resources/template/key/treble/e_flat_treble.jpg",
        "resources/template/key/treble/e_treble.jpg",
        "resources/template/key/treble/f_treble.jpg",
        "resources/template/key/treble/f#_treble.jpg",
        "resources/template/key/treble/g_flat_treble.jpg",
        "resources/template/key/treble/g_treble.jpg"
    ],
    "bass": [
        "resources/template/key/bass/a_flat_treble.jpg",
        "resources/template/key/bass/a_treble.jpg",
        "resources/template/key/bass/b_flat_treble.jpg",
        "resources/template/key/bass/b_treble.jpg",
        "resources/template/key/bass/c_flat_treble.jpg",
        "resources/template/key/bass/c_flat_treble.jpg"
        "resources/template/key/bass/c#_treble.jpg",
        "resources/template/key/bass/d_flat_treble.jpg",
        "resources/template/key/bass/d_treble.jpg",
        "resources/template/key/bass/e_flat_treble.jpg",
        "resources/template/key/bass/e_treble.jpg",
        "resources/template/key/bass/f_treble.jpg",
        "resources/template/key/bass/f#_treble.jpg",
        "resources/template/key/bass/g_flat_treble.jpg",
        "resources/template/key/bass/g_treble.jpg"
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
    "eighth": "resources/template/rest/eighth_rest.jpg",
    "quarter": "resources/template/rest/quarter_rest.jpg",
    "half": "resources/template/rest/half_rest.jpg",
    "whole": "resources/template/rest/whole_rest.jpg"
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

barline_path = "resources/template/barline.png"

clef_imgs = [cv2.imread(clef_file, 0) for clef_file in clef_paths]

clef_lower, clef_upper, clef_thresh = 50, 150, 0.77
sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
flat_lower, flat_upper, flat_thresh = 50, 150, 0.77
quarter_lower, quarter_upper, quarter_thresh = 50, 150, 0.70
half_lower, half_upper, half_thresh = 50, 150, 0.70
whole_lower, whole_upper, whole_thresh = 50, 150, 0.70


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


def locate_images(img, templates, width, height, threshold):
    locations = fit(img, templates, width, height, threshold)
    img_locations = []
    for i in range(len(templates)):
        img_locations.append([Rectangle(pt[0], pt[1], width, height) for pt in zip(*locations[i][::-1])])

    return img_locations


def merge_recs(recs, threshold):
    filtered_recs = []
    while len(recs) > 0:
        r = recs.pop(0)
        recs.sort(key=lambda rec: rec.distance(r))
        merged = True
        while(merged):
            merged = False
            i = 0
            for _ in range(len(recs)):
                if r.overlap(recs[i]) > threshold or recs[i].overlap(r) > threshold:
                    r = r.merge(recs.pop(i))
                    merged = True
                elif recs[i].distance(r) > r.w/2 + recs[i].w/2:
                    break
                else:
                    i += 1
        filtered_recs.append(r)
    return filtered_recs


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
    staff_recs = []
    isolated_staff_images = []  # Cropped Staff Images
    half_dist_between_staffs = (all_staffline_vertical_indices[1][0][0] - all_staffline_vertical_indices[0][4][line_width - 1])//2

    for i in range(len(all_staffline_vertical_indices)):
        # Create Bounding Rectangle
        x = all_staffline_horizontal_indices[i][0]
        y = all_staffline_vertical_indices[i][0][0]
        width = all_staffline_horizontal_indices[i][1] - x
        height = all_staffline_vertical_indices[i][4][line_width - 1] - y
        staff_rec = Rectangle(x, y, width, height)
        staff_recs.append(staff_rec)

        # Create Cropped Staff Image
        staff_img = img[max(0, y - half_dist_between_staffs): min(y+ height + half_dist_between_staffs, img.shape[0] - 1), x:x+width]
        isolated_staff_images.append(staff_img)

    staff_recs_img = img.copy()
    staff_recs_img = cv2.cvtColor(staff_recs_img, cv2.COLOR_GRAY2RGB)
    red = (0, 0, 255)
    rec_thickness = 2
    for r in staff_recs:
        r.draw(staff_recs_img, red, rec_thickness)

    cv2.imwrite('staff_recs_img.png', staff_recs_img)
    open_file('staff_recs_img.png')
    print("[INFO] Output image showing detected staffs")




    #-------------------------------------------------------------------------------
    # Symbol Segmentation
    #-------------------------------------------------------------------------------

    # The score is then divided into regions of interest to localize and isolate the musical primitives.
    # Music score is analyzed and split by staves
    # Primitive symbols extracted

    # Find all primitives on each stave first
    # then move from left to right and create structure

    # ============ Determine Clefs ============
    clef_height = 119
    height = (199/119) * staff_recs[0].get_height()
    print("clef height: ", height)
    width = height/clef_imgs[0].shape[1] * clef_imgs[0].shape[1]
    print("clef width: ", width)

    print("[INFO] Matching clef image...")
    clef_recs = locate_images(isolated_staff_images[0], [clef_imgs[0]], width, height, clef_thresh)

    print("Merging clef image results...")
    clef_recs = merge_recs([j for i in clef_recs for j in i], 0.5)
    clef_recs_img = isolated_staff_images[0].copy()
    for r in clef_recs:
        r.draw(clef_recs_img, (0, 0, 255), 2)
    cv2.imwrite('clef_recs_img.png', clef_recs_img)
    open_file('clef_recs_img.png')


    # ============ Determine Time Signatures ============



    # ============ Determine Key Signatures ============



    # ============ Remove Staff Lines ============

    # The most simple line removal algorithm removes the line piecewise — following it along
    # and replacing the black line pixels with white pixels unless there is evidence of
    # an object on either side of the line

    # no_staff_img = remove_stafflines(img, all_staffline_vertical_indices)
    # cv2.imshow("Input", no_staff_img)
    # cv2.waitKey(0)

    # ============ Find Notes ============

    # always assert that notes in a bar equal duration dictated by time signature



    #-------------------------------------------------------------------------------
    # Object Recognition
    #-------------------------------------------------------------------------------

    # Use one of the following:
    # projection profiles
    # template matching: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#py-template-matching
    # Perceptron


    # Classifiers are built by taking a set of labeled examples of
    # music symbols and randomly split them into training and test
    # sets. The best parameterization for each model is normally
    # found based on a cross validation scheme conducted on the
    # training set.

    #-------------------------------------------------------------------------------
    # Semantic Reconstruction
    #-------------------------------------------------------------------------------

    # reconstruct the musical semantics from previously recognized graphical
    # primitives and store the information in a suitable data structure.

    # Grammar?
    # build the semantic reconstruction on a set of rules and heuristics

    # transformation of semantically recognized scores in a coding format that is able to model and
    # store music information

    # correct for key signature




