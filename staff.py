import numpy as np

class Staff(object):
    def __init__(self, staff_matrix, staff_box, line_width, line_spacing, staff_img, clef="treble", time_signature="44", instrument=-1):
        self.clef = clef
        self.time_signature = time_signature
        self.instrument = instrument
        self.line_one = staff_matrix[0]
        self.line_two = staff_matrix[1]
        self.line_three = staff_matrix[2]
        self.line_four = staff_matrix[3]
        self.line_five = staff_matrix[4]
        self.staff_box = staff_box
        self.img = staff_img
        self.bars = []
        self.line_width = line_width
        self.line_spacing = line_spacing

    def setClef(self, clef):
        self.clef = clef

    def setTimeSignature(self, time):
        self.time_signature = time

    def setInstrument(self, instrument):
        self.instrument = instrument

    def addBar(self, bar):
        self.bars.append(bar)

    def getClef(self):
        return self.clef

    def getTimeSignature(self):
        return self.time_signature

    def getBox(self):
        return self.staff_box

    def getImage(self):
        return self.img

    def getLineWidth(self):
        return self.line_width

    def getLineSpacing(self):
        return self.line_spacing

    def getBars(self):
        return self.bars

    def getPitch(self, note_center_y):
        clef_info = {
            "treble": [("F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4"), (5,3), (4,2)],
            "bass": [("A3", "G3", "F3", "E3", "D3", "C3", "B2", "A2", "G2"), (3,5), (2,4)]
        }
        note_names = ["C", "D", "E", "F", "G", "A", "B"]

        #print("[getPitch] Using {} clef".format(self.clef))

        # Check within staff first
        if (note_center_y in self.line_one):
            return clef_info[self.clef][0][0]
        elif (note_center_y in list(range(self.line_one[-1] + 1, self.line_two[0]))):
            return clef_info[self.clef][0][1]
        elif (note_center_y in self.line_two):
            return clef_info[self.clef][0][2]
        elif (note_center_y in list(range(self.line_two[-1] + 1, self.line_three[0]))):
            return clef_info[self.clef][0][3]
        elif (note_center_y in self.line_three):
            return clef_info[self.clef][0][4]
        elif (note_center_y in list(range(self.line_three[-1] + 1, self.line_four[0]))):
            return clef_info[self.clef][0][5]
        elif (note_center_y in self.line_four):
            return clef_info[self.clef][0][6]
        elif (note_center_y in list(range(self.line_four[-1] + 1, self.line_five[0]))):
            return clef_info[self.clef][0][7]
        elif (note_center_y in self.line_five):
            return clef_info[self.clef][0][8]
        else:
            # print("[getPitch] Note was not within staff")
            if (note_center_y < self.line_one[0]):
                # print("[getPitch] Note above staff ")
                # Check above staff
                line_below = self.line_one
                current_line = [pixel - self.line_spacing for pixel in self.line_one] # Go to next line above
                octave = clef_info[self.clef][1][0]  # The octave number at line one
                note_index = clef_info[self.clef][1][1]  # Line one's pitch has this index in note_names

                while (current_line[0] > 0):
                    if (note_center_y in current_line):
                        # Grab note two places above
                        octave = octave + 1 if (note_index + 2 >= 7) else octave
                        note_index = (note_index + 2) % 7
                        return note_names[note_index] + str(octave)
                    elif (note_center_y in range(current_line[-1] + 1, line_below[0])):
                        # Grab note one place above
                        octave = octave + 1 if (note_index + 1 >= 7) else octave
                        note_index = (note_index + 1) % 7
                        return note_names[note_index] + str(octave)
                    else:
                        # Check next line above
                        octave = octave + 1 if (note_index + 2 >= 7) else octave
                        note_index = (note_index + 2) % 7
                        line_below = current_line.copy()
                        current_line = [pixel - self.line_spacing for pixel in current_line]

                assert False, "[ERROR] Note was above staff, but not found"
            elif (note_center_y > self.line_five[-1]):
                # print("[getPitch] Note below staff ")
                # Check below staff
                line_above = self.line_five
                current_line = [pixel + self.line_spacing for pixel in self.line_five]  # Go to next line above
                octave = clef_info[self.clef][2][0]  # The octave number at line five
                note_index = clef_info[self.clef][2][1]  # Line five's pitch has this index in note_names

                while (current_line[-1] < self.img.shape[0]):
                    if (note_center_y in current_line):
                        # Grab note two places above
                        octave = octave - 1 if (note_index - 2 <= 7) else octave
                        note_index = (note_index - 2) % 7
                        return note_names[note_index] + str(octave)
                    elif (note_center_y in range(line_above[-1] + 1, current_line[0])):
                        # Grab note one place above
                        octave = octave - 1 if (note_index - 1 >= 7) else octave
                        note_index = (note_index - 1) % 7
                        return note_names[note_index] + str(octave)
                    else:
                        # Check next line above
                        octave = octave - 1 if (note_index - 2 <= 7) else octave
                        note_index = (note_index - 2) % 7
                        line_above = current_line.copy()
                        current_line = [pixel + self.line_spacing for pixel in current_line]
                assert False, "[ERROR] Note was below staff, but not found"
            else:
                # Should not get here
                assert False, "[ERROR] Note was neither, within, above or below staff"