class Staff(object):
    def __init__(self, staff_matrix, staff_box, staff_img, clef=-1, key_signature="C", time_signature="4/4", instrument=-1):
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

    def setClef(self, clef):
        self.clef = clef

    def setTimeSignature(self, time):
        self.time_signature = time

    def setKeySignature(self, key):
        self.key_signature = key

    def setInstrument(self, instrument):
        self.instrument = instrument

    def addBar(self, bar):
        self.bars.append(bar)

    def getClef(self):
        return self.clef

    def getTimeSignature(self):
        return self.time_signature

    def getKeySignature(self):
        return self.key_signature

    def getBox(self):
        return self.staff_box

    def getImage(self):
        return self.img

    def getPitch(self, pixel_range):
        # above, on, below a line note
        # a function
        return