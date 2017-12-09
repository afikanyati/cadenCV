class Staff(object):
    def __init__(self, clef, staff_matrix, key_signature="C", time_signature="4/4", instrument=-1):
        self.clef = clef
        self.key_signature = key_signature
        self.time_signature = time_signature
        self.instrument = instrument
        self.line_one = staff_matrix[0]
        self.line_two = staff_matrix[1]
        self.line_three = staff_matrix[2]
        self.line_four = staff_matrix[3]
        self.line_five = staff_matrix[4]




    # above, on, below a line
    # a function
