class Primitive(object):
    def __init__(self, primitive, duration, box, pitch=-1):
        self.pitch = pitch
        self.duration = duration
        self.primitive = primitive
        self.box = box

    def addFlat(self):
        if (self.primitive == "note"):
            self.pitch -= 1

    def addSharp(self):
        if (self.primitive == "note"):
            self.pitch += 1
    
    def getPrimitive(self):
        return self.primitive

    def getPitch(self):
        return self.pitch

    def getDuration(self):
        return self.duration

    def getBox(self):
        return self.box
