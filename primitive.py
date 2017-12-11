class Primitive(object):
    def __init__(self, primitive, duration, box, pitch=-1):
        self.pitch = pitch
        self.duration = duration
        self.primitive = primitive
        self.box = box

    def setPitch(self, pitch):
        self.pitch = pitch

    def setDuration(self, duration):
        self.duration = duration
    
    def getPrimitive(self):
        return self.primitive

    def getPitch(self):
        return self.pitch

    def getDuration(self):
        return self.duration

    def getBox(self):
        return self.box
