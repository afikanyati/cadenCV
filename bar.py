class Bar(object):
    def __init__(self, key_signature="c"):
        self.primitives = []

    def addPrimitive(self, primitive):
        self.primitives.append(primitive)

    def getPrimitives(self):
        return self.primitives