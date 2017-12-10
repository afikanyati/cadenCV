class Bar(object):
    def __init__(self, key_signature="c"):
        self.key_signature = key_signature
        self.primitives = []

    def setKeySignature(self, key):
        self.key_signature = key

    def addPrimitive(self, primitive):
        self.primitive.append(primitive)