from AutoLib import sigmoid, randWeight

class LayerAuto:

    def __init__(self, layerIn, layerOut):
        self.ouput = None;
        self.z = None;
        self.input = None;
        self.weight = randWeight(layerOut, layerIn)
        self.bias = randWeight(layerOut, 1)
        self.input = None
        self.z = None;
        self.output = None;

    def setInput(self, x):
        self.input = x

    def forward(self):
        self.z = self.weight * self.input + self.bias
        self.output = sigmoid(self.z)

        return self.output

    def back
