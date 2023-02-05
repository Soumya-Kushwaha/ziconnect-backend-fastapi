class ConnectivityModel:
    def __init__(self):
        self.r = 2.0
        self.s = 0.1
    def predict(self, x: float) -> float:
        y = self.r * x + self.s
        return y