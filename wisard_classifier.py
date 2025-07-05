import numpy as np

class WiSARD:
    def __init__(self, address_size):
        self.address_size = address_size
        self.memory = {}

    def train(self, features, labels):
        """
        Treina o classificador WiSARD.
        """
        for feature, label in zip(features, labels):
            if label not in self.memory:
                self.memory[label] = set()
            self.memory[label].add(tuple(feature))

    def classify(self, features):
        """
        Classifica os dados de entrada.
        """
        predictions = []
        for feature in features:
            scores = {label: sum(1 for addr in self.memory[label] if addr == tuple(feature))
                      for label in self.memory}
            predictions.append(max(scores, key=scores.get))
        return predictions

def evaluate_wisard(features, labels, address_size=10):
    """
    Treina e avalia o classificador WiSARD.
    """
    wisard = WiSARD(address_size)
    wisard.train(features, labels)
    predictions = wisard.classify(features)
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    return accuracy