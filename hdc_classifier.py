from sklearn.linear_model import LogisticRegression

def train_hdc(features, labels):
    """
    Treina o classificador HDC usando regressão logística.
    """
    model = LogisticRegression()
    model.fit(features, labels)
    return model

def evaluate_hdc(model, features, labels):
    """
    Avalia o classificador HDC.
    """
    accuracy = model.score(features, labels)
    return accuracy