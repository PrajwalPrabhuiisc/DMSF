import numpy as np
from sklearn.linear_model import LogisticRegression

class DecisionModel:
    def __init__(self):
        X = np.array([
            [1, 0.2, 0.1], [3, 0.5, 0.3], [5, 0.8, 0.5],
            [2, 0.3, 0.7], [3, 0.4, 0.8], [4, 0.6, 0.9],
            [1, 0.1, 1.0], [2, 0.2, 0.9], [3, 0.3, 1.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3])
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.model.fit(X, y)

    def predict(self, workload: float, fatigue: float, event_severity: float) -> str:
        action_probs = self.model.predict_proba([[workload, fatigue, event_severity]])[0]
        actions = ["ignore", "report", "act", "escalate"]
        return actions[np.argmax(action_probs)]