from sklearn.linear_model import LogisticRegression
import numpy as np

class DecisionModel:
    def __init__(self):
        # Inputs: [workload, fatigue, event_severity, experience, time_pressure, resource_availability, risk_tolerance, stress, recent_hazard]
        X = np.array([
            [1, 0.1, 0.5, 0.2, 0.2, 0.8, 0.3, 0.5, 0.0],  # Low severity, low exp -> report
            [1, 0.1, 1.0, 0.8, 0.2, 0.8, 0.3, 0.5, 1.0],  # HAZARD, high exp, resources -> act
            [3, 0.5, 0.5, 0.3, 0.6, 0.2, 0.5, 0.5, 0.0],  # Low resources, high workload -> report
            [3, 0.5, 1.0, 0.3, 0.6, 0.2, 0.5, 1.0, 1.0],  # HAZARD, low exp, stress -> escalate
            [2, 0.3, 0.7, 0.5, 0.4, 0.7, 0.4, 0.5, 0.0],  # DELAY, decent resources -> act
            [2, 0.2, 0.5, 0.4, 0.4, 0.6, 0.3, 0.5, 0.0],  # Low severity, moderate exp -> report
            [1, 0.4, 0.7, 0.6, 0.2, 0.8, 0.3, 0.5, 0.0],  # DELAY, high exp -> act
            [3, 0.3, 1.0, 0.2, 0.6, 0.3, 0.5, 0.5, 1.0],  # HAZARD, low resources -> escalate
            [2, 0.5, 0.7, 0.5, 0.4, 0.7, 0.4, 0.5, 0.0],  # DELAY, follow-up -> act
            [1, 0.2, 0.7, 0.8, 0.2, 0.8, 0.3, 0.5, 0.0],  # DELAY, high exp -> act
            [1, 0.1, 1.0, 0.7, 0.2, 0.9, 0.2, 0.5, 1.0],  # HAZARD, high exp, resources -> act
            [3, 0.6, 0.5, 0.3, 0.8, 0.2, 0.6, 1.0, 0.0],  # High stress, low resources -> report
            [2, 0.3, 0.7, 0.4, 0.4, 0.6, 0.5, 0.5, 0.0],  # DELAY, moderate conditions -> act
            [1, 0.2, 1.0, 0.9, 0.2, 0.8, 0.3, 0.5, 1.0],  # HAZARD, high exp -> act
            [3, 0.5, 0.7, 0.3, 0.6, 0.3, 0.6, 0.5, 0.0],  # DELAY, low resources -> report
            [2, 0.4, 1.0, 0.6, 0.4, 0.7, 0.4, 1.0, 1.0],  # HAZARD, stress -> act
            [1, 0.1, 0.7, 0.7, 0.2, 0.8, 0.3, 0.5, 0.0],  # DELAY, good conditions -> act
            [3, 0.6, 0.5, 0.2, 0.8, 0.2, 0.5, 1.0, 0.0],  # High workload, stress -> report
            [2, 0.3, 1.0, 0.5, 0.4, 0.6, 0.4, 0.5, 1.0],  # HAZARD, recent hazard -> act
            [1, 0.2, 0.7, 0.6, 0.2, 0.7, 0.3, 0.5, 0.0],  # DELAY, follow-up -> act
            [1, 0.1, 1.0, 0.8, 0.2, 0.9, 0.2, 0.5, 1.0],  # HAZARD, high exp -> act
            [3, 0.5, 0.7, 0.3, 0.6, 0.3, 0.6, 0.5, 0.0],  # DELAY, low resources -> report
            [2, 0.4, 1.0, 0.6, 0.4, 0.7, 0.4, 1.0, 1.0],  # HAZARD, stress -> act
            [1, 0.1, 0.7, 0.7, 0.2, 0.8, 0.3, 0.5, 0.0],  # DELAY, good conditions -> act
            [3, 0.6, 0.5, 0.2, 0.8, 0.2, 0.5, 1.0, 0.0],  # High workload, stress -> report
            [2, 0.3, 1.0, 0.5, 0.4, 0.6, 0.4, 0.5, 1.0],  # HAZARD, recent hazard -> act
            [1, 0.2, 0.7, 0.6, 0.2, 0.7, 0.3, 0.5, 0.0],  # DELAY, follow-up -> act
            [1, 0.1, 1.0, 0.8, 0.2, 0.9, 0.2, 0.5, 1.0],  # HAZARD, high exp -> act
            [3, 0.5, 0.7, 0.3, 0.6, 0.3, 0.6, 0.5, 0.0],  # DELAY, low resources -> report
            [2, 0.4, 1.0, 0.6, 0.4, 0.7, 0.4, 1.0, 1.0],  # HAZARD, stress -> act
        ])
        y = np.array([
            1, 2, 1, 3, 2, 1, 2, 3, 2, 2,
            2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
            2, 1, 2, 2, 1, 2, 2, 2, 1, 2
        ])  # 0=ignore, 1=report, 2=act, 3=escalate; no ignore for HAZARD
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
