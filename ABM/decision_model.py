from sklearn.linear_model import LogisticRegression
import numpy as np

class DecisionModel:
    def __init__(self):
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self._train_model()

    def _train_model(self):
        # Training data: [workload, fatigue, event_severity, experience, time_pressure, resource_availability, risk_tolerance, stress, recent_hazard]
        X = np.array([
            [1, 0.2, 1.0, 0.5, 0.2, 0.8, 0.5, 0.5, 0.0],  # HAZARD, low workload
            [3, 0.7, 1.0, 0.3, 0.6, 0.5, 0.7, 1.0, 1.0],  # HAZARD, high workload
            [2, 0.5, 0.7, 0.6, 0.4, 0.7, 0.4, 0.5, 0.0],  # DELAY, moderate
            [4, 0.8, 0.7, 0.2, 0.8, 0.3, 0.6, 1.0, 0.0],  # DELAY, high fatigue
            [2, 0.3, 0.5, 0.7, 0.4, 0.9, 0.3, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [5, 0.9, 0.5, 0.1, 1.0, 0.2, 0.8, 1.0, 0.0],  # RESOURCE_SHORTAGE, high stress
            [1, 0.1, 1.0, 0.9, 0.2, 0.9, 0.2, 0.5, 1.0],  # HAZARD, experienced
            [3, 0.6, 0.7, 0.4, 0.6, 0.6, 0.5, 0.5, 0.0],  # DELAY, balanced
            [2, 0.4, 0.5, 0.8, 0.4, 0.8, 0.4, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [4, 0.7, 1.0, 0.3, 0.8, 0.4, 0.6, 1.0, 1.0],  # HAZARD, high risk
            # Additional samples for better coverage
            [2, 0.5, 1.0, 0.6, 0.5, 0.7, 0.5, 0.5, 1.0],  # HAZARD
            [3, 0.6, 0.7, 0.5, 0.6, 0.6, 0.4, 0.5, 0.0],  # DELAY
            [1, 0.3, 0.5, 0.7, 0.3, 0.9, 0.3, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [5, 0.8, 1.0, 0.2, 0.9, 0.3, 0.7, 1.0, 1.0],  # HAZARD
            [2, 0.4, 0.7, 0.6, 0.4, 0.8, 0.4, 0.5, 0.0],  # DELAY
            [3, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [1, 0.2, 1.0, 0.8, 0.2, 0.9, 0.3, 0.5, 1.0],  # HAZARD
            [4, 0.7, 0.7, 0.3, 0.7, 0.4, 0.6, 1.0, 0.0],  # DELAY
            [2, 0.3, 0.5, 0.7, 0.3, 0.8, 0.4, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [3, 0.6, 1.0, 0.4, 0.6, 0.5, 0.5, 1.0, 1.0],  # HAZARD
            [2, 0.4, 0.7, 0.6, 0.4, 0.7, 0.4, 0.5, 0.0],  # DELAY
            [1, 0.2, 0.5, 0.8, 0.2, 0.9, 0.3, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [5, 0.9, 1.0, 0.1, 0.9, 0.2, 0.8, 1.0, 1.0],  # HAZARD
            [3, 0.5, 0.7, 0.5, 0.5, 0.6, 0.5, 0.5, 0.0],  # DELAY
            [2, 0.3, 0.5, 0.7, 0.3, 0.8, 0.4, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [4, 0.7, 1.0, 0.3, 0.7, 0.4, 0.6, 1.0, 1.0],  # HAZARD
            [2, 0.4, 0.7, 0.6, 0.4, 0.7, 0.4, 0.5, 0.0],  # DELAY
            [1, 0.2, 0.5, 0.8, 0.2, 0.9, 0.3, 0.5, 0.0],  # RESOURCE_SHORTAGE
            [3, 0.6, 1.0, 0.4, 0.6, 0.5, 0.5, 1.0, 1.0],  # HAZARD
            [2, 0.4, 0.7, 0.6, 0.4, 0.7, 0.4, 0.5, 0.0]   # DELAY
        ])
        # Updated: Include all 4 actions
        y = np.array([
            'ignore', 'act', 'report', 'ignore', 'act', 'report', 'act', 'report', 'act', 'ignore',
            'escalate', 'act', 'report', 'ignore', 'act', 'report', 'act', 'ignore', 'report', 'escalate',
            'act', 'report', 'ignore', 'act', 'report', 'escalate', 'act', 'report', 'ignore', 'act'
        ])
        self.model.fit(X, y)

    def predict_proba(self, inputs):
        # Ensure probabilities for all 4 actions
        return self.model.predict_proba(inputs)
