from sklearn.neural_network import MLPClassifier
import numpy as np

class DecisionModel:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Two hidden layers with 64 and 32 neurons
            activation='relu',            # ReLU activation for non-linearity
            solver='adam',                # Adam optimizer for efficient training
            max_iter=1000,                # Maximum iterations for convergence
            learning_rate_init=0.001,     # Initial learning rate
            random_state=42,              # Reproducibility
            tol=1e-4,                     # Tolerance for early stopping
            n_iter_no_change=10           # Early stopping if no improvement
        )
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
