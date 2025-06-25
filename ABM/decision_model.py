import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Any, List

class MLDecisionSupport:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        self.training_data = []
        self.training_labels = []

    def extract_features(self, decision_context: List[tuple]) -> np.ndarray:
        features = []
        for agent, event, context in decision_context:
            feature_vector = [
                agent.awareness.perception,
                agent.awareness.comprehension,
                agent.awareness.projection,
                agent.workload,
                agent.fatigue,
                agent.experience_level,
                event['severity'].value if event else 0,
                context.get('urgency', 0.5),
                context.get('visibility', 1.0)
            ]
            features.append(feature_vector)
        return np.array(features)

    def train_model(self, agent_decisions: List[Dict], outcomes: List[str]):
        features = self.extract_features([(d['agent'], d['event'], d['context']) for d in agent_decisions])
        self.training_data.extend(features)
        self.training_labels.extend(outcomes)
        if len(self.training_data) >= 10:  # Minimum samples to train
            self.model.fit(self.training_data, self.training_labels)

    def predict_action(self, agent: Any, event: Dict, context: Dict) -> str:
        features = self.extract_features([(agent, event, context)])
        if len(self.training_data) >= 10:
            return self.model.predict(features)[0]
        return "ignore"  # Default action if model not trained