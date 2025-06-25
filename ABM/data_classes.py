from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SituationalAwareness:
    perception: float = 0.0
    comprehension: float = 0.0
    projection: float = 0.0
    temporal_depth: int = 10
    prediction_horizon: int = 5
    safety_awareness: float = 0.0
    operational_awareness: float = 0.0
    resource_awareness: float = 0.0
    perception_confidence: float = 0.0
    comprehension_confidence: float = 0.0
    projection_confidence: float = 0.0
    information_sources: List[Dict] = None
    last_update_step: int = 0
    update_frequency: int = 0

    def __post_init__(self):
        if self.information_sources is None:
            self.information_sources = []

    def total_score(self) -> float:
        return (self.perception + self.comprehension + self.projection) / 3

@dataclass
class ProjectOutcomes:
    safety_incidents: int = 0
    incident_points: int = 0
    tasks_completed_on_time: int = 0
    total_tasks: int = 0
    cost_overruns: float = 0.0