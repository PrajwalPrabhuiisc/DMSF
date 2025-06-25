from dataclasses import dataclass

@dataclass
class SituationalAwareness:
    perception: float = 0.0
    comprehension: float = 0.0
    projection: float = 0.0

    def total_score(self) -> float:
        return (self.perception + self.comprehension + self.projection) / 3

@dataclass
class ProjectOutcomes:
    safety_incidents: int = 0
    incident_points: int = 0
    tasks_completed_on_time: int = 0
    total_tasks: int = 0
    cost_overruns: float = 0.0