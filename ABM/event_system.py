import random
import numpy as np
from typing import List, Dict, Tuple
from enums import EventType, EventSeverity

class EventCategory:
    def __init__(self, event_type: EventType, severity: EventSeverity, duration: int, spread_pattern: str, step: int):
        self.type = event_type
        self.severity = severity
        self.duration = duration
        self.spread_pattern = spread_pattern
        self.step = step
        self.cascading_probability = self.calculate_cascade_probability()

    def calculate_cascade_probability(self) -> float:
        return 0.1 * self.severity.value

class SpatialEventModel:
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
        self.risk_zones = self.initialize_risk_zones()
        self.event_clustering_tendency = 0.3

    def initialize_risk_zones(self) -> Dict[Tuple[int, int], float]:
        risk_zones = {}
        high_risk = [(0, 0), (19, 19), (10, 10)]
        moderate_risk = [(i, j) for i in [0, 19] for j in range(20)] + [(i, j) for j in [0, 19] for i in range(20)]
        low_risk = [(i, j) for i in range(5, 15) for j in range(5, 15)]
        for pos in high_risk:
            risk_zones[pos] = 2.0
        for pos in moderate_risk:
            risk_zones[pos] = 1.5 if pos not in high_risk else 2.0
        for pos in low_risk:
            risk_zones[pos] = 1.0 if pos not in high_risk and pos not in moderate_risk else risk_zones.get(pos, 1.0)
        return risk_zones

    def select_event_location(self, event_type: EventType, existing_events: List[Dict], current_step: int) -> Tuple[int, int]:
        location_probabilities = np.ones((self.grid_size[0], self.grid_size[1]))
        for (x, y), risk_level in self.risk_zones.items():
            location_probabilities[x][y] *= risk_level
        if existing_events and random.random() < self.event_clustering_tendency:
            recent_events = [e for e in existing_events if e['step'] > (current_step - 10)]
            for event in recent_events:
                x, y = event['location']
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                            location_probabilities[nx][ny] *= 1.5
        total_prob = np.sum(location_probabilities)
        location_probabilities /= total_prob
        flat_indices = np.arange(self.grid_size[0] * self.grid_size[1])
        chosen_index = np.random.choice(flat_indices, p=location_probabilities.flatten())
        return divmod(chosen_index, self.grid_size[1])

class EventGenerator:
    def __init__(self, model: any):
        self.model = model
        self.base_probabilities = {
            EventType.HAZARD: 0.05,
            EventType.DELAY: 0.15,
            EventType.RESOURCE: 0.03,
            EventType.QUALITY: 0.08,
            EventType.ENVIRONMENTAL: 0.02,
            EventType.EQUIPMENT: 0.04,
            EventType.PERSONNEL: 0.03
        }
        self.event_history = []
        self.spatial_model = SpatialEventModel((20, 20))

    def generate_events(self, current_step: int) -> List[Dict]:
        generated_events = []
        for event_type in EventType:
            modified_prob = self.calculate_modified_probability(event_type, current_step)
            if random.random() < modified_prob:
                event = self.create_event(event_type, current_step)
                generated_events.append(event)
                self.event_history.append(event)
                cascade_events = self.calculate_cascade_effects(event)
                generated_events.extend(cascade_events)
                self.event_history.extend(cascade_events)
        return generated_events

    def calculate_modified_probability(self, event_type: EventType, current_step: int) -> float:
        base_prob = self.base_probabilities[event_type]
        cascade_modifier = 1.0 + 0.1 * len([e for e in self.event_history if e['step'] > current_step - 5])
        temporal_modifier = 1.0  # Simplified
        spatial_modifier = 1.0  # Simplified
        org_state_modifier = 1.0 - 0.05 * (self.model.outcomes.safety_incidents / max(1, current_step))
        environmental_modifier = 1.0  # Simplified
        total_modifier = cascade_modifier * temporal_modifier * spatial_modifier * org_state_modifier * environmental_modifier
        return min(0.8, base_prob * total_modifier)

    def create_event(self, event_type: EventType, step: int, severity: EventSeverity = None, caused_by: Dict = None) -> Dict:
        severity = severity or random.choices(
            [EventSeverity.LOW, EventSeverity.MEDIUM, EventSeverity.HIGH, EventSeverity.CRITICAL],
            weights=[0.4, 0.3, 0.2, 0.1])[0]
        duration = random.randint(1, 5)
        spread_pattern = random.choice(['local', 'radial', 'random'])
        location = self.spatial_model.select_event_location(event_type, self.event_history, step)
        event = EventCategory(event_type, severity, duration, spread_pattern, step)
        return {
            'type': event_type,
            'severity': severity,
            'duration': duration,
            'spread_pattern': spread_pattern,
            'location': location,
            'step': step,
            'cascading_probability': event.cascading_probability,
            'caused_by': caused_by
        }

    def calculate_cascade_effects(self, primary_event: Dict) -> List[Dict]:
        cascade_events = []
        cascade_rules = {
            EventType.HAZARD: {
                EventType.DELAY: 0.6,
                EventType.RESOURCE: 0.3,
                EventType.PERSONNEL: 0.2
            },
            EventType.EQUIPMENT: {
                EventType.DELAY: 0.8,
                EventType.HAZARD: 0.4,
                EventType.RESOURCE: 0.5
            },
            EventType.DELAY: {
                EventType.RESOURCE: 0.4,
                EventType.PERSONNEL: 0.3,
                EventType.QUALITY: 0.2
            }
        }
        if primary_event['type'] in cascade_rules:
            for secondary_type, probability in cascade_rules[primary_event['type']].items():
                severity_multiplier = primary_event['severity'].value / 4.0
                cascade_prob = probability * severity_multiplier
                if random.random() < cascade_prob:
                    secondary_severity = EventSeverity(max(1, primary_event['severity'].value - 1))
                    secondary_event = self.create_event(
                        secondary_type,
                        primary_event['step'] + 1,
                        severity=secondary_severity,
                        caused_by=primary_event
                    )
                    cascade_events.append(secondary_event)
        return cascade_events