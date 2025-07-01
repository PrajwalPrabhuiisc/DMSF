import logging
import numpy as np
import mesa
import random
import traceback
from typing import Dict, Optional, List
from enums import AgentRole, EventType, OrgStructure, ReportingStructure
from data_classes import SituationalAwareness
from decision_model import DecisionModel

# Configure logging
logging.basicConfig(filename='agent_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ConstructionAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', role: AgentRole, pos: tuple):
        super().__init__(unique_id, model)
        self.role = role
        self.pos = pos
        self.sa = SituationalAwareness()
        self.workload = random.randint(1, 3)
        self.fatigue = random.uniform(0, 0.5)
        self.experience = random.uniform(0, 0.5) if role == AgentRole.WORKER else random.uniform(0.5, 1.0)
        self.risk_tolerance = random.uniform(0.3, 0.7) if role == AgentRole.WORKER else random.uniform(0.1, 0.5)
        self.stress = 0.5
        self.reports_sent = 0
        self.received_reports = []
        self.decision_model = DecisionModel()
        self.action_counts = {"ignore": 0, "report": 0, "act": 0, "escalate": 0}
        self.q_table = {
            EventType.HAZARD: {"ignore": 0, "report": 0, "act": 0, "escalate": 0},
            EventType.DELAY: {"ignore": 0, "report": 0, "act": 0, "escalate": 0},
            EventType.RESOURCE_SHORTAGE: {"ignore": 0, "report": 0, "act": 0, "escalate": 0}
        }
        self.last_event_action = None

        if role == AgentRole.REPORTER:
            self.detection_accuracy = self.model.reporter_detection
            self.reporting_probability = self.model.reporter_reporting
        elif role == AgentRole.WORKER:
            self.detection_accuracy = self.model.worker_detection
            self.reporting_probability = self.model.worker_reporting
        elif role == AgentRole.MANAGER:
            self.detection_accuracy = self.model.manager_detection
            self.reporting_probability = self.model.manager_reporting
        elif role == AgentRole.DIRECTOR:
            self.detection_accuracy = self.model.director_detection
            self.reporting_probability = self.model.director_reporting

    def observe_events(self, events: List[Optional[Dict]]) -> List[Dict]:
        observed_events = []
        for event in events:
            if not event:
                continue
            detection_modifier = 1.5 if self.role == AgentRole.REPORTER else 1.0
            if random.random() < 0.005 * (1 - self.experience * detection_modifier):  # Further reduced
                logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to observe event {event['type'].value}")
                print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to observe event {event['type'].value}")
                continue
            
            org_sa_modifier = (
                1.2 if self.model.org_structure == OrgStructure.FLAT else
                0.8 if self.model.org_structure == OrgStructure.HIERARCHICAL else
                1.0
            )
            sa_reduction = 0.7 if self.model.outcomes.safety_incidents > 5 else 1.0
            if self.role == AgentRole.REPORTER:
                self.sa.perception = min(self.sa.perception + self.detection_accuracy * 50 * org_sa_modifier * sa_reduction, 100)
                self.sa.comprehension = min(self.sa.comprehension + self.detection_accuracy * 30 * org_sa_modifier * sa_reduction, 100)
                self.sa.projection = min(self.sa.projection + self.detection_accuracy * 15 * org_sa_modifier * sa_reduction, 100)
            else:
                self.sa.perception = min(self.sa.perception + self.detection_accuracy * 40 * (1 + self.experience) * org_sa_modifier * sa_reduction, 100)
                self.sa.comprehension = min(self.sa.comprehension + self.detection_accuracy * 20 * (1 - self.workload / 5) * org_sa_modifier * sa_reduction, 100)
                self.sa.projection = min(self.sa.projection + self.detection_accuracy * 10 * (1 - self.fatigue) * org_sa_modifier * sa_reduction, 100)
            
            if self.received_reports and self.role != AgentRole.REPORTER:
                additional_aspects = sum(1 for r in self.received_reports if r["type"] == event["type"])
                self.sa.comprehension = min(self.sa.comprehension + additional_aspects * 10 * sa_reduction, 100)
            elif self.role == AgentRole.REPORTER:
                self.sa.comprehension = min(self.sa.comprehension + len(self.received_reports) * 15 * sa_reduction, 100)
            
            logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) observed event {event['type'].value}, SA updated: "
                         f"Perception={self.sa.perception:.2f}, Comprehension={self.sa.comprehension:.2f}, Projection={self.sa.projection:.2f}")
            print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) observed event {event['type'].value}, SA updated: "
                  f"Perception={self.sa.perception:.2f}, Comprehension={self.sa.comprehension:.2f}, Projection={self.sa.projection:.2f}")
            observed_events.append(event)
        return observed_events

    def decide_action(self, event: Optional[Dict], is_follow_up: bool = False) -> str:
        if not event:
            logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) no event, action: ignore")
            return "ignore"
        event_severity = event.get("severity", 1.0)
        org_modifier = (
            0.8 if self.model.org_structure == OrgStructure.FLAT else
            1.2 if self.model.org_structure == OrgStructure.HIERARCHICAL else
            1.0
        )
        time_pressure = min(self.workload / 5, 1.0)
        resource_availability = min(self.model.budget / 1000000, 1.0) * min(self.model.equipment / 500, 1.0)
        self.stress = 1.0 if len([e for e in self.model.get_events() if e["type"] == EventType.HAZARD]) > 1 else 0.5
        recent_hazard = 1.0 if any(r["type"] == EventType.HAZARD for r in self.received_reports[-3:]) else 0.0

        inputs = [self.workload * org_modifier, self.fatigue, event_severity, self.experience, time_pressure, resource_availability, self.risk_tolerance, self.stress, recent_hazard]
        try:
            action_probs = self.decision_model.predict_proba([inputs])[0]
        except Exception as e:
            logging.error(f"Error in predict_proba for Agent {self.unique_id}: {e}")
            print(f"Error in predict_proba for Agent {self.unique_id}: {e}")
            action_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        if len(action_probs) != 4:
            logging.warning(f"Step {self.model.schedule.steps}: Agent {self.unique_id} action_probs length {len(action_probs)} instead of 4")
            print(f"Warning: action_probs has length {len(action_probs)} instead of 4 for Agent {self.unique_id}")
            action_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        actions = ["ignore", "report", "act", "escalate"]
        action = np.random.choice(actions, p=action_probs)
        
        # Adjust action based on role and reporting structure
        if self.role == AgentRole.WORKER and action == "escalate" and self.model.reporting_structure == ReportingStructure.CENTRALIZED:
            action = "report"  # Workers report instead of escalate in centralized structures
        elif self.role == AgentRole.REPORTER and action == "act":
            action = "report"  # Reporters prefer reporting over acting
        elif self.role == AgentRole.DIRECTOR and action == "ignore":
            action = "escalate"  # Directors are less likely to ignore events
        
        # Update action counts and Q-table
        self.action_counts[action] += 1
        self.q_table[event["type"]][action] += 1
        self.last_event_action = action
        
        # Update stress and fatigue based on action
        if action == "act" or action == "escalate":
            self.fatigue = min(self.fatigue + 0.1, 1.0)
            self.stress = min(self.stress + 0.1, 1.0)
        elif action == "ignore":
            self.stress = max(self.stress - 0.05, 0.0)
        
        logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) decided action {action} for event {event['type'].value}")
        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) decided action {action} for event {event['type'].value}")
        return action
