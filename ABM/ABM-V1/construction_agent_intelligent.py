import random
import logging
from typing import Dict, List, Optional
from enums import AgentRole, EventType, ActionType, OrgStructure, ReportingStructure
from data_classes import SituationalAwareness
from construction_model import ConstructionModel
from decision_model import DecisionModel
import numpy as np

logging.basicConfig(filename='simulation_agent.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ConstructionAgent:
    def __init__(self, unique_id: int, model: ConstructionModel, role: AgentRole, pos: tuple):
        self.unique_id = unique_id
        self.model = model
        self.role = role
        self.pos = pos
        self.awareness = SituationalAwareness()
        self.reports_sent = 0
        self.reports_received: List[Dict] = []
        self.actions_taken = {action: 0 for action in ActionType}
        self.workload = random.uniform(0, 5)
        self.fatigue = random.uniform(0, 1)
        self.experience = random.uniform(0, 1)
        self.risk_tolerance = random.uniform(0, 1)
        self.detection_accuracy = {
            AgentRole.WORKER: model.worker_detection,
            AgentRole.MANAGER: model.manager_detection,
            AgentRole.REPORTER: model.reporter_detection,
            AgentRole.DIRECTOR: model.director_detection
        }[role]
        self.reporting_probability = {
            AgentRole.WORKER: model.worker_reporting,
            AgentRole.MANAGER: model.manager_reporting,
            AgentRole.REPORTER: model.reporter_reporting,
            AgentRole.DIRECTOR: model.director_reporting
        }[role]
        self.detection_modifier = 1.5 if role == AgentRole.REPORTER else 1.0
        # Additional features for MLP
        self.time_pressure = random.uniform(0, 1)
        self.stress = random.uniform(0, 1)
        self.recent_hazard = 0.0  # Tracks recent hazard exposure
        # Initialize DecisionModel
        self.decision_model = DecisionModel()

    def step(self):
        """Agent's step method called by the scheduler each model step."""
        for event in self.model.get_events():
            self.observe_event(event)

        for report in self.reports_received[:]:
            event_data = report.get("event")
            if event_data and not report.get("acted_on", False):
                action = self.decide_action(event_data, is_follow_up=True)
                if action == ActionType.ACT and self.model.budget >= 1000:
                    self.execute_action(event_data, action)
                    report["acted_on"] = True
                    logging.info(
                        f"Agent {self.unique_id} ({self.role.value}) acted on report with severity {event_data.get('severity', 0):.2f}, "
                        f"action={action} at step {self.model.schedule.steps}"
                    )
                elif action == ActionType.ESCALATE and self.model.budget >= 1500:
                    self.execute_action(event_data, action)
                    report["acted_on"] = True
                    logging.info(
                        f"Agent {self.unique_id} ({self.role.value}) escalated report for event {event_data.get('type')}, "
                        f"at step {self.model.schedule.steps}"
                    )
                self.reports_received.remove(report)

    def observe_event(self, event: Optional[Dict]) -> bool:
        if event is None:
            return False

        org_sa_modifier = 1.2 if self.model.org_structure == OrgStructure.FLAT else 1.0 if self.model.org_structure == OrgStructure.FUNCTIONAL else 0.8
        sa_reduction = 0.7 if self.model.outcomes.safety_incidents > 5 else 1.0

        # Update recent_hazard for MLP input
        if event.get("type") == EventType.HAZARD.value:
            self.recent_hazard = 1.0
        else:
            self.recent_hazard *= 0.9  # Decay over time

        if random.random() < self.detection_accuracy * self.detection_modifier:
            # Stochastic SA update
            severity = event.get("severity", 0.5)
            noise_std = 0.2 + 0.3 * self.fatigue
            random_noise = random.gauss(0, noise_std)
            
            if self.role == AgentRole.REPORTER:
                self.awareness.perception = min(
                    self.awareness.perception + self.detection_accuracy * 50 * (1 + random_noise) * severity * (1 - self.fatigue) * org_sa_modifier * sa_reduction, 100)
                self.awareness.comprehension = min(
                    self.awareness.comprehension + self.detection_accuracy * 30 * (1 + random_noise) * severity * (1 - self.workload / 5) * org_sa_modifier * sa_reduction, 100)
                self.awareness.projection = min(
                    self.awareness.projection + self.detection_accuracy * 20 * (1 + random_noise) * severity * self.experience * org_sa_modifier * sa_reduction, 100)
            else:
                self.awareness.perception = min(
                    self.awareness.perception + self.detection_accuracy * 40 * (1 + random_noise) * severity * (1 + self.experience) * org_sa_modifier * sa_reduction, 100)
                self.awareness.comprehension = min(
                    self.awareness.comprehension + self.detection_accuracy * 20 * (1 + random_noise) * severity * (1 - self.workload / 5) * org_sa_modifier * sa_reduction, 100)
                self.awareness.projection = min(
                    self.awareness.projection + self.detection_accuracy * 15 * (1 + random_noise) * severity * self.experience * org_sa_modifier * sa_reduction, 100)

            logging.debug(
                f"Agent {self.unique_id} ({self.role.value}) observed event {event.get('type', 'None')} "
                f"at step {self.model.schedule.steps}, SA updated: "
                f"P={self.awareness.perception:.2f}, C={self.awareness.comprehension:.2f}, Proj={self.awareness.projection:.2f}, "
                f"Random_Noise={random_noise:.2f}"
            )

            action = self.decide_action(event)
            if action == ActionType.ACT and self.model.budget >= 1000:
                if event["type"] == EventType.HAZARD and self.role == AgentRole.WORKER and not self.model.hazard_acted_this_step:
                    self.execute_action(event, action)
                    self.model.hazard_acted_this_step = True
                    logging.info(
                        f"Agent {self.unique_id} ({self.role.value}) acted on {event.get('type')} event, "
                        f"at step {self.model.schedule.steps}"
                    )
                elif event["type"] != EventType.HAZARD:
                    self.execute_action(event, action)
                    logging.info(
                        f"Agent {self.unique_id} ({self.role.value}) acted on {event.get('type')} event, "
                        f"at step {self.model.schedule.steps}"
                    )
            elif event["type"] != EventType.DELAY and random.random() < self.reporting_probability:
                report = {"event": event, "source": self.role.value, "timestamp": self.model.schedule.steps}
                if self.send_report(report):
                    self.reports_sent += 1
                    logging.debug(
                        f"Agent {self.unique_id} ({self.role.value}) sent report for event {event.get('type')} "
                        f"at step {self.model.schedule.steps}"
                    )

            return True
        return False

    def send_report(self, report: Dict) -> bool:
        return self.model.send_report(self, report)

    def decide_action(self, event: Dict, is_follow_up: bool = False) -> ActionType:
        # Handle specific cases directly
        if event["type"] == EventType.DELAY:
            return ActionType.ACT
        if self.role == AgentRole.WORKER and event["type"] == EventType.HAZARD and not is_follow_up:
            return ActionType.ACT

        # Prepare features for MLP
        resource_availability = self.model.equipment_available / self.model.initial_equipment if self.model.initial_equipment > 0 else 0.5
        features = [
            self.workload / 5,  # Normalize to [0,1]
            self.fatigue,
            event.get("severity", 0.5),
            self.experience,
            self.time_pressure,
            resource_availability,
            self.risk_tolerance,
            self.stress,
            self.recent_hazard
        ]
        features = np.array([features])

        # Get action probabilities from MLP
        try:
            action_probs = self.decision_model.predict_proba(features)[0]
            actions = self.decision_model.model.classes_  # ['act', 'escalate', 'ignore', 'report']
            if is_follow_up:
                # Adjust probabilities for follow-up: exclude 'report', boost 'act'
                action_probs = np.array([p if action in ['act', 'escalate'] else 0 for action, p in zip(actions, action_probs)])
                if action_probs.sum() > 0:
                    action_probs = action_probs / action_probs.sum()  # Renormalize
                else:
                    action_probs = np.array([0.9 if action == 'act' else 0.1 if action == 'escalate' else 0.0 for action in actions])
            selected_action = np.random.choice(actions, p=action_probs)
            action_map = {'act': ActionType.ACT, 'escalate': ActionType.ESCALATE, 'report': ActionType.REPORT, 'ignore': None}
            selected_action_type = action_map.get(selected_action)
            
            # Log decision
            logging.debug(
                f"Agent {self.unique_id} ({self.role.value}) decided action {selected_action_type} for event {event.get('type')} "
                f"at step {self.model.schedule.steps}, probabilities={dict(zip(actions, action_probs.round(3)))}"
            )
            
            # If 'ignore' is selected, return None to skip action
            if selected_action == 'ignore':
                return None
            return selected_action_type
        except Exception as e:
            logging.error(
                f"Agent {self.unique_id} ({self.role.value}) failed to decide action for event {event.get('type')}: {e}"
            )
            # Fallback to original logic
            base_prob = {
                ActionType.REPORT: 0.0 if is_follow_up else 0.1,
                ActionType.ACT: 0.9 if is_follow_up else 0.8,
                ActionType.ESCALATE: 0.1
            }
            total = sum(base_prob.values())
            r = random.uniform(0, total)
            cumulative = 0
            for action, prob in base_prob.items():
                cumulative += prob
                if r <= cumulative:
                    logging.debug(
                        f"Agent {self.unique_id} ({self.role.value}) fell back to action {action} for event {event.get('type')} "
                        f"at step {self.model.schedule.steps}"
                    )
                    return action
            return ActionType.ACT

    def execute_action(self, event: Dict, action: ActionType) -> bool:
        if action is None:  # Handle 'ignore' action
            logging.debug(
                f"Agent {self.unique_id} ({self.role.value}) ignored event {event.get('type')} "
                f"at step {self.model.schedule.steps}"
            )
            return False

        success = False
        severity = event.get("severity", 0.5)
        noise_std = 0.2 + 0.3 * self.fatigue
        random_noise = random.gauss(0, noise_std)
        org_sa_modifier = 1.2 if self.model.org_structure == OrgStructure.FLAT else 1.0 if self.model.org_structure == OrgStructure.FUNCTIONAL else 0.8
        sa_reduction = 0.7 if self.model.outcomes.safety_incidents > 5 else 1.0

        if action == ActionType.REPORT:
            report = {"event": event, "source": self.role.value, "timestamp": self.model.schedule.steps}
            if self.send_report(report):
                self.reports_sent += 1
                self.actions_taken[action] += 1
                self.model.budget -= 1000
                success = True
                logging.debug(
                    f"Agent {self.unique_id} ({self.role.value}) executed REPORT action for event {event.get('type')} "
                    f"at step {self.model.schedule.steps}, budget={self.model.budget:.2f}"
                )
        elif action == ActionType.ACT:
            if self.model.budget < 1000:
                logging.warning(
                    f"Agent {self.unique_id} ({self.role.value}) failed to ACT on {event.get('type')} "
                    f"due to insufficient budget {self.model.budget:.2f} at step {self.model.schedule.steps}"
                )
                return False
            # Update SA for ACT
            self.awareness.perception = min(
                self.awareness.perception + self.detection_accuracy * 20 * (1 + random_noise) * severity * (1 + self.experience) * org_sa_modifier * sa_reduction, 100)
            self.awareness.comprehension = min(
                self.awareness.comprehension + self.detection_accuracy * 15 * (1 + random_noise) * severity * (1 - self.workload / 5) * org_sa_modifier * sa_reduction, 100)
            self.awareness.projection = min(
                self.awareness.projection + self.detection_accuracy * 10 * (1 + random_noise) * severity * self.experience * org_sa_modifier * sa_reduction, 100)
            if event["type"] == EventType.HAZARD:
                if self.role == AgentRole.WORKER:
                    self.model.outcomes.safety_incidents += 1
                    self.model.outcomes.incident_points += event["severity"] * 100
                    self.model.budget -= 10000
                    if self.model.equipment_available > 0:
                        self.model.equipment_available -= 1
                    logging.info(
                        f"Agent {self.unique_id} ({self.role.value}) triggered safety incident for HAZARD, "
                        f"severity={event['severity']:.2f}, incident_points={self.model.outcomes.incident_points:.1f}, "
                        f"budget={self.model.budget:.2f}, equipment={self.model.equipment_available} "
                        f"at step {self.model.schedule.steps}"
                    )
            elif event["type"] == EventType.DELAY:
                self.model.outcomes.tasks_completed_on_time += 1
                self.model.budget -= 1000
                if self.role == AgentRole.MANAGER and self.experience > 0.7 and random.random() < 0.8:
                    if self.model.budget >= 1000:
                        self.model.outcomes.tasks_completed_on_time += 2
                        self.model.budget -= 1000
                        logging.info(
                            f"Agent {self.unique_id} (Manager) completed extra tasks for DELAY, "
                            f"budget={self.model.budget:.2f} at step {self.model.schedule.steps}"
                        )
                logging.info(
                    f"Agent {self.unique_id} ({self.role.value}) completed task for DELAY, "
                    f"budget={self.model.budget:.2f} at step {self.model.schedule.steps}"
                )
            elif event["type"] == EventType.RESOURCE_SHORTAGE:
                if self.model.equipment_available > 0:
                    self.model.budget -= 3000
                    self.model.equipment_available -= 1
                    logging.info(
                        f"Agent {self.unique_id} ({self.role.value}) used equipment for RESOURCE_SHORTAGE, "
                        f"budget={self.model.budget:.2f}, equipment={self.model.equipment_available} "
                        f"at step {self.model.schedule.steps}"
                    )
            self.actions_taken[action] += 1
            success = True
            logging.debug(
                f"Agent {self.unique_id} ({self.role.value}) updated SA for ACT on {event.get('type')}: "
                f"P={self.awareness.perception:.2f}, C={self.awareness.comprehension:.2f}, Proj={self.awareness.projection:.2f}, "
                f"Random_Noise={random_noise:.2f}"
            )
        elif action == ActionType.ESCALATE:
            if self.model.budget < 1500:
                logging.warning(
                    f"Agent {self.unique_id} ({self.role.value}) failed to ESCALATE on {event.get('type')} "
                    f"due to insufficient budget {self.model.budget:.2f} at step {self.model.schedule.steps}"
                )
                return False
            # Update SA for ESCALATE
            self.awareness.perception = min(
                self.awareness.perception + self.detection_accuracy * 15 * (1 + random_noise) * severity * (1 + self.experience) * org_sa_modifier * sa_reduction, 100)
            self.awareness.comprehension = min(
                self.awareness.comprehension + self.detection_accuracy * 10 * (1 + random_noise) * severity * (1 - self.workload / 5) * org_sa_modifier * sa_reduction, 100)
            self.awareness.projection = min(
                self.awareness.projection + self.detection_accuracy * 8 * (1 + random_noise) * severity * self.experience * org_sa_modifier * sa_reduction, 100)
            report = {"event": event, "source": self.role.value, "timestamp": self.model.schedule.steps}
            if self.send_report(report):
                self.reports_sent += 1
                self.actions_taken[action] += 1
                self.model.budget -= 1500
                success = True
                logging.debug(
                    f"Agent {self.unique_id} ({self.role.value}) executed ESCALATE action for event {event.get('type')} "
                    f"at step {self.model.schedule.steps}, budget={self.model.budget:.2f}, "
                    f"SA updated: P={self.awareness.perception:.2f}, C={self.awareness.comprehension:.2f}, Proj={self.awareness.projection:.2f}, "
                    f"Random_Noise={random_noise:.2f}"
                )
        return success
