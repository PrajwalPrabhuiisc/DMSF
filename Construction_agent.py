import logging
import numpy as np
import mesa
import random
import traceback
from typing import Dict, Optional, List
from enums import AgentRole, EventType, OrgStructure
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
        
        q_values = self.q_table[event["type"]]
        org_memory = self.model.organizational_memory[event["type"]]
        if org_memory['success_count'] > 0:
            best_action = org_memory['best_action']
            if best_action and random.random() < 0.7:
                action_probs[["ignore", "report", "act", "escalate"].index(best_action)] *= 2.0
        for i, action in enumerate(["ignore", "report", "act", "escalate"]):
            action_probs[i] += q_values[action] * 0.2
        if event["type"] == EventType.DELAY:
            action_probs[2] *= 20.0  # Further increased to favor 'act'
        elif event["type"] == EventType.HAZARD:
            action_probs[1] *= 5.0
            action_probs[3] *= 5.0
            action_probs[2] *= 2.0
        action_probs /= action_probs.sum()

        if self.risk_tolerance > 0.6 and random.random() < 0.3:
            action_probs[2] *= 2.0
            action_probs /= action_probs.sum()

        logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) action probs for {event['type'].value} {'(follow-up)' if is_follow_up else ''}: {action_probs}")
        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) action probs for {event['type'].value} {'(follow-up)' if is_follow_up else ''}: {action_probs}")
        actions = ["ignore", "report", "act", "escalate"]
        action = actions[np.argmax(action_probs)]

        if self.model.reporting_structure == self.model.ReportingStructure.SELF:
            if action == "report" and random.random() > self.reporting_probability * 0.9:  # Increased threshold
                self.action_counts["report"] -= 1
                self.action_counts["act"] += 1
                action = "act"
            elif self.model.org_structure == OrgStructure.HIERARCHICAL and action == "escalate" and self.role == AgentRole.WORKER:
                action = "act"
                self.action_counts["escalate"] -= 1
                self.action_counts["act"] += 1

        self.last_event_action = (event["type"], action)
        logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) decided action: {action} for event {event['type'].value} {'(follow-up)' if is_follow_up else ''}")
        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) decided action: {action} for event {event['type'].value} {'(follow-up)' if is_follow_up else ''}")
        return action

    def execute_action(self, event: Dict, action: str):
        self.action_counts[action] += 1
        success = False
        event_severity = event.get("severity", 1.0)
        resource_cost = 200 * event_severity if event["type"] == EventType.DELAY else 500 * event_severity  # Further lowered for DELAY
        equipment_needed = 0.1 * event_severity if event["type"] == EventType.DELAY else 0.5 * event_severity if event["type"] == EventType.HAZARD else 0.5 * event_severity

        logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executing action {action} for {event['type'].value} (severity={event_severity:.2f}), "
                     f"Budget={self.model.budget:.2f}, Equipment={self.model.equipment}")
        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executing action {action} for {event['type'].value} (severity={event_severity:.2f}), "
              f"Budget={self.model.budget:.2f}, Equipment={self.model.equipment}")

        if action == "act":
            budget_sufficient = self.model.budget >= resource_cost
            equipment_sufficient = self.model.equipment >= equipment_needed
            success_prob = min(0.9 + self.experience * 0.05 - self.fatigue * 0.1 - self.workload * 0.05, 0.95)  # Increased for DELAY
            if budget_sufficient and equipment_sufficient:
                self.model.budget -= resource_cost
                self.model.equipment -= equipment_needed
                success = random.random() < success_prob
                if event["type"] == EventType.HAZARD and success:
                    logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) successfully mitigated HAZARD (prob={success_prob:.2f})")
                    print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) successfully mitigated HAZARD (prob={success_prob:.2f})")
                elif event["type"] == EventType.DELAY and success:
                    completion_prob = max(0.95, 0.98 - (self.fatigue * 0.03 + self.workload * 0.03 + (1 - self.experience) * 0.03))  # Increased
                    if random.random() < completion_prob:
                        self.model.outcomes.tasks_completed_on_time += 1
                        logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) completed DELAY task on time (prob={completion_prob:.2f})")
                        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) completed DELAY task on time (prob={completion_prob:.2f})")
                    else:
                        logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to complete DELAY task on time (prob={completion_prob:.2f})")
                        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to complete DELAY task on time (prob={completion_prob:.2f})")
                elif not success and event["type"] == EventType.HAZARD:
                    self.model.outcomes.safety_incidents += 1
                    self.model.outcomes.incident_points += 5 * event_severity
                    self.model.outcomes.cost_overruns += 25000 * event_severity
                    logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to mitigate HAZARD (prob={success_prob:.2f}), caused incident (points={5 * event_severity:.1f})")
                    print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to mitigate HAZARD (prob={success_prob:.2f}), caused incident (points={5 * event_severity:.1f})")
            else:
                failure_reason = []
                if not budget_sufficient:
                    failure_reason.append(f"Budget insufficient: {self.model.budget:.2f} < {resource_cost}")
                if not equipment_sufficient:
                    failure_reason.append(f"Equipment insufficient: {self.model.equipment} < {equipment_needed}")
                logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) lacked resources to act on {event['type'].value} "
                             f"({', '.join(failure_reason)})")
                print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) lacked resources to act on {event['type'].value} "
                      f"({', '.join(failure_reason)})")
                if event["type"] == EventType.HAZARD:
                    self.model.outcomes.safety_incidents += 1
                    self.model.outcomes.incident_points += 5 * event_severity
                    self.model.outcomes.cost_overruns += 25000 * event_severity
                    logging.debug(f"Step {self.model.schedule.steps}: HAZARD incident due to insufficient resources (points={5 * event_severity:.1f})")
                    print(f"Step {self.model.schedule.steps}: HAZARD incident due to insufficient resources (points={5 * event_severity:.1f})")
        elif action == "report" and random.random() > 0.03:  # Further reduced failure chance
            self.reports_sent += 1
            success = self.model.send_report(self, event)
            logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) report sent: {success}")
            print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) report sent: {success}")
        elif action == "escalate" and random.random() > 0.03:
            self.reports_sent += 1
            success = self.model.send_report(self, event)
            logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) escalate sent: {success}")
            print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) escalate sent: {success}")
        elif action == "ignore" and event["type"] == EventType.HAZARD:
            ignore_risk = 0.15 * (1 - self.experience) * event_severity
            if random.random() < ignore_risk:
                self.model.outcomes.safety_incidents += 1
                self.model.outcomes.incident_points += 5 * event_severity
                self.model.outcomes.cost_overruns += 25000 * event_severity
                logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) ignored HAZARD, caused incident (prob={ignore_risk:.3f}, points={5 * event_severity:.1f})")
                print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) ignored HAZARD, caused incident (prob={ignore_risk:.3f}, points={5 * event_severity:.1f})")

        if action == "report" or action == "escalate":
            self.workload = max(1, self.workload - 1 if success else self.workload + 1)
            self.fatigue = min(self.fatigue + (0.1 if not success else -0.05), 1.0)
        elif action == "act":
            self.workload = min(self.workload + 1, 5)
            self.fatigue = min(self.fatigue + 0.1, 1.0)
            if success:
                self.experience = min(self.experience + 0.05, 1.0)
                self.model.organizational_memory[event["type"]]['success_count'] += 1
                self.model.organizational_memory[event["type"]]['best_action'] = action
        else:
            self.fatigue = max(0, self.fatigue - 0.05)

        if self.last_event_action:
            event_type, chosen_action = self.last_event_action
            reward = 1.0 if success else -1.0 if action == "act" else 0.0
            if event_type == EventType.HAZARD and action == "act" and success:
                reward = 3.0 * event_severity
            elif event_type == EventType.HAZARD and action == "ignore":
                reward = -3.0 * event_severity
            self.q_table[event_type][chosen_action] += 0.2 * (reward - self.q_table[event_type][chosen_action])

    def step(self):
        try:
            events = self.model.get_events()
            observed_events = self.observe_events(events)
            for event in observed_events:
                action = self.decide_action(event)
                self.execute_action(event, action)
                if action in ["report", "escalate"] and self.model.reporting_structure in [self.model.ReportingStructure.SELF, self.model.ReportingStructure.NONE]:
                    for report in self.received_reports:
                        if report["type"] == event["type"] and not report.get("acted_on", False):
                            follow_up_action = self.decide_action(event, is_follow_up=True)
                            self.execute_action(event, follow_up_action)
                            report["acted_on"] = True
                            logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executed follow-up action {follow_up_action} for event {event['type'].value}")
                            print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executed follow-up action {follow_up_action} for event {event['type'].value}")
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            self.model.grid.move_agent(self, random.choice(possible_moves))
        except Exception as e:
            logging.error(f"Error in agent {self.unique_id} step: {traceback.format_exc()}")
            print(f"Error in agent {self.unique_id} step: {e}")
