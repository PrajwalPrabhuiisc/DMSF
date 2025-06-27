import logging
import numpy as np
import mesa
import random
import traceback
from typing import Dict, Optional
from enums import AgentRole, EventType, OrgStructure
from data_classes import SituationalAwareness
from decision_model import DecisionModel

class ConstructionAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', role: AgentRole, pos: tuple):
        super().__init__(unique_id, model)
        self.role = role
        self.pos = pos
        self.awareness = SituationalAwareness()
        self.workload = random.randint(1, 3)
        self.fatigue = random.uniform(0, 0.5)
        self.experience = random.uniform(0, 0.5) if role == AgentRole.WORKER else random.uniform(0.5, 1.0)
        self.risk_tolerance = random.uniform(0.3, 0.7) if role == AgentRole.WORKER else random.uniform(0.1, 0.5)
        self.reports_sent = 0
        self.reports_received = []
        self.decision_model = DecisionModel()
        self.actions_taken = {"ignore": 0, "report": 0, "act": 0, "escalate": 0}
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

    def observe_event(self, event: Optional[Dict]) -> bool:
        if not event:
            return False
        detection_modifier = 1.5 if self.role == AgentRole.REPORTER else 1.0
        if random.random() < 0.02 * (1 - self.experience * detection_modifier):
            print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to observe event {event['type'].value}")
            return False
        
        org_sa_modifier = (
            1.2 if self.model.org_structure == OrgStructure.FLAT else
            0.8 if self.model.org_structure == OrgStructure.HIERARCHICAL else
            1.0
        )
        if self.role == AgentRole.REPORTER:
            self.awareness.perception = min(self.awareness.perception + self.detection_accuracy * 50 * org_sa_modifier, 100)
            self.awareness.comprehension = min(self.awareness.comprehension + self.detection_accuracy * 30 * org_sa_modifier, 100)
            self.awareness.projection = min(self.awareness.projection + self.detection_accuracy * 15 * org_sa_modifier, 100)
        else:
            self.awareness.perception = min(self.awareness.perception + self.detection_accuracy * 40 * (1 + self.experience) * org_sa_modifier, 100)
            self.awareness.comprehension = min(self.awareness.comprehension + self.detection_accuracy * 20 * (1 - self.workload / 5) * org_sa_modifier, 100)
            self.awareness.projection = min(self.awareness.projection + self.detection_accuracy * 10 * (1 - self.fatigue) * org_sa_modifier, 100)
        
        if self.reports_received and self.role != AgentRole.REPORTER:
            additional_aspects = sum(1 for r in self.reports_received if r.get("event", {}).get("type") == event["type"])
            self.awareness.comprehension = min(self.awareness.comprehension + additional_aspects * 10, 100)
        elif self.role == AgentRole.REPORTER:
            self.awareness.comprehension = min(self.awareness.comprehension + len(self.reports_received) * 15, 100)
        
        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) observed event {event['type'].value}, SA updated: "
              f"Perception={self.awareness.perception:.2f}, Comprehension={self.awareness.comprehension:.2f}, Projection={self.awareness.projection:.2f}")
        return True

    def decide_action(self, event: Optional[Dict], is_follow_up: bool = False) -> str:
        if not event:
            return "ignore"
        event_severity = 1.0 if event["type"] == EventType.HAZARD else 0.7 if event["type"] == EventType.DELAY else 0.5
        org_modifier = (
            0.8 if self.model.org_structure == OrgStructure.FLAT else
            1.2 if self.model.org_structure == OrgStructure.HIERARCHICAL else
            1.0
        )
        time_pressure = min(self.workload / 5, 1.0)
        resource_availability = min(self.model.budget / 1000000, 1.0) * min(self.model.equipment_available / 50, 1.0)
        stress = 1.0 if len([e for e in self.model.current_events if e["type"] == EventType.HAZARD]) > 1 else 0.5
        recent_hazard = 1.0 if any(r.get("event", {}).get("type") == EventType.HAZARD for r in self.reports_received[-3:]) else 0.0

        inputs = [self.workload * org_modifier, self.fatigue, event_severity, self.experience, time_pressure, resource_availability, self.risk_tolerance, stress, recent_hazard]
        try:
            action_probs = self.decision_model.predict_proba([inputs])[0]
        except Exception as e:
            print(f"Error in predict_proba for Agent {self.unique_id}: {e}")
            action_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        if len(action_probs) != 4:
            print(f"Warning: action_probs has length {len(action_probs)} instead of 4 for Agent {self.unique_id}")
            action_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        q_values = self.q_table[event["type"]]
        org_memory = self.model.organizational_memory[event["type"]]
        if org_memory:
            best_action = max(set(org_memory), key=org_memory.count) if random.random() < 0.7 else None
            if best_action:
                action_probs[["ignore", "report", "act", "escalate"].index(best_action)] *= 2.0
        for i, action in enumerate(["ignore", "report", "act", "escalate"]):
            action_probs[i] += q_values[action] * 0.2
        if event["type"] == EventType.DELAY and is_follow_up:
            action_probs[2] *= 10.0
        elif event["type"] == EventType.DELAY:
            action_probs[2] *= 8.0
        elif event["type"] == EventType.HAZARD:
            action_probs[1] *= 5.0  # Increased to 5.0
            action_probs[3] *= 5.0  # Increased to 5.0
            action_probs[2] *= 1.2
        action_probs /= action_probs.sum()

        if self.risk_tolerance > 0.6 and random.random() < 0.3:
            action_probs[2] *= 2.0
            action_probs /= action_probs.sum()

        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) action probs for {event['type'].value} {'(follow-up)' if is_follow_up else ''}: {action_probs}")
        actions = ["ignore", "report", "act", "escalate"]
        action = actions[np.argmax(action_probs)]

        if self.model.reporting_structure == self.model.ReportingStructure.SELF:
            if action == "report" and random.random() > self.reporting_probability * 0.6:
                self.actions_taken["report"] -= 1
                self.actions_taken["act"] += 1
                action = "act"
            elif self.model.org_structure == OrgStructure.HIERARCHICAL and action == "escalate" and self.role == AgentRole.WORKER:
                action = "act"
                self.actions_taken["escalate"] -= 1
                self.actions_taken["act"] += 1
        elif self.model.reporting_structure == self.model.ReportingStructure.NONE:
            if action in ["report", "escalate"] and random.random() > 0.2:
                self.actions_taken["report"] -= 1 if action == "report" else 0
                self.actions_taken["escalate"] -= 1 if action == "escalate" else 0
                self.actions_taken["act"] += 1
                action = "act"
        
        self.last_event_action = (event["type"], action)
        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) decided action: {action} for event {event['type'].value} {'(follow-up)' if is_follow_up else ''}")
        return action

    def execute_action(self, event: Dict, action: str):
        self.actions_taken[action] += 1
        success = False
        resource_cost = 1000 if event["type"] in [EventType.HAZARD, EventType.DELAY] else 1000  # Unified to 1000
        equipment_needed = 1 if event["type"] == EventType.HAZARD else 0.5 if event["type"] == EventType.DELAY else 1

        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executing action {action} for {event['type'].value}, "
              f"Budget={self.model.budget:.2f}, Equipment={self.model.equipment_available}")

        if action == "act":
            budget_sufficient = self.model.budget >= resource_cost
            equipment_sufficient = self.model.equipment_available >= equipment_needed
            if budget_sufficient and equipment_sufficient:
                self.model.budget -= resource_cost
                self.model.equipment_available -= equipment_needed
                success = random.random() < 0.9 + self.experience * 0.2
                if event["type"] == EventType.HAZARD and success:
                    print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) successfully mitigated HAZARD")
                elif event["type"] == EventType.DELAY and success:
                    completion_prob = max(0.90, 0.98 - (self.fatigue * 0.15 + self.workload * 0.05 + (1 - self.experience) * 0.1))  # Min 0.90
                    if random.random() < completion_prob:
                        self.model.outcomes.tasks_completed_on_time += 1
                        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) completed DELAY task on time (prob={completion_prob:.2f})")
                    else:
                        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to complete DELAY task on time (prob={completion_prob:.2f})")
                elif not success:
                    if event["type"] == EventType.HAZARD:
                        self.model.outcomes.safety_incidents += 1
                        self.model.outcomes.incident_points += 5
                        self.model.outcomes.cost_overruns += 25000
                        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) failed to mitigate HAZARD, caused incident")
            else:
                failure_reason = []
                if not budget_sufficient:
                    failure_reason.append(f"Budget insufficient: {self.model.budget:.2f} < {resource_cost}")
                if not equipment_sufficient:
                    failure_reason.append(f"Equipment insufficient: {self.model.equipment_available} < {equipment_needed}")
                print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) lacked resources to act on {event['type'].value} "
                      f"({', '.join(failure_reason)})")
                if event["type"] == EventType.HAZARD:
                    self.model.outcomes.safety_incidents += 1
                    self.model.outcomes.incident_points += 5
                    self.model.outcomes.cost_overruns += 25000
                    print(f"Step {self.model.schedule.steps}: HAZARD incident due to insufficient resources")
                elif event["type"] == EventType.DELAY:
                    print(f"Step {self.model.schedule.steps}: DELAY task failed due to insufficient resources")
        elif action == "report" and random.random() > 0.1:
            self.reports_sent += 1
            success = self.model.send_report(self, {"agent_id": self.unique_id, "event": event, "action": action})
        elif action == "escalate" and random.random() > 0.1:
            self.reports_sent += 1
            success = self.model.send_report(self, {"agent_id": self.unique_id, "event": event, "action": action})
        elif action == "ignore" and event["type"] == EventType.HAZARD:
            if random.random() < 0.05 * (1 - self.experience):  # Reduced to 0.05
                self.model.outcomes.safety_incidents += 1
                self.model.outcomes.incident_points += 5
                self.model.outcomes.cost_overruns += 25000
                print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) ignored HAZARD, caused incident")

        if action == "report" or action == "escalate":
            self.workload = max(1, self.workload - 1 if success else self.workload + 1)
            self.fatigue = min(self.fatigue + (0.1 if not success else -0.05), 1.0)
        elif action == "act":
            self.workload = min(self.workload + 1, 5)
            self.fatigue = min(self.fatigue + 0.1, 1.0)
            if success:
                self.experience = min(self.experience + 0.05, 1.0)
                self.model.organizational_memory[event["type"]].append(action)
        else:
            self.fatigue = max(0, self.fatigue - 0.05)

        if self.last_event_action:
            event_type, chosen_action = self.last_event_action
            reward = 1.0 if success else -1.0 if action == "act" else 0.0
            if event_type == EventType.HAZARD and action == "act" and success:
                reward = 3.0
            elif event_type == EventType.HAZARD and action == "ignore":
                reward = -3.0
            self.q_table[event_type][chosen_action] += 0.2 * (reward - self.q_table[event_type][chosen_action])

    def step(self):
        try:
            events = self.model.get_events()
            observed_events = [event for event in events if self.observe_event(event)]

            for event in observed_events:
                action = self.decide_action(event)
                self.execute_action(event, action)

                if action in ["report", "escalate"] and self.model.reporting_structure in [self.model.ReportingStructure.SELF, self.model.ReportingStructure.NONE]:
                    if not any(r.get("acted_on", False) for r in self.model.reports if r.get("event") == event):
                        follow_up_action = self.decide_action(event, is_follow_up=True)
                        self.execute_action(event, follow_up_action)
                        print(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executed fallback follow-up action {follow_up_action} for event {event['type'].value}")
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            self.model.grid.move_agent(self, random.choice(possible_moves))
        except Exception as e:
            print(f"Error in agent {self.unique_id} step: {e}")
            logging.error(f"Agent step error: {traceback.format_exc()}")
