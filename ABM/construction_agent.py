import mesa
import random
from typing import Dict, Optional
from enums import AgentRole, EventType
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
        self.reports_sent = 0
        self.reports_received = []
        self.decision_model = DecisionModel()
        self.actions_taken = {"ignore": 0, "report": 0, "act": 0, "escalate": 0}

        self.detection_accuracy = (
            self.model.reporter_detection if role == AgentRole.REPORTER else
            self.model.worker_detection if role == AgentRole.WORKER else
            self.model.manager_detection
        )
        self.reporting_chance = (
            self.model.reporter_reporting if role == AgentRole.REPORTER else
            self.model.worker_reporting if role == AgentRole.WORKER else
            self.model.manager_reporting
        )

    def observe_event(self, event: Optional[Dict]) -> bool:
        if not event:
            return False
        if random.random() < 0.1 + self.fatigue * 0.2:
            return False
        self.awareness.perception = min(self.awareness.perception + self.detection_accuracy * 40, 100)
        self.awareness.comprehension = min(self.awareness.comprehension + self.detection_accuracy * 20 * (1 - self.workload / 5), 100)
        self.awareness.projection = min(self.awareness.projection + self.detection_accuracy * 10 * (1 - self.fatigue), 100)
        if self.reports_received:
            self.awareness.comprehension = min(self.awareness.comprehension + len(self.reports_received) * 5, 100)
        return True

    def decide_action(self, event: Optional[Dict]) -> str:
        if not event:
            return "ignore"
        event_severity = 1.0 if event["type"] == EventType.HAZARD else 0.7 if event["type"] == EventType.DELAY else 0.5
        return self.decision_model.predict(self.workload, self.fatigue, event_severity)

    def step(self):
        try:
            events = self.model.get_events()
            observed_events = [event for event in events if self.observe_event(event)]

            for event in observed_events:
                action = self.decide_action(event)
                self.actions_taken[action] += 1

                if self.model.reporting_structure == self.model.ReportingStructure.DEDICATED and self.role != AgentRole.REPORTER:
                    if action in ["report", "escalate"]:
                        action = "act"
                        self.actions_taken["report"] -= 1
                        self.actions_taken["escalate"] -= 1
                        self.actions_taken["act"] += 1
                elif self.model.reporting_structure == self.model.ReportingStructure.SELF:
                    if action == "report" and random.random() > self.reporting_chance:
                        self.actions_taken["report"] -= 1
                        self.actions_taken["ignore"] += 1
                        action = "ignore"
                elif self.model.reporting_structure == self.model.ReportingStructure.NONE:
                    if action in ["report", "escalate"] and random.random() > 0.5:
                        self.actions_taken["report"] -= 1
                        self.actions_taken["escalate"] -= 1
                        self.actions_taken["ignore"] += 1
                        action = "ignore"

                report_success = False
                if action == "report" and random.random() > 0.1:
                    self.reports_sent += 1
                    report_success = self.model.send_report(self, {"agent_id": self.unique_id, "event": event, "action": action})
                elif action == "escalate" and random.random() > 0.1:
                    self.reports_sent += 1
                    report_success = self.model.send_report(self, {"agent_id": self.unique_id, "event": event, "action": action})
                elif action == "act":
                    if event["type"] == EventType.HAZARD:
                        if random.random() < 0.2:
                            self.model.outcomes.incident_points += 5
                            self.model.outcomes.safety_incidents += 1
                    elif event["type"] == EventType.DELAY:
                        self.model.outcomes.total_tasks += 1
                        if random.random() > 0.1:
                            self.model.outcomes.tasks_completed_on_time += 1

                if action == "report" or action == "escalate":
                    self.workload = max(1, self.workload - 1 if report_success else self.workload + 1)
                    self.fatigue = min(self.fatigue + (0.1 if not report_success else -0.05), 1.0)
                elif action == "act":
                    self.workload = min(self.workload + 1, 5)
                    self.fatigue = min(self.fatigue + 0.1, 1.0)
                else:
                    self.fatigue = max(0, self.fatigue - 0.05)

            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            self.model.grid.move_agent(self, random.choice(possible_moves))
        except Exception as e:
            print(f"Error in agent {self.unique_id} step: {e}")
            traceback.print_exc()