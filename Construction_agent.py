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

    def step(self):
        """Main step function called by the scheduler"""
        try:
            # Get events from the model
            events = self.model.get_events()
            
            # Observe events
            observed_events = self.observe_events(events)
            
            # Process each observed event
            for event in observed_events:
                action = self.decide_action(event)
                self.execute_action(event, action)
            
            # Process received reports
            self.process_received_reports()
            
            # Update agent state
            self.update_agent_state()
            
        except Exception as e:
            logging.error(f"Error in agent step for Agent {self.unique_id}: {traceback.format_exc()}")
            print(f"Error in agent step for Agent {self.unique_id}: {e}")

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

    def execute_action(self, event: Dict, action: str):
        """Execute the decided action for an event"""
        try:
            event_type = event["type"]
            event_severity = event.get("severity", 1.0)
            
            if action == "ignore":
                self._handle_ignore_action(event)
            elif action == "report":
                self._handle_report_action(event)
            elif action == "act":
                self._handle_act_action(event)
            elif action == "escalate":
                self._handle_escalate_action(event)
            
            # Update organizational memory based on action effectiveness
            self._update_organizational_memory(event_type, action)
            
            logging.debug(f"Step {self.model.schedule.steps}: Agent {self.unique_id} ({self.role.value}) executed {action} for {event_type.value}")
            
        except Exception as e:
            logging.error(f"Error executing action {action} for Agent {self.unique_id}: {traceback.format_exc()}")
            print(f"Error executing action {action} for Agent {self.unique_id}: {e}")

    def _handle_ignore_action(self, event: Dict):
        """Handle ignore action - minimal impact but potential consequences"""
        event_type = event["type"]
        event_severity = event.get("severity", 1.0)
        
        # Ignoring events can lead to escalation of problems
        if event_type == EventType.HAZARD and random.random() < 0.3 * event_severity:
            self.model.outcomes.safety_incidents += 1
            self.model.outcomes.incident_points += 3 * event_severity
            logging.debug(f"Ignored HAZARD led to safety incident")
        elif event_type == EventType.DELAY and random.random() < 0.4 * event_severity:
            self.model.outcomes.cost_overruns += 15000 * event_severity
            logging.debug(f"Ignored DELAY led to cost overrun")

    def _handle_report_action(self, event: Dict):
        """Handle report action - communicate the event to others"""
        if random.random() < self.reporting_probability:
            success = self.model.send_report(self, event)
            if success:
                self.reports_sent += 1
                # Slight positive impact on situational awareness
                self.sa.comprehension = min(self.sa.comprehension + 5, 100)
                logging.debug(f"Agent {self.unique_id} successfully reported {event['type'].value}")
            else:
                logging.debug(f"Agent {self.unique_id} failed to report {event['type'].value}")

    def _handle_act_action(self, event: Dict):
        """Handle act action - directly address the event"""
        event_type = event["type"]
        event_severity = event.get("severity", 1.0)
        
        # Calculate action effectiveness based on agent capabilities
        effectiveness = self._calculate_action_effectiveness(event)
        
        if event_type == EventType.HAZARD:
            if effectiveness > 0.6:
                # Successful mitigation
                self.model.outcomes.incident_points = max(0, self.model.outcomes.incident_points - 2)
                self.sa.projection = min(self.sa.projection + 10, 100)
            else:
                # Partial mitigation
                if random.random() < 0.2:
                    self.model.outcomes.safety_incidents += 0.5
                    self.model.outcomes.incident_points += 1 * event_severity
                    
        elif event_type == EventType.DELAY:
            if effectiveness > 0.5:
                # Successfully handle delay
                self.model.outcomes.tasks_completed_on_time += 1
                self.model.outcomes.total_tasks += 1
            else:
                # Partial success
                self.model.outcomes.total_tasks += 1
                if random.random() < 0.3:
                    self.model.outcomes.cost_overruns += 8000 * event_severity
                    
        elif event_type == EventType.RESOURCE_SHORTAGE:
            if effectiveness > 0.4:
                # Resource issue resolved
                self.model.budget = max(self.model.budget - 5000 * event_severity, 0)
                self.model.equipment = max(self.model.equipment - int(10 * event_severity), 0)
            else:
                # Resource issue persists
                self.model.budget = max(self.model.budget - 12000 * event_severity, 0)
                self.model.equipment = max(self.model.equipment - int(20 * event_severity), 0)

    def _handle_escalate_action(self, event: Dict):
        """Handle escalate action - escalate to higher authority"""
        event_type = event["type"]
        event_severity = event.get("severity", 1.0)
        
        # Find higher-level agents to escalate to
        target_roles = []
        if self.role == AgentRole.WORKER:
            target_roles = [AgentRole.MANAGER, AgentRole.DIRECTOR]
        elif self.role == AgentRole.MANAGER:
            target_roles = [AgentRole.DIRECTOR]
        elif self.role == AgentRole.REPORTER:
            target_roles = [AgentRole.MANAGER, AgentRole.DIRECTOR]
        
        if target_roles:
            # Find nearby agents with target roles
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=5)
            targets = [agent for agent in neighbors if agent.role in target_roles]
            
            if targets:
                target = random.choice(targets)
                escalation_report = {
                    "type": event_type,
                    "severity": event_severity * 1.2,  # Escalated severity
                    "escalated_from": self.role,
                    "acted_on": False
                }
                target.received_reports.append(escalation_report)
                
                # Target agent immediately considers action
                action = target.decide_action(escalation_report, is_follow_up=True)
                target.execute_action(escalation_report, action)
                escalation_report["acted_on"] = True
                
                logging.debug(f"Agent {self.unique_id} escalated {event_type.value} to Agent {target.unique_id}")

    def _calculate_action_effectiveness(self, event: Dict) -> float:
        """Calculate how effective an agent's action will be"""
        base_effectiveness = 0.5
        
        # Role-based effectiveness
        role_multipliers = {
            AgentRole.WORKER: 0.6,
            AgentRole.MANAGER: 0.8,
            AgentRole.DIRECTOR: 0.9,
            AgentRole.REPORTER: 0.4  # Reporters are better at reporting than acting
        }
        
        effectiveness = base_effectiveness * role_multipliers[self.role]
        
        # Experience factor
        effectiveness += self.experience * 0.3
        
        # Fatigue and stress penalties
        effectiveness -= self.fatigue * 0.2
        effectiveness -= self.stress * 0.15
        
        # Workload penalty
        effectiveness -= (self.workload / 5) * 0.1
        
        # Situational awareness bonus
        sa_bonus = (self.sa.total_score() / 300) * 0.2
        effectiveness += sa_bonus
        
        return max(0.1, min(1.0, effectiveness))

    def _update_organizational_memory(self, event_type: EventType, action: str):
        """Update organizational memory based on action outcomes"""
        if event_type in self.model.organizational_memory:
            memory = self.model.organizational_memory[event_type]
            
            # Simple success tracking (could be enhanced with actual outcome measurement)
            if action in ["act", "escalate", "report"]:
                memory['success_count'] += 1
                if memory['best_action'] is None or action == memory['best_action']:
                    memory['best_action'] = action

    def process_received_reports(self):
        """Process reports received from other agents"""
        try:
            for report in self.received_reports:
                if not report.get("acted_on", False):
                    # Decide whether to act on the received report
                    if random.random() < 0.7:  # 70% chance to act on received reports
                        action = self.decide_action(report, is_follow_up=True)
                        if action != "ignore":
                            self.execute_action(report, action)
                            report["acted_on"] = True
                            logging.debug(f"Agent {self.unique_id} acted on received report: {report['type'].value}")
                    
                    # Update situational awareness based on report
                    self.sa.comprehension = min(self.sa.comprehension + 3, 100)
                    
        except Exception as e:
            logging.error(f"Error processing received reports for Agent {self.unique_id}: {traceback.format_exc()}")
            print(f"Error processing received reports for Agent {self.unique_id}: {e}")

    def update_agent_state(self):
        """Update agent's internal state each step"""
        try:
            # Natural recovery from fatigue and stress
            self.fatigue = max(0, self.fatigue - 0.02)
            self.stress = max(0, self.stress - 0.03)
            
            # Experience growth
            self.experience = min(1.0, self.experience + 0.001)
            
            # Workload variation
            if random.random() < 0.1:  # 10% chance of workload change
                self.workload = max(1, min(5, self.workload + random.choice([-1, 1])))
            
            # Situational awareness decay
            self.sa.perception = max(0, self.sa.perception - 1)
            self.sa.comprehension = max(0, self.sa.comprehension - 0.5)
            self.sa.projection = max(0, self.sa.projection - 0.8)
            
            # Clear old received reports (keep only last 5)
            if len(self.received_reports) > 5:
                self.received_reports = self.received_reports[-5:]
                
        except Exception as e:
            logging.error(f"Error updating agent state for Agent {self.unique_id}: {traceback.format_exc()}")
            print(f"Error updating agent state for Agent {self.unique_id}: {e}")

    def move(self):
        """Move agent to a random neighboring cell"""
        try:
            possible_steps = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            self.pos = new_position
        except Exception as e:
            logging.error(f"Error moving Agent {self.unique_id}: {e}")

    def get_neighbors_by_role(self, role: AgentRole, radius: int = 3) -> List['ConstructionAgent']:
        """Get neighboring agents with a specific role"""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=radius)
        return [agent for agent in neighbors if agent.role == role]

    def __str__(self):
        return f"Agent {self.unique_id} ({self.role.value}) - SA: {self.sa.total_score():.1f}, Experience: {self.experience:.2f}, Fatigue: {self.fatigue:.2f}"
