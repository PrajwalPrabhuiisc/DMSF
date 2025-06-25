import mesa
import random
import numpy as np
from typing import Dict, Optional, List, Any
from enums import AgentRole, EventType, EventSeverity, ReportingStructure
from data_classes import SituationalAwareness
from decision_model import MLDecisionSupport  # Changed from DecisionModel to MLDecisionSupport

class ConstructionAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', role: AgentRole, pos: tuple):
        super().__init__(unique_id, model)
        self.role = role
        self.pos = pos
        self.awareness = SituationalAwareness()
        self.workload = random.randint(1, 3)
        self.fatigue = random.uniform(0, 0.5)
        self.experience_level = random.uniform(0.1, 1.0)
        self.stress_level = random.uniform(0, 0.3)
        self.reports_sent = 0
        self.reports_received = []
        self.decision_model = MLDecisionSupport()  # Updated to MLDecisionSupport
        self.actions_taken = {"ignore": 0, "report": 0, "act": 0, "escalate": 0}
        self.trust_in_management = random.uniform(0.3, 0.9)
        self.recent_actions = {
            'successful_reports': 0,
            'correct_detections': 0,
            'witnessed_incidents': 0,
            'all_actions': []
        }

        self.detection_accuracy = (
            0.95 if role == AgentRole.REPORTER else
            0.95 if role == AgentRole.DIRECTOR else
            0.90 if role == AgentRole.MANAGER else
            0.80
        )
        self.reporting_chance = (
            0.98 if role == AgentRole.REPORTER else
            0.95 if role == AgentRole.DIRECTOR else
            0.90 if role == AgentRole.MANAGER else
            0.80
        )
        self.false_positive_rate = (
            0.01 if role == AgentRole.REPORTER else
            0.01 if role == AgentRole.DIRECTOR else
            0.02 if role == AgentRole.MANAGER else
            0.05
        )

    def update_cognitive_state(self):
        if self.recent_actions['successful_reports'] > 0:
            self.workload = max(1, self.workload - 0.2)
        activity_level = len(self.recent_actions['all_actions'])
        self.fatigue = min(1.0, self.fatigue + 0.02 * activity_level)
        if self.recent_actions['correct_detections'] > 0:
            self.experience_level = min(1.0, self.experience_level + 0.001)
        if self.recent_actions['witnessed_incidents'] > 0:
            self.stress_level = min(1.0, self.stress_level + 0.1)
        else:
            self.stress_level = max(0, self.stress_level - 0.05)

    def modify_decision_probability(self, base_action: float, context: Dict) -> float:
        modifiers = {
            'report': self.trust_in_management * self.reporting_chance,
            'act': self.experience_level * (1 - self.stress_level),
            'escalate': (1 - self.experience_level) * self.trust_in_management,
            'ignore': self.fatigue * (1 - self.detection_accuracy)
        }
        return base_action * modifiers.get(context.get('action_type', 'ignore'), 1.0)

    def update_perception(self, event: Dict, context: Dict):
        base_gain = 40 * (event['severity'].value / 4.0)
        attention_factor = 1.0 - (self.workload - 1) * 0.15
        fatigue_factor = 1.0 - self.fatigue * 0.3
        experience_modifier = 0.8 + (self.experience_level * 0.4)
        visibility_factor = context.get('visibility', 1.0)
        noise_factor = context.get('noise_level', 1.0)
        total_gain = base_gain * attention_factor * fatigue_factor * experience_modifier * visibility_factor * noise_factor
        decay_factor = 0.95
        self.awareness.perception = min(100, (self.awareness.perception * decay_factor) + total_gain)
        self.awareness.perception_confidence = min(1.0, self.awareness.perception_confidence + 0.1)
        self.awareness.information_sources.append({
            'type': 'direct_observation',
            'event': event,
            'step': context.get('current_step', 0),
            'confidence': self.awareness.perception_confidence
        })

    def update_comprehension(self, event: Dict, context: Dict, shared_information: Optional[List[Dict]] = None):
        base_gain = 20 * (event['severity'].value / 4.0)
        workload_penalty = (self.workload - 1) * 0.2
        cognitive_capacity = max(0.1, 1.0 - workload_penalty)
        integration_bonus = 1.0
        if shared_information:
            unique_sources = len(set([info['source'] for info in shared_information]))
            integration_bonus = 1.0 + (unique_sources - 1) * 0.15
        domain_knowledge = context.get('domain_relevance', 0.5)
        knowledge_modifier = 0.5 + (domain_knowledge * 0.5)
        time_pressure = context.get('time_pressure', 0.0)
        time_factor = max(0.3, 1.0 - time_pressure * 0.4)
        comprehension_gain = base_gain * cognitive_capacity * integration_bonus * knowledge_modifier * time_factor
        current_normalized = self.awareness.comprehension / 100.0
        learning_rate = 1.0 - (current_normalized ** 2)
        self.awareness.comprehension = min(100, self.awareness.comprehension + (comprehension_gain * learning_rate))
        information_quality = context.get('information_quality', 0.5)
        confidence_gain = 0.05 * information_quality * integration_bonus
        self.awareness.comprehension_confidence = min(1.0, self.awareness.comprehension_confidence + confidence_gain)

    def update_projection(self, event: Dict, context: Dict, trend_data: Optional[List[Dict]] = None):
        if self.awareness.comprehension < 30:
            return
        base_gain = 10 * (event['severity'].value / 4.0)
        mental_model_strength = self.awareness.comprehension / 100.0
        fatigue_impact = 1.0 - (self.fatigue * 0.4)
        experience_bonus = self.experience_level * 0.3
        pattern_recognition = context.get('pattern_familiarity', 0.5)
        trend_bonus = 1.0
        if trend_data and len(trend_data) > 3:
            trend_bonus = 1.0 + (0.25 * (len(trend_data) / 10))
        situational_complexity = context.get('complexity', 0.5)
        uncertainty_penalty = situational_complexity * 0.3
        projection_gain = base_gain * mental_model_strength * fatigue_impact * (1.0 + experience_bonus) * trend_bonus * (1.0 - uncertainty_penalty)
        confidence_weight = (self.awareness.perception_confidence + self.awareness.comprehension_confidence) / 2
        weighted_gain = projection_gain * confidence_weight
        self.awareness.projection = min(100, self.awareness.projection + weighted_gain)
        base_confidence_gain = 0.03
        confidence_modifier = mental_model_strength * (1.0 - uncertainty_penalty)
        self.awareness.projection_confidence = min(0.8, self.awareness.projection_confidence + (base_confidence_gain * confidence_modifier))

    def apply_sa_decay(self, time_elapsed: int):
        perception_decay_rate = 0.02 * time_elapsed
        comprehension_decay_rate = 0.01 * time_elapsed
        projection_decay_rate = 0.015 * time_elapsed
        self.awareness.perception = max(0, self.awareness.perception - perception_decay_rate)
        self.awareness.comprehension = max(0, self.awareness.comprehension - comprehension_decay_rate)
        self.awareness.projection = max(0, self.awareness.projection - projection_decay_rate)
        confidence_decay = 0.005 * time_elapsed
        self.awareness.perception_confidence = max(0, self.awareness.perception_confidence - confidence_decay)
        self.awareness.comprehension_confidence = max(0, self.awareness.comprehension_confidence - confidence_decay)
        self.awareness.projection_confidence = max(0, self.awareness.projection_confidence - confidence_decay)

    def receive_information(self, information: Dict):
        self.reports_received.append(information)
        credibility = random.uniform(0.3, 0.9)
        if credibility > 0.3:
            if information['type'] == 'observation':
                self.awareness.perception = min(100, self.awareness.perception + 5 * credibility)
            elif information['type'] == 'analysis':
                self.awareness.comprehension = min(100, self.awareness.comprehension + 5 * credibility)
            elif information['type'] == 'prediction':
                self.awareness.projection = min(100, self.awareness.projection + 5 * credibility)

    def observe_event(self, event: Optional[Dict], context: Dict) -> bool:
        if not event:
            return False
        if random.random() < (0.1 + self.fatigue * 0.2):
            return False
        self.update_perception(event, context)
        self.update_comprehension(event, context, self.reports_received)
        self.update_projection(event, context, self.model.event_generator.event_history)
        self.recent_actions['correct_detections'] += 1
        return True

    def decide_action(self, event: Optional[Dict], context: Dict) -> str:
        if not event:
            return "ignore"
        sa_score = self.awareness.total_score()
        decision_thresholds = {
            'report': 50,
            'act': 60,
            'escalate': 70,
            'ignore': 30
        }
        context['action_type'] = 'report'
        modified_prob = self.modify_decision_probability(1.0, context)
        decisions = {
            'report': sa_score >= decision_thresholds['report'],
            'act': sa_score >= decision_thresholds['act'] and context.get('urgency', 0.5) > 0.5,
            'escalate': sa_score >= decision_thresholds['escalate'] and self.role in [AgentRole.MANAGER, AgentRole.DIRECTOR, AgentRole.REPORTER],
            'ignore': sa_score < decision_thresholds['ignore']
        }
        valid_decisions = [d for d, valid in decisions.items() if valid]
        ml_action = self.decision_model.predict_action(self, event, context)
        if ml_action in valid_decisions:
            return ml_action
        if valid_decisions:
            return random.choices(valid_decisions, weights=[modified_prob] * len(valid_decisions))[0]
        return "ignore"

    def step(self):
        try:
            self.apply_sa_decay(1)
            self.update_cognitive_state()
            events = self.model.get_events()
            context = {
                'current_step': self.model.schedule.steps,
                'visibility': random.uniform(0.8, 1.0),
                'noise_level': random.uniform(0.8, 1.0),
                'urgency': random.uniform(0.3, 0.7),
                'domain_relevance': random.uniform(0.5, 1.0),
                'time_pressure': random.uniform(0.0, 0.5),
                'pattern_familiarity': random.uniform(0.3, 0.7),
                'complexity': random.uniform(0.3, 0.7),
                'information_quality': random.uniform(0.5, 1.0)
            }
            observed_events = [event for event in events if self.observe_event(event, context)]

            for event in observed_events:
                action = self.decide_action(event, context)
                self.actions_taken[action] += 1
                self.recent_actions['all_actions'].append(action)

                if self.model.reporting_structure == ReportingStructure.CENTRALIZED and self.role != AgentRole.REPORTER:
                    if action in ["report", "escalate"]:
                        action = "act"
                        self.actions_taken["report"] -= 1
                        self.actions_taken["escalate"] -= 1
                        self.actions_taken["act"] += 1
                elif self.model.reporting_structure == ReportingStructure.DECENTRALIZED:
                    if action == "report" and random.random() > self.reporting_chance:
                        self.actions_taken["report"] -= 1
                        self.actions_taken["ignore"] += 1
                        action = "ignore"
                elif self.model.reporting_structure == ReportingStructure.HYBRID:
                    if action in ["report", "escalate"] and random.random() > 0.5:
                        self.actions_taken["report"] -= 1
                        self.actions_taken["escalate"] -= 1
                        self.actions_taken["ignore"] += 1
                        action = "ignore"

                report_success = False
                if action == "report" and random.random() > self.false_positive_rate:
                    self.reports_sent += 1
                    report = {"agent_id": self.unique_id, "event": event, "action": action, "type": "observation", "quality": 0.8}
                    report_success = self.model.send_report(self, report)
                    self.recent_actions['successful_reports'] += 1 if report_success else 0
                elif action == "escalate" and random.random() > self.false_positive_rate:
                    self.reports_sent += 1
                    report = {"agent_id": self.unique_id, "event": event, "action": action, "type": "prediction", "quality": 0.9}
                    report_success = self.model.send_report(self, report)
                    self.recent_actions['successful_reports'] += 1 if report_success else 0
                elif action == "act":
                    if event["type"] == EventType.HAZARD:
                        if random.random() < 0.2:
                            self.model.outcomes.safety_incidents += 1
                            self.model.outcomes.incident_points += 5
                            self.recent_actions['witnessed_incidents'] += 1
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

                outcome = "success" if action in ["report", "act", "escalate"] and report_success else "failure" if action != "ignore" else "neutral"
                self.decision_model.train_model([{"agent": self, "event": event, "context": context}], [outcome])

            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            self.model.grid.move_agent(self, random.choice(possible_moves))
        except Exception as e:
            print(f"Error in agent {self.unique_id} step: {e}")
            import traceback
            traceback.print_exc()

class WorkerAgent(ConstructionAgent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', pos: tuple):
        super().__init__(unique_id, model, AgentRole.WORKER, pos)
        self.communication_preference = random.choice(['direct', 'formal', 'informal'])
        self.peer_influence_susceptibility = random.uniform(0.1, 0.8)
        self.mobility = random.uniform(0.7, 1.0)
        self.equipment_familiarity = random.uniform(0.5, 1.0)
        self.safety_training_level = random.uniform(0.6, 1.0)

class ManagerAgent(ConstructionAgent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', pos: tuple):
        super().__init__(unique_id, model, AgentRole.MANAGER, pos)
        self.span_of_control = random.randint(3, 8)
        self.delegation_tendency = random.uniform(0.3, 0.8)
        self.risk_tolerance = random.uniform(0.1, 0.6)
        self.decision_authority_level = random.randint(2, 4)
        self.information_synthesis_ability = random.uniform(0.7, 1.0)
        self.prioritization_skill = random.uniform(0.6, 1.0)
        self.communication_effectiveness = random.uniform(0.7, 1.0)
        self.subordinates = []
        self.assign_subordinates()

    def assign_subordinates(self):
        workers = [a for a in self.model.schedule.agents if a.role == AgentRole.WORKER]
        self.subordinates = random.sample(workers, min(self.span_of_control, len(workers)))

    def supervise_subordinates(self):
        for subordinate in self.subordinates:
            if subordinate.awareness.total_score() < 50:
                info = {"type": "analysis", "quality": 0.7, "source": self.unique_id}
                self.model.network.transmit_information(self, subordinate, info)
            if subordinate.workload > 4:
                subordinate.workload = max(1, subordinate.workload - 1)

    def step(self):
        self.supervise_subordinates()
        super().step()

class DirectorAgent(ConstructionAgent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', pos: tuple):
        super().__init__(unique_id, model, AgentRole.DIRECTOR, pos)
        self.strategic_vision_range = 50
        self.resource_allocation_authority = True
        self.policy_making_ability = True
        self.external_stakeholder_communication = True
        self.system_level_awareness = SituationalAwareness()
        self.organizational_memory = []
        self.performance_metrics_tracking = {}

    def make_strategic_decisions(self):
        overall_sa = np.mean([a.awareness.total_score() for a in self.model.schedule.agents])
        if overall_sa < 60:
            alert = {"type": "prediction", "quality": 0.95, "severity": 3}
            self.model.network.broadcast_safety_alert(self, alert)

    def step(self):
        self.make_strategic_decisions()
        super().step()

class ReporterAgent(ConstructionAgent):
    def __init__(self, unique_id: int, model: 'ConstructionModel', pos: tuple):
        super().__init__(unique_id, model, AgentRole.REPORTER, pos)
        self.incident_classification_skill = 0.95
        self.investigation_ability = 0.90
        self.documentation_quality = 0.95
        self.regulatory_knowledge = 0.90
        self.direct_reporting_channels = [a for a in self.model.schedule.agents if a.role in [AgentRole.MANAGER, AgentRole.DIRECTOR]]
        self.external_reporting_capability = True
        self.cross_functional_communication = True

    def conduct_investigation(self, event: Dict) -> Dict:
        return {
            'incident_details': event,
            'contributing_factors': random.uniform(0.7, 1.0),
            'recommendations': random.uniform(0.8, 1.0),
            'regulatory_compliance': self.regulatory_knowledge
        }

    def step(self):
        for event in self.model.get_events():
            if event['severity'].value >= 3:
                report = self.conduct_investigation(event)
                self.model.send_report(self, {"type": "analysis", "quality": self.documentation_quality, "event": report})
        super().step()
