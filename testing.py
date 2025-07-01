import logging
import pandas as pd
import mesa
import random
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from enums import AgentRole, EventType, OrgStructure, ReportingStructure
from data_classes import SituationalAwareness
from decision_model import DecisionModel
from construction_agent import ConstructionAgent

# Configure logging
logging.basicConfig(filename='simulation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SimulationOutcomes:
    def __init__(self):
        self.safety_incidents = 0
        self.delays = 0
        self.resource_issues = 0

class ConstructionModel(mesa.Model):
    def __init__(self, num_workers=10, num_reporters=2, num_managers=2, num_directors=1,
                 org_structure=OrgStructure.HIERARCHICAL,
                 reporting_structure=ReportingStructure.CENTRALIZED,
                 width=10, height=10):
        super().__init__()
        self.num_agents = num_workers + num_reporters + num_managers + num_directors
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.org_structure = org_structure
        self.reporting_structure = reporting_structure
        self.budget = 1000000
        self.equipment = 500
        self.outcomes = SimulationOutcomes()
        self.current_events = []
        self.trial_id = str(uuid.uuid4())
        self.excel_file = f"simulation_trial_{self.trial_id}.xlsx"
        self.data_log = []
        self.tick_count = 0

        # Agent detection and reporting probabilities
        self.worker_detection = 0.6
        self.worker_reporting = 0.5
        self.reporter_detection = 0.9
        self.reporter_reporting = 0.8
        self.manager_detection = 0.7
        self.manager_reporting = 0.6
        self.director_detection = 0.8
        self.director_reporting = 0.7

        # Initialize agents
        agent_id = 0
        for _ in range(num_workers):
            pos = (random.randrange(self.grid.width), random.randrange(self.grid.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.WORKER, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1

        for _ in range(num_reporters):
            pos = (random.randrange(self.grid.width), random.randrange(self.grid.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.REPORTER, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1

        for _ in range(num_managers):
            pos = (random.randrange(self.grid.width), random.randrange(self.grid.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.MANAGER, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1

        for _ in range(num_directors):
            pos = (random.randrange(self.grid.width), random.randrange(self.grid.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.DIRECTOR, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1

    def get_events(self) -> List[Dict]:
        return self.current_events

    def generate_events(self):
        self.current_events = []
        if random.random() < 0.1:
            event = {
                "type": random.choice([EventType.HAZARD, EventType.DELAY, EventType.RESOURCE_SHORTAGE]),
                "severity": random.uniform(0.1, 1.0)
            }
            self.current_events.append(event)
            logging.info(f"Step {self.schedule.steps}: Generated event {event['type'].value} with severity {event['severity']:.2f}")

    def log_agent_data(self):
        for agent in self.schedule.agents:
            agent_data = {
                "tick": self.tick_count,
                "agent_id": agent.unique_id,
                "role": agent.role.value,
                "workload": agent.workload,
                "fatigue": agent.fatigue,
                "experience": agent.experience,
                "risk_tolerance": agent.risk_tolerance,
                "stress": agent.stress,
                "perception": agent.sa.perception,
                "comprehension": agent.sa.comprehension,
                "projection": agent.sa.projection,
                "action_counts_ignore": agent.action_counts["ignore"],
                "action_counts_report": agent.action_counts["report"],
                "action_counts_act": agent.action_counts["act"],
                "action_counts_escalate": agent.action_counts["escalate"],
                "last_event_action": agent.last_event_action or "none",
                "reports_sent": agent.reports_sent,
                "received_reports_count": len(agent.received_reports)
            }
            self.data_log.append(agent_data)

    def save_to_excel(self):
        if self.data_log:
            df = pd.DataFrame(self.data_log)
            try:
                df.to_excel(self.excel_file, index=False)
                logging.info(f"Saved simulation data to {self.excel_file}")
            except Exception as e:
                logging.error(f"Error saving to Excel: {e}")

    def step(self):
        self.tick_count += 1
        self.generate_events()
        self.schedule.step()

        # Log data only at 150 ticks
        if self.tick_count == 150:
            self.log_agent_data()
            self.save_to_excel()
            logging.info(f"Step {self.schedule.steps}: Reached 150 ticks, data logged and saved to {self.excel_file}")
