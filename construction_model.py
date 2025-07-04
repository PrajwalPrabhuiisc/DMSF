import pandas as pd
import numpy as np
import random
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from enums import ReportingStructure, OrgStructure, AgentRole, EventType
from data_classes import ProjectOutcomes

# Configure logging
logging.basicConfig(filename='simulation_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ConstructionModel(Model):
    def __init__(self, width: int = 20, height: int = 20, reporting_structure: str = "dedicated",
                 org_structure: str = "functional", hazard_prob: float = 0.05, delay_prob: float = 0.10,
                 resource_prob: float = 0.05, comm_failure_dedicated: float = 0.05, 
                 comm_failure_self: float = 0.05, comm_failure_none: float = 0.10, 
                 worker_detection: float = 0.80, manager_detection: float = 0.90, 
                 reporter_detection: float = 0.95, director_detection: float = 0.85, 
                 worker_reporting: float = 0.80, manager_reporting: float = 0.90, 
                 reporter_reporting: float = 0.95, director_reporting: float = 0.85,
                 initial_budget: float = 1000000, initial_equipment: int = 500, run_id: int = 0):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id:04d}"
        try:
            self.reporting_structure = getattr(ReportingStructure, reporting_structure.upper())
            self.org_structure = getattr(OrgStructure, org_structure.upper())
        except AttributeError as e:
            logging.error(f"Invalid enum value: {e}")
            raise ValueError(f"Invalid reporting_structure ({reporting_structure}) or org_structure ({org_structure})")
        self.hazard_prob = hazard_prob
        self.delay_prob = delay_prob
        self.resource_prob = resource_prob
        self.comm_failure_dedicated = comm_failure_dedicated
        self.comm_failure_self = comm_failure_self
        self.comm_failure_none = comm_failure_none
        self.worker_detection = worker_detection
        self.manager_detection = manager_detection
        self.reporter_detection = reporter_detection
        self.director_detection = director_detection
        self.worker_reporting = worker_reporting
        self.manager_reporting = manager_reporting
        self.reporter_reporting = reporter_reporting
        self.director_reporting = director_reporting
        self.budget = initial_budget
        self.equipment = initial_equipment
        self.outcomes = ProjectOutcomes()
        self.organizational_memory = {et: {'best_action': None, 'success_count': 0} for et in EventType}
        self.configuration_log = [{
            'simulation_id': self.simulation_id,
            'reporting_structure': self.reporting_structure.value,
            'org_structure': self.org_structure.value,
            'hazard_prob': self.hazard_prob,
            'delay_prob': self.delay_prob,
            'resource_prob': self.resource_prob,
            'comm_failure_dedicated': self.comm_failure_dedicated,
            'comm_failure_self': self.comm_failure_self,
            'comm_failure_none': self.comm_failure_none,
            'worker_detection': self.worker_detection,
            'manager_detection': self.manager_detection,
            'reporter_detection': self.reporter_detection,
            'director_detection': self.director_detection,
            'worker_reporting': self.worker_reporting,
            'manager_reporting': self.manager_reporting,
            'reporter_reporting': self.reporter_reporting,
            'director_reporting': self.director_reporting,
            'initial_budget': self.budget,
            'initial_equipment': self.equipment,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }]

        self.datacollector = DataCollector(
            model_reporters={
                "Average_SA": lambda m: np.mean([agent.sa.total_score() for agent in m.schedule.agents]),
                "Worker_SA": lambda m: np.mean([agent.sa.total_score() for agent in m.schedule.agents if agent.role == AgentRole.WORKER]),
                "Manager_SA": lambda m: np.mean([agent.sa.total_score() for agent in m.schedule.agents if agent.role == AgentRole.MANAGER]),
                "Director_SA": lambda m: np.mean([agent.sa.total_score() for agent in m.schedule.agents if agent.role == AgentRole.DIRECTOR]),
                "Reporter_SA": lambda m: np.mean([agent.sa.total_score() for agent in m.schedule.agents if agent.role == AgentRole.REPORTER]) if any(agent.role == AgentRole.REPORTER for agent in m.schedule.agents) else 0,
                "Safety_Incidents": lambda m: m.outcomes.safety_incidents,
                "Incident_Points": lambda m: m.outcomes.incident_points,
                "Schedule_Adherence": lambda m: (m.outcomes.tasks_completed_on_time / m.outcomes.total_tasks * 100) if m.outcomes.total_tasks > 0 else 0,
                "Cost_Overruns": lambda m: m.outcomes.cost_overruns,
                "Worker_Act_Count": lambda m: sum(agent.action_counts.get('act', 0) for agent in m.schedule.agents if agent.role == AgentRole.WORKER),
                "Manager_Act_Count": lambda m: sum(agent.action_counts.get('act', 0) for agent in m.schedule.agents if agent.role == AgentRole.MANAGER),
                "Director_Act_Count": lambda m: sum(agent.action_counts.get('act', 0) for agent in m.schedule.agents if agent.role == AgentRole.DIRECTOR),
                "Reporter_Act_Count": lambda m: sum(agent.action_counts.get('act', 0) for agent in m.schedule.agents if agent.role == AgentRole.REPORTER),
                "Worker_Report_Count": lambda m: sum(agent.action_counts.get('report', 0) for agent in m.schedule.agents if agent.role == AgentRole.WORKER),
                "Manager_Report_Count": lambda m: sum(agent.action_counts.get('report', 0) for agent in m.schedule.agents if agent.role == AgentRole.MANAGER),
                "Director_Report_Count": lambda m: sum(agent.action_counts.get('report', 0) for agent in m.schedule.agents if agent.role == AgentRole.DIRECTOR),
                "Reporter_Report_Count": lambda m: sum(agent.action_counts.get('report', 0) for agent in m.schedule.agents if agent.role == AgentRole.REPORTER),
            },
            agent_reporters={
                "Role": lambda a: a.role.value,
                "SA_Total": lambda a: a.sa.total_score(),
                "SA_Perception": lambda a: a.sa.perception,
                "SA_Comprehension": lambda a: a.sa.comprehension,
                "SA_Projection": lambda a: a.sa.projection,
                "Workload": lambda a: a.workload,
                "Fatigue": lambda a: a.fatigue,
                "Experience": lambda a: a.experience,
                "Stress": lambda a: a.stress,
                "Risk_Tolerance": lambda a: a.risk_tolerance
            }
        )

        self.initialize_agents()
        self.setup_excel_logging()

    def initialize_agents(self):
        from construction_agent import ConstructionAgent  # Local import to avoid circular dependency
        num_workers = 50
        num_managers = random.randint(5, 10)
        num_directors = random.randint(1, 3)
        num_reporters = 5 if self.reporting_structure == ReportingStructure.DEDICATED else 0

        for i in range(num_workers):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            agent = ConstructionAgent(i, self, AgentRole.WORKER, (x, y))
            self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y))

        for i in range(num_workers, num_workers + num_managers):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            agent = ConstructionAgent(i, self, AgentRole.MANAGER, (x, y))
            self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y))

        for i in range(num_workers + num_managers, num_workers + num_managers + num_directors):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            agent = ConstructionAgent(i, self, AgentRole.DIRECTOR, (x, y))
            self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y))

        for i in range(num_workers + num_managers + num_directors, num_workers + num_managers + num_directors + num_reporters):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            agent = ConstructionAgent(i, self, AgentRole.REPORTER, (x, y))
            self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y))

    def setup_excel_logging(self):
        output_dir = os.path.join(os.getcwd(), "simulation_outputs")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logging.debug(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Failed to create output directory {output_dir}: {e}")
                logging.error(f"Directory creation error: {traceback.format_exc()}")

        filename = f"construction_simulation_{self.simulation_id}.xlsx"
        self.excel_filepath = os.path.join(output_dir, filename)
        self.csv_filepath = os.path.join(output_dir, filename.replace('.xlsx', '.csv'))

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with pd.ExcelWriter(self.excel_filepath, engine='openpyxl', mode='w') as writer:
                    pd.DataFrame(self.configuration_log).to_excel(writer, sheet_name='Configuration', index=False)
                logging.debug(f"Initialized new Excel file: {self.excel_filepath}")
                break
            except (PermissionError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                print(f"Error initializing Excel file {self.excel_filepath} after {max_attempts} attempts: {e}")
                logging.error(f"Excel initialization error: {traceback.format_exc()}")
                if self.configuration_log:
                    pd.DataFrame(self.configuration_log).to_csv(self.csv_filepath, index=False)
                    logging.debug(f"Configuration saved to fallback CSV: {self.csv_filepath}")

    def get_events(self):
        events = []
        org_factor = 1.0 if self.org_structure == OrgStructure.FLAT else 1.2 if self.org_structure == OrgStructure.HIERARCHICAL else 1.0
        incident_factor = min(1.5, 1 + 0.1 * self.outcomes.safety_incidents)
        
        effective_hazard_prob = min(self.hazard_prob * org_factor * incident_factor, 0.75)
        effective_delay_prob = min(self.delay_prob * org_factor, 0.80)  # Increased for DELAY
        effective_resource_prob = min(self.resource_prob + 0.05 * self.outcomes.safety_incidents, 0.50)

        event_types = [EventType.HAZARD, EventType.DELAY, EventType.RESOURCE_SHORTAGE]
        weights = [effective_hazard_prob, effective_delay_prob, effective_resource_prob]
        num_events = max(1, np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))  # Increased chance of multiple events
        chosen_types = np.random.choice(event_types, size=num_events, p=np.array(weights)/sum(weights), replace=False)

        for event_type in chosen_types:
            if event_type == EventType.HAZARD:
                severity = random.uniform(0.5, 1.0)
                events.append({"type": EventType.HAZARD, "severity": severity})
            elif event_type == EventType.DELAY:
                severity = random.uniform(0.3, 0.7)
                events.append({"type": EventType.DELAY, "severity": severity})
                self.outcomes.total_tasks += 1
            elif event_type == EventType.RESOURCE_SHORTAGE:
                severity = random.uniform(0.2, 0.5)
                events.append({"type": EventType.RESOURCE_SHORTAGE, "severity": severity})
        
        for event in events:
            if event["type"] == EventType.HAZARD and random.random() < 0.10:  # Reduced probability
                self.outcomes.safety_incidents += 1
                self.outcomes.incident_points += 5 * event["severity"]
                self.outcomes.cost_overruns += 25000 * event["severity"]
                logging.debug(f"Step {self.schedule.steps}: Unaddressed HAZARD caused incident (points={5 * event['severity']:.1f})")
                print(f"Step {self.schedule.steps}: Unaddressed HAZARD caused incident (points={5 * event['severity']:.1f})")
        
        logging.debug(f"Step {self.schedule.steps}: Generated events: {[e['type'].value for e in events]}")
        return events

    def send_report(self, sender, event: Dict):
        from construction_agent import ConstructionAgent  # Local import to avoid circular dependency
        comm_failure = (self.comm_failure_dedicated if self.reporting_structure == ReportingStructure.DEDICATED else
                        self.comm_failure_self if self.reporting_structure == ReportingStructure.SELF else
                        self.comm_failure_none)
        comm_failure *= 0.8 if self.outcomes.safety_incidents > 3 else 0.7  # Further reduced
        comm_failure *= 0.6 if self.org_structure == OrgStructure.FLAT else 1.0 if self.org_structure == OrgStructure.HIERARCHICAL else 0.8

        if random.random() > comm_failure:
            if self.reporting_structure == ReportingStructure.DEDICATED and sender.role == AgentRole.REPORTER:
                for agent in self.schedule.agents:
                    if agent.role != AgentRole.REPORTER:
                        agent.received_reports.append({"type": event["type"], "severity": event["severity"], "acted_on": False})
                        if event["type"] == EventType.DELAY:
                            action = agent.decide_action(event, is_follow_up=True)
                            agent.execute_action(event, action)
                            agent.received_reports[-1]["acted_on"] = True
                            logging.debug(f"Step {self.schedule.steps}: Agent {agent.unique_id} acted on DELAY report from Agent {sender.unique_id}")
            elif self.reporting_structure == ReportingStructure.SELF:
                radius = 3 if self.org_structure == OrgStructure.FLAT else 2
                neighbors = self.grid.get_neighbors(sender.pos, moore=True, radius=radius)
                target_role = AgentRole.MANAGER if sender.role == AgentRole.WORKER else AgentRole.DIRECTOR
                for neighbor in neighbors:
                    if neighbor.role == target_role:
                        neighbor.received_reports.append({"type": event["type"], "severity": event["severity"], "acted_on": False})
                        if event["type"] == EventType.DELAY:
                            action = neighbor.decide_action(event, is_follow_up=True)
                            neighbor.execute_action(event, action)
                            neighbor.received_reports[-1]["acted_on"] = True
                            logging.debug(f"Step {self.schedule.steps}: Agent {neighbor.unique_id} acted on DELAY report from Agent {sender.unique_id}")
                        break
            else:  # ReportingStructure.NONE
                radius = 5
                neighbors = self.grid.get_neighbors(sender.pos, moore=True, radius=radius)
                for neighbor in neighbors:
                    if neighbor.role in [AgentRole.MANAGER, AgentRole.DIRECTOR]:
                        neighbor.received_reports.append({"type": event["type"], "severity": event["severity"], "acted_on": False})
                        if event["type"] == EventType.DELAY:
                            action = neighbor.decide_action(event, is_follow_up=True)
                            neighbor.execute_action(event, action)
                            neighbor.received_reports[-1]["acted_on"] = True
                            logging.debug(f"Step {self.schedule.steps}: Agent {neighbor.unique_id} acted on DELAY report from Agent {sender.unique_id}")
            logging.debug(f"Step {self.schedule.steps}: Report from Agent {sender.unique_id} ({sender.role.value}) sent successfully for {event['type'].value}")
            return True
        logging.debug(f"Step {self.schedule.steps}: Report from Agent {sender.unique_id} ({sender.role.value}) failed for {event['type'].value}")
        return False

    def log_metrics(self):
        try:
            self.datacollector.collect(self)
            logging.debug(f"Step {self.schedule.steps}: Metrics collected - Safety_Incidents={self.outcomes.safety_incidents}, "
                         f"Tasks_Completed={self.outcomes.tasks_completed_on_time}, Total_Tasks={self.outcomes.total_tasks}, "
                         f"Schedule_Adherence={(self.outcomes.tasks_completed_on_time / self.outcomes.total_tasks * 100) if self.outcomes.total_tasks > 0 else 0:.2f}%")
        except Exception as e:
            logging.error(f"Error in log_metrics at step {self.schedule.steps}: {traceback.format_exc()}")
            print(f"Error in log_metrics at step {self.schedule.steps}: {e}")

    def save_to_excel(self):
        max_attempts = 5  # Increased attempts
        for attempt in range(max_attempts):
            try:
                mode = 'a' if os.path.exists(self.excel_filepath) else 'w'
                with pd.ExcelWriter(self.excel_filepath, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
                    model_data = self.datacollector.get_model_vars_dataframe().reset_index().rename(columns={'index': 'Step'})
                    agent_data = self.datacollector.get_agent_vars_dataframe().reset_index().rename(columns={'index': 'Step'})
                    model_data.to_excel(writer, sheet_name='Model_Metrics', index=False)
                    agent_data.to_excel(writer, sheet_name='Agent_SA', index=False)
                logging.debug(f"Step {self.schedule.steps}: Data saved to Excel: {self.excel_filepath}, Rows={len(model_data)}")
                break
            except (PermissionError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.3 * (attempt + 1))  # Increased sleep
                    continue
                print(f"Error saving to Excel {self.excel_filepath} after {max_attempts} attempts: {e}")
                logging.error(f"Excel save error: {traceback.format_exc()}")
                model_data = self.datacollector.get_model_vars_dataframe().reset_index().rename(columns={'index': 'Step'})
                agent_data = self.datacollector.get_agent_vars_dataframe().reset_index().rename(columns={'index': 'Step'})
                model_data.to_csv(self.csv_filepath.replace('.csv', '_model_metrics.csv'), index=False)
                agent_data.to_csv(self.csv_filepath.replace('.csv', '_agent_sa.csv'), index=False)
                logging.debug(f"Data saved to fallback CSV: {self.csv_filepath}")

    def step(self):
        try:
            events = self.get_events()
            for agent in self.schedule.agents:
                agent.step()
            self.log_metrics()
        except Exception as e:
            logging.error(f"Error in model step {self.schedule.steps}: {traceback.format_exc()}")
            print(f"Error in model step {self.schedule.steps}: {e}")

    def run_simulation(self, steps=150):
        try:
            for i in range(steps):
                logging.debug(f"Starting step {i}")
                self.step()
                if i % 10 == 0:
                    self.budget += 10000
                    self.equipment += 10
                self.save_to_excel()
            logging.debug(f"Simulation {self.simulation_id} completed {steps} steps")
        except Exception as e:
            logging.error(f"Error in run_simulation: {traceback.format_exc()}")
            print(f"Error in run_simulation: {e}")
        finally:
            self.save_to_excel()
