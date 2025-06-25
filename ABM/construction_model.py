import mesa
import random
import numpy as np
import pandas as pd
from datetime import datetime
import os
import traceback
import logging
from typing import List, Dict, Any
from enums import ReportingStructure, AgentRole, EventType
from data_classes import ProjectOutcomes
from construction_agent import WorkerAgent, ManagerAgent, DirectorAgent, ReporterAgent
from communication_network import CommunicationNetwork
from event_system import EventGenerator

logging.getLogger("tornado.access").setLevel(logging.ERROR)

class ConstructionModel(mesa.Model):
    def __init__(self, width: int = 20, height: int = 20, reporting_structure: str = "centralized",
                 hazard_prob: float = 0.05, delay_prob: float = 0.15, resource_prob: float = 0.03,
                 num_workers: int = 50, num_managers: int = 10, num_directors: int = 3, num_reporters: int = 5):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = mesa.space.MultiGrid(width, height, torus=True)
        self.schedule = mesa.time.RandomActivation(self)
        self.ReportingStructure = ReportingStructure
        self.reporting_structure = ReportingStructure(reporting_structure.lower())
        self.outcomes = ProjectOutcomes()
        self.reports = []
        self.current_events = []
        self.event_counts = {et: 0 for et in EventType}
        self.hazard_prob = hazard_prob
        self.delay_prob = delay_prob
        self.resource_prob = resource_prob
        self.num_workers = num_workers
        self.num_managers = num_managers
        self.num_directors = num_directors
        self.num_reporters = num_reporters
        
        self.setup_excel_logging()
        self.event_generator = EventGenerator(self)
        self.metrics_log = []
        self.agent_sa_log = []
        self.configuration_log = []
        self.network = None  # Initialized after agents
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "SafetyIncidents": lambda m: m.outcomes.safety_incidents,
                "IncidentPoints": lambda m: m.outcomes.incident_points,
                "ScheduleAdherence": lambda m: (m.outcomes.tasks_completed_on_time / m.outcomes.total_tasks
                                               if m.outcomes.total_tasks > 0 else 0),
                "CostOverruns": lambda m: m.outcomes.cost_overruns,
                "AverageSA": lambda m: np.mean([a.awareness.total_score() for a in m.schedule.agents]),
                "Worker_SA": lambda m: np.mean([a.awareness.total_score() for a in m.schedule.agents
                                               if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0,
                "Manager_SA": lambda m: np.mean([a.awareness.total_score() for a in m.schedule.agents
                                                if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0,
                "Director_SA": lambda m: np.mean([a.awareness.total_score() for a in m.schedule.agents
                                                 if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0,
                "Reporter_SA": lambda m: np.mean([a.awareness.total_score() for a in m.schedule.agents
                                                 if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0,
            },
            agent_reporters={
                "Role": lambda a: a.role.value,
                "SA_Score": lambda a: a.awareness.total_score(),
                "ReportsSent": lambda a: a.reports_sent,
                "Workload": lambda a: a.workload,
                "Fatigue": lambda a: a.fatigue
            }
        )
        self.initialize_agents()
        self.network = CommunicationNetwork(self.schedule.agents, self.reporting_structure)
        self.log_configuration()

    def setup_excel_logging(self):
        output_dir = os.path.join(os.getcwd(), "simulation_outputs")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Failed to create output directory {output_dir}: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"construction_ABM_{self.reporting_structure.value}_{timestamp}.xlsx"
        self.excel_filepath = os.path.join(output_dir, filename)
        print(f"Model initialized with reporting_structure: {self.reporting_structure.value}")
        print(f"Excel filepath: {self.excel_filepath}")

    def log_configuration(self):
        config_data = {
            "Parameter": [
                "Reporting_Structure", "Hazard_Probability", "Delay_Probability", "Resource_Probability",
                "Num_Workers", "Num_Managers", "Num_Directors", "Num_Reporters",
                "Grid_Width", "Grid_Height", "Timestamp"
            ],
            "Value": [
                self.reporting_structure.value, self.hazard_prob, self.delay_prob, self.resource_prob,
                self.num_workers, self.num_managers, self.num_directors, self.num_reporters,
                self.width, self.height, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        self.configuration_log = config_data

    def initialize_agents(self):
        agent_id = 0
        for _ in range(self.num_workers):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = WorkerAgent(agent_id, self, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1
        for _ in range(self.num_managers):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = ManagerAgent(agent_id, self, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1
        for _ in range(self.num_directors):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = DirectorAgent(agent_id, self, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1
        for _ in range(self.num_reporters):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = ReporterAgent(agent_id, self, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1

    def get_events(self) -> List[Dict]:
        self.current_events = self.event_generator.generate_events(self.schedule.steps)
        for event in self.current_events:
            self.event_counts[event['type']] += 1
            if event['type'] == EventType.HAZARD and random.random() < 0.05:
                self.outcomes.safety_incidents += 1
                self.outcomes.incident_points += 5 * event['severity'].value
                self.outcomes.cost_overruns += 25000 * (event['severity'].value / 4.0)
        return self.current_events if self.current_events else [None]

    def send_report(self, sender: Any, report: Dict) -> bool:
        self.reports.append(report)
        success = False
        target_roles = []
        
        if self.reporting_structure == ReportingStructure.CENTRALIZED and sender.role != AgentRole.REPORTER:
            target_roles = [AgentRole.REPORTER, AgentRole.MANAGER, AgentRole.DIRECTOR]
        elif self.reporting_structure == ReportingStructure.DECENTRALIZED:
            target_roles = [AgentRole.WORKER, AgentRole.MANAGER, AgentRole.DIRECTOR, AgentRole.REPORTER]
        elif self.reporting_structure == ReportingStructure.HYBRID:
            target_roles = [AgentRole.MANAGER, AgentRole.DIRECTOR] if sender.role == AgentRole.WORKER else [AgentRole.DIRECTOR]

        neighbors = self.grid.get_neighbors(sender.pos, moore=True, include_center=False, radius=3)
        for agent in self.schedule.agents:
            if (agent in neighbors or agent.role in target_roles) and agent != sender:
                if self.network.transmit_information(sender, agent, report):
                    success = True
        return success

    def log_metrics(self):
        metrics_data = {
            "Step": self.schedule.steps,
            "Reporting_Structure": self.reporting_structure.value,
            "Worker_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0,
            "Manager_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0,
            "Director_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0,
            "Reporter_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0,
            "Safety_Incidents": self.outcomes.safety_incidents,
            "Incident_Points": self.outcomes.incident_points,
            "Schedule_Adherence": (self.outcomes.tasks_completed_on_time / self.outcomes.total_tasks
                                  if self.outcomes.total_tasks > 0 else 0),
            "Cost_Overruns": self.outcomes.cost_overruns,
            **self.network.calculate_network_metrics()
        }
        self.metrics_log.append(metrics_data)

    def log_agent_situational_awareness(self):
        for agent in self.schedule.agents:
            sa_data = {
                "Step": self.schedule.steps,
                "Agent_ID": agent.unique_id,
                "Role": agent.role.value,
                "SA_Perception": agent.awareness.perception,
                "SA_Comprehension": agent.awareness.comprehension,
                "SA_Projection": agent.awareness.projection,
                "SA_Total": agent.awareness.total_score(),
                "Workload": agent.workload,
                "Fatigue": agent.fatigue,
                "Reports_Sent": agent.reports_sent,
                "Reports_Received": len(agent.reports_received),
                "Position_X": agent.pos[0],
                "Position_Y": agent.pos[1],
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.agent_sa_log.append(sa_data)

    def save_to_excel(self):
        try:
            with pd.ExcelWriter(self.excel_filepath, engine='openpyxl') as writer:
                config_df = pd.DataFrame(self.configuration_log)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
                if self.metrics_log:
                    metrics_df = pd.DataFrame(self.metrics_log)
                    metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
                if self.agent_sa_log:
                    agent_df = pd.DataFrame(self.agent_sa_log)
                    agent_df.to_excel(writer, sheet_name='Agent_SA', index=False)
                model_data = self.datacollector.get_model_vars_dataframe()
                if not model_data.empty:
                    model_data.to_excel(writer, sheet_name='Mesa_Model_Data')
                agent_data = self.datacollector.get_agent_vars_dataframe()
                if not agent_data.empty:
                    agent_data.to_excel(writer, sheet_name='Mesa_Agent_Data')
                print(f"Data saved to {self.excel_filepath} at step {self.schedule.steps}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            traceback.print_exc()

    def step(self):
        try:
            print(f"Executing step {self.schedule.steps + 1}")
            self.schedule.step()
            self.network.update_edges()
            self.datacollector.collect(self)
            self.log_metrics()
            if self.schedule.steps % 10 == 0:
                self.log_agent_situational_awareness()
            if self.schedule.steps % 50 == 0 and self.schedule.steps > 0:
                self.save_to_excel()
            print(f"Completed step {self.schedule.steps}")
        except Exception as e:
            print(f"Error in model step {self.schedule.steps + 1}: {e}")
            traceback.print_exc()

    def run_simulation(self, steps: int = 500):
        print(f"Starting simulation with {steps} steps...")
        for i in range(steps):
            self.step()
            if (i + 1) % 25 == 0:
                print(f"Completed step {i + 1}/{steps}")
        self.save_to_excel()
        print(f"Simulation completed. Results saved to {self.excel_filepath}")
        self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        print(f"Total Steps: {self.schedule.steps}")
        print(f"Reporting Structure: {self.reporting_structure.value}")
        print("\nSafety Outcomes:")
        print(f"  Safety Incidents: {self.outcomes.safety_incidents}")
        print(f"  Incident Points: {self.outcomes.incident_points}")
        print(f"  Cost Overruns: ${self.outcomes.cost_overruns:,.2f}")
        print("\nSchedule Performance:")
        if self.outcomes.total_tasks > 0:
            adherence = (self.outcomes.tasks_completed_on_time / self.outcomes.total_tasks) * 100
            print(f"  Schedule Adherence: {adherence:.1f}%")
        else:
            print("  No tasks recorded")
        print("\nSituational Awareness (Average):")
        for role in AgentRole:
            agents = [a for a in self.schedule.agents if a.role == role]
            if agents:
                avg_sa = np.mean([a.awareness.total_score() for a in agents])
                print(f"  {role.value.capitalize()}: {avg_sa:.2f}")