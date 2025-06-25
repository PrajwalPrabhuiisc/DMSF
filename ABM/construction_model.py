import mesa
import random
import numpy as np
import pandas as pd
from datetime import datetime
import os
import traceback
import logging
from typing import List, Dict
from enums import ReportingStructure, OrgStructure, AgentRole, EventType
from data_classes import ProjectOutcomes
from construction_agent import ConstructionAgent

logging.getLogger("tornado.access").setLevel(logging.ERROR)

class ConstructionModel(mesa.Model):
    def __init__(self, width: int = 20, height: int = 20, reporting_structure: str = "dedicated",
                 org_structure: str = "functional",
                 hazard_prob: float = 0.10, delay_prob: float = 0.35, resource_prob: float = 0.05,
                 comm_failure_dedicated: float = 0.05, comm_failure_self: float = 0.05, comm_failure_none: float = 0.10,
                 worker_detection: float = 0.80, manager_detection: float = 0.90, reporter_detection: float = 0.95,
                 director_detection: float = 0.90,
                 worker_reporting: float = 0.80, manager_reporting: float = 0.90, reporter_reporting: float = 0.95,
                 director_reporting: float = 0.85,
                 initial_budget: float = 1000000, initial_equipment: int = 50):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.ReportingStructure = ReportingStructure
        self.OrgStructure = OrgStructure
        self.reporting_structure = ReportingStructure(reporting_structure.lower())
        self.org_structure = OrgStructure(org_structure.lower())
        self.outcomes = ProjectOutcomes()
        self.reports = []
        self.current_events = []
        self.event_counts = {EventType.HAZARD: 0, EventType.DELAY: 0, EventType.RESOURCE_SHORTAGE: 0}
        self.base_hazard_prob = hazard_prob
        self.base_delay_prob = delay_prob
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
        self.equipment_available = initial_equipment
        self.organizational_memory = {EventType.HAZARD: [], EventType.DELAY: [], EventType.RESOURCE_SHORTAGE: []}
        
        self.setup_excel_logging()
        
        self.metrics_log = []
        self.agent_sa_log = []
        self.configuration_log = []
        
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
                "Worker_Reports_Sent": lambda m: np.sum([a.reports_sent for a in m.schedule.agents
                                                        if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0,
                "Manager_Reports_Sent": lambda m: np.sum([a.reports_sent for a in m.schedule.agents
                                                         if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0,
                "Director_Reports_Sent": lambda m: np.sum([a.reports_sent for a in m.schedule.agents
                                                          if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0,
                "Reporter_Reports_Sent": lambda m: np.sum([a.reports_sent for a in m.schedule.agents
                                                          if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0,
                "Worker_Reports_Received": lambda m: np.sum([len(a.reports_received) for a in m.schedule.agents
                                                            if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0,
                "Manager_Reports_Received": lambda m: np.sum([len(a.reports_received) for a in m.schedule.agents
                                                             if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0,
                "Director_Reports_Received": lambda m: np.sum([len(a.reports_received) for a in m.schedule.agents
                                                              if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0,
                "Reporter_Reports_Received": lambda m: np.sum([len(a.reports_received) for a in m.schedule.agents
                                                              if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0,
                "TotalTasks": lambda m: m.outcomes.total_tasks,
                "TasksCompletedOnTime": lambda m: m.outcomes.tasks_completed_on_time,
                "BudgetRemaining": lambda m: m.budget,
                "EquipmentAvailable": lambda m: m.equipment_available,
                "Worker_Act_Count": lambda m: np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0,
                "Manager_Act_Count": lambda m: np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0,
                "Director_Act_Count": lambda m: np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0,
                "Reporter_Act_Count": lambda m: np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0
            },
            agent_reporters={
                "Role": lambda a: a.role.value,
                "SA_Score": lambda a: a.awareness.total_score(),
                "ReportsSent": lambda a: a.reports_sent,
                "Workload": lambda a: a.workload,
                "Fatigue": lambda a: a.fatigue,
                "Experience": lambda a: a.experience,
                "RiskTolerance": lambda a: a.risk_tolerance
            }
        )
        self.initialize_agents()
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
        filename_parts = [
            f"construction_ABM",
            f"struct_{self.reporting_structure.value}",
            f"org_{self.org_structure.value}",
            f"hazard_{self.base_hazard_prob:.3f}",
            f"delay_{self.base_delay_prob:.3f}",
            f"resource_{self.resource_prob:.3f}",
            f"comm_{self.get_current_comm_failure():.3f}",
            f"wdet_{self.worker_detection:.2f}",
            f"mdet_{self.manager_detection:.2f}",
            f"ddet_{self.director_detection:.2f}",
            timestamp
        ]
        filename = "_".join(filename_parts) + ".xlsx"
        self.excel_filepath = os.path.join(output_dir, filename)
        print(f"Model initialized with reporting_structure: {self.reporting_structure.value}, org_structure: {self.org_structure.value}")
        print(f"Excel filepath: {self.excel_filepath}")

    def get_current_comm_failure(self):
        base_comm_failure = (
            self.comm_failure_dedicated if self.reporting_structure == ReportingStructure.DEDICATED else
            self.comm_failure_self if self.reporting_structure == ReportingStructure.SELF else
            self.comm_failure_none
        )
        if self.org_structure == OrgStructure.FLAT:
            return base_comm_failure * 0.8
        elif self.org_structure == OrgStructure.HIERARCHICAL:
            return base_comm_failure * 1.2
        return base_comm_failure

    def log_configuration(self):
        config_data = {
            "Parameter": [
                "Reporting_Structure", "Org_Structure", "Hazard_Probability", "Delay_Probability", "Resource_Shortage_Probability",
                "Comm_Failure_Dedicated", "Comm_Failure_Self", "Comm_Failure_None",
                "Worker_Detection", "Manager_Detection", "Reporter_Detection", "Director_Detection",
                "Worker_Reporting", "Manager_Reporting", "Reporter_Reporting", "Director_Reporting",
                "Initial_Budget", "Initial_Equipment", "Grid_Width", "Grid_Height", "Timestamp"
            ],
            "Value": [
                self.reporting_structure.value, self.org_structure.value, self.base_hazard_prob, self.base_delay_prob, self.resource_prob,
                self.comm_failure_dedicated, self.comm_failure_self, self.comm_failure_none,
                self.worker_detection, self.manager_detection, self.reporter_detection, self.director_detection,
                self.worker_reporting, self.manager_reporting, self.reporter_reporting, self.director_reporting,
                self.budget, self.equipment_available, self.width, self.height, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        self.configuration_log = config_data

    def initialize_agents(self):
        for agent in self.schedule.agents[:]:
            self.schedule.remove(agent)
            self.grid.remove_agent(agent)
        
        agent_id = 0
        if self.reporting_structure == ReportingStructure.DEDICATED:
            for _ in range(5):
                pos = (random.randrange(self.width), random.randrange(self.height))
                agent = ConstructionAgent(agent_id, self, AgentRole.REPORTER, pos)
                self.schedule.add(agent)
                self.grid.place_agent(agent, pos)
                agent_id += 1
        worker_count = 50
        manager_count = 10 if self.org_structure != OrgStructure.FLAT else 5
        director_count = 3 if self.org_structure != OrgStructure.FLAT else 1
        for _ in range(worker_count):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.WORKER, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1
        for _ in range(manager_count):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.MANAGER, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1
        for _ in range(director_count):
            pos = (random.randrange(self.width), random.randrange(self.height))
            agent = ConstructionAgent(agent_id, self, AgentRole.DIRECTOR, pos)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)
            agent_id += 1

    def get_events(self) -> List[Dict]:
        self.event_counts = {EventType.HAZARD: 0, EventType.DELAY: 0, EventType.RESOURCE_SHORTAGE: 0}
        self.current_events = []
        hazard_prob = max(self.base_hazard_prob * (0.95 ** self.outcomes.safety_incidents), 0.05)
        delay_prob = min(self.base_delay_prob * (1 + 0.1 * (1 - min(self.budget / 1000000, 1.0))), 0.5)
        effective_resource_prob = self.resource_prob

        # Generate events at random grid positions
        if random.random() < hazard_prob:
            event_pos = (random.randrange(self.width), random.randrange(self.height))
            self.current_events.append({"type": EventType.HAZARD, "description": "Loose scaffold", "pos": event_pos})
            self.event_counts[EventType.HAZARD] += 1
            delay_prob += 0.05
            effective_resource_prob += 0.03
            print(f"Step {self.schedule.steps}: Generated HAZARD event at {event_pos}")

        if random.random() < delay_prob:
            event_pos = (random.randrange(self.width), random.randrange(self.height))
            self.current_events.append({"type": EventType.DELAY, "description": "Supply chain delay", "pos": event_pos})
            self.event_counts[EventType.DELAY] += 1
            effective_resource_prob += 0.02
            print(f"Step {self.schedule.steps}: Generated DELAY event at {event_pos}")

        if random.random() < effective_resource_prob:
            event_pos = (random.randrange(self.width), random.randrange(self.height))
            self.current_events.append({"type": EventType.RESOURCE_SHORTAGE, "description": "Material unavailability", "pos": event_pos})
            self.event_counts[EventType.RESOURCE_SHORTAGE] += 1
            print(f"Step {self.schedule.steps}: Generated RESOURCE_SHORTAGE event at {event_pos}")

        for event in self.current_events:
            if event["type"] == EventType.HAZARD and random.random() < 0.02:
                self.outcomes.safety_incidents += 1
                self.outcomes.incident_points += 5
                self.outcomes.cost_overruns += 25000
                print(f"Step {self.schedule.steps}: HAZARD triggered safety incident (unaddressed) at {event['pos']}")

        if not self.current_events:
            print(f"Step {self.schedule.steps}: No events generated")
        return self.current_events if self.current_events else [None]

    def send_report(self, sender: ConstructionAgent, report: Dict) -> bool:
        self.reports.append(report)
        comm_failure = self.get_current_comm_failure()
        success = False
        target_roles = []
        event = report.get("event")

        if self.reporting_structure == ReportingStructure.DEDICATED and sender.role != AgentRole.REPORTER:
            target_roles = [AgentRole.REPORTER]
        elif self.reporting_structure == ReportingStructure.SELF:
            if self.org_structure == OrgStructure.HIERARCHICAL:
                target_roles = [AgentRole.MANAGER if sender.role == AgentRole.WORKER else AgentRole.DIRECTOR]
            else:
                target_roles = [AgentRole.WORKER, AgentRole.MANAGER, AgentRole.DIRECTOR]
        elif self.reporting_structure == ReportingStructure.NONE:
            target_roles = [AgentRole.WORKER, AgentRole.MANAGER, AgentRole.DIRECTOR]

        neighbors = self.grid.get_neighbors(sender.pos, moore=True, include_center=False, radius=5)  # Increased radius
        if sender.role == AgentRole.REPORTER:
            if self.reporting_structure == ReportingStructure.DEDICATED:
                for agent in self.schedule.agents:
                    if agent.role != AgentRole.REPORTER and random.random() > comm_failure:
                        agent.reports_received.append(report)
                        agent.awareness.perception = min(agent.awareness.perception + self.reporter_detection * 20, 100)
                        agent.awareness.comprehension = min(agent.awareness.comprehension + self.reporter_detection * 10, 100)
                        agent.awareness.projection = min(agent.awareness.projection + self.reporter_detection * 5, 100)
                        success = True
                        if event and not report.get("acted_on", False):
                            follow_up_action = agent.decide_action(event, is_follow_up=True)
                            agent.execute_action(event, follow_up_action)
                            report["acted_on"] = True
                            print(f"Step {self.schedule.steps}: Agent {agent.unique_id} ({agent.role.value}) executed follow-up action {follow_up_action} for reported event {event['type'].value}")
                        print(f"Step {self.schedule.steps}: Reporter {sender.unique_id} shared SA with Agent {agent.unique_id}")
            else:
                if random.random() > comm_failure:
                    sender.reports_received.append(report)
                    success = True
                    print(f"Step {self.schedule.steps}: Reporter {sender.unique_id} sent report to self")
        else:
            for agent in self.schedule.agents:
                if agent in neighbors and agent.role in target_roles:
                    if random.random() > comm_failure:
                        agent.reports_received.append(report)
                        success = True
                        if event and not report.get("acted_on", False):
                            follow_up_action = agent.decide_action(event, is_follow_up=True)
                            agent.execute_action(event, follow_up_action)
                            report["acted_on"] = True
                            print(f"Step {self.schedule.steps}: Agent {agent.unique_id} ({agent.role.value}) executed follow-up action {follow_up_action} for reported event {event['type'].value}")
                        print(f"Step {self.schedule.steps}: Agent {sender.unique_id} ({sender.role.value}) sent report to Agent {agent.unique_id} ({agent.role.value})")
        return success

    def log_metrics(self):
        comm_failure = self.get_current_comm_failure()
        schedule_adherence = (self.outcomes.tasks_completed_on_time / self.outcomes.total_tasks
                              if self.outcomes.total_tasks > 0 else 0)
        metrics_data = {
            "Step": self.schedule.steps,
            "Reporting_Structure": self.reporting_structure.value,
            "Org_Structure": self.org_structure.value,
            "Worker_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0,
            "Manager_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0,
            "Director_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0,
            "Reporter_SA": np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0,
            "Worker_Reports_Sent": np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0,
            "Manager_Reports_Sent": np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0,
            "Director_Reports_Sent": np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0,
            "Reporter_Reports_Sent": np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0,
            "Worker_Reports_Received": np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0,
            "Manager_Reports_Received": np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0,
            "Director_Reports_Received": np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0,
            "Reporter_Reports_Received": np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0,
            "Safety_Incidents": self.outcomes.safety_incidents,
            "Incident_Points": self.outcomes.incident_points,
            "Schedule_Adherence": schedule_adherence,
            "Cost_Overruns": self.outcomes.cost_overruns,
            "Comm_Failure_Rate": comm_failure,
            "Hazard_Events": self.event_counts[EventType.HAZARD],
            "Delay_Events": self.event_counts[EventType.DELAY],
            "Resource_Shortage_Events": self.event_counts[EventType.RESOURCE_SHORTAGE],
            "Total_Tasks": self.outcomes.total_tasks,
            "Tasks_Completed_On_Time": self.outcomes.tasks_completed_on_time,
            "Budget_Remaining": self.budget,
            "Equipment_Available": self.equipment_available,
            "Worker_Act_Count": np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0,
            "Manager_Act_Count": np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0,
            "Director_Act_Count": np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0,
            "Reporter_Act_Count": np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0
        }
        self.metrics_log.append(metrics_data)
        print(f"Step {self.schedule.steps}: Schedule Adherence={schedule_adherence:.2f}, Total Tasks={self.outcomes.total_tasks}, Completed On Time={self.outcomes.tasks_completed_on_time}, "
              f"Safety Incidents={self.outcomes.safety_incidents}, Worker Act={metrics_data['Worker_Act_Count']}, Manager Act={metrics_data['Manager_Act_Count']}, "
              f"Director SA={metrics_data['Director_SA']:.2f}, Worker SA={metrics_data['Worker_SA']:.2f}, Budget={self.budget:.2f}, Equipment={self.equipment_available}")

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
                "Experience": agent.experience,
                "RiskTolerance": agent.risk_tolerance,
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
                
                print(f"Data successfully saved to {self.excel_filepath} at step {self.schedule.steps}")
                
        except Exception as e:
            print(f"Error saving to Excel at step {self.schedule.steps}: {e}")
            traceback.print_exc()

    def step(self):
        try:
            print(f"Executing step {self.schedule.steps + 1}")
            self.schedule.step()
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

    def run_simulation(self, steps: int = 100):
        print(f"Starting simulation with {steps} steps...")
        print(f"Reporting structure: {self.reporting_structure.value}, Org structure: {self.org_structure.value}")
        print(f"Agent counts: Workers={len([a for a in self.schedule.agents if a.role == AgentRole.WORKER])}, "
              f"Managers={len([a for a in self.schedule.agents if a.role == AgentRole.MANAGER])}, "
              f"Directors={len([a for a in self.schedule.agents if a.role == AgentRole.DIRECTOR])}, "
              f"Reporters={len([a for a in self.schedule.agents if a.role == AgentRole.REPORTER])}")
        
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
        print(f"Organizational Structure: {self.org_structure.value}")
        print(f"Communication Failure Rate: {self.get_current_comm_failure():.3f}")
        print(f"Budget Remaining: ${self.budget:,.2f}")
        print(f"Equipment Available: {self.equipment_available}")
        print("\nSafety Outcomes:")
        print(f"  Safety Incidents: {self.outcomes.safety_incidents}")
        print(f"  Incident Points: {self.outcomes.incident_points}")
        print(f"  Cost Overruns: ${self.outcomes.cost_overruns:,.2f}")
        
        print("\nSchedule Performance:")
        if self.outcomes.total_tasks > 0:
            adherence = (self.outcomes.tasks_completed_on_time / self.outcomes.total_tasks) * 100
            print(f"  Schedule Adherence: {adherence:.1f}%")
            print(f"  Tasks Completed On Time: {self.outcomes.tasks_completed_on_time}/{self.outcomes.total_tasks}")
        else:
            print("  No tasks recorded")
        
        print("\nSituational Awareness (Average):")
        for role in AgentRole:
            agents_of_role = [a for a in self.schedule.agents if a.role == role]
            if agents_of_role:
                avg_sa = np.mean([a.awareness.total_score() for a in agents_of_role])
                print(f"  {role.value.capitalize()}: {avg_sa:.2f}")
        
        print("\nReporting Activity:")
        for role in AgentRole:
            agents_of_role = [a for a in self.schedule.agents if a.role == role]
            if agents_of_role:
                total_sent = sum([a.reports_sent for a in agents_of_role])
                total_received = sum([len(a.reports_received) for a in agents_of_role])
                print(f"  {role.value.capitalize()}: Sent={total_sent}, Received={total_received}")
