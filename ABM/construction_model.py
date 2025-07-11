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

logging.basicConfig(filename='simulation_errors.log', level=logging.DEBUG)

class ConstructionModel(mesa.Model):
    def __init__(self, width: int = 20, height: int = 20, reporting_structure: str = "dedicated",
                 org_structure: str = "functional",
                 hazard_prob: float = 0.05, delay_prob: float = 0.10, resource_prob: float = 0.05,
                 comm_failure_dedicated: float = 0.05, comm_failure_self: float = 0.05, comm_failure_none: float = 0.10,
                 worker_detection: float = 0.80, manager_detection: float = 0.90, reporter_detection: float = 0.95,
                 director_detection: float = 0.85,
                 worker_reporting: float = 0.80, manager_reporting: float = 0.90, reporter_reporting: float = 0.95,
                 director_reporting: float = 0.85,
                 initial_budget: float = 1000000, initial_equipment: int = 500):
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
        self.previous_adherence = 0.0
        self.simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._initialized = False  # Flag to prevent reinitialization
        
        # Initialize DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "SafetyIncidents": lambda m: m.outcomes.safety_incidents,
                "IncidentPoints": lambda m: m.outcomes.incident_points,
                "ScheduleAdherence": lambda m: (
                    m.outcomes.tasks_completed_on_time / m.outcomes.total_tasks * 100
                    if m.outcomes.total_tasks > 0 else 0
                ),
                "CostOverruns": lambda m: m.outcomes.cost_overruns,
                "AverageSA": lambda m: np.mean([a.awareness.total_score() for a in m.schedule.agents]),
                "Worker_SA": lambda m: (
                    np.mean([a.awareness.total_score() for a in m.schedule.agents if a.role == AgentRole.WORKER])
                    if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0
                ),
                "Manager_SA": lambda m: (
                    np.mean([a.awareness.total_score() for a in m.schedule.agents if a.role == AgentRole.MANAGER])
                    if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0
                ),
                "Director_SA": lambda m: (
                    np.mean([a.awareness.total_score() for a in m.schedule.agents if a.role == AgentRole.DIRECTOR])
                    if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0
                ),
                "Reporter_SA": lambda m: (
                    np.mean([a.awareness.total_score() for a in m.schedule.agents if a.role == AgentRole.REPORTER])
                    if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0
                ),
                "Worker_Reports_Sent": lambda m: (
                    np.sum([a.reports_sent for a in m.schedule.agents if a.role == AgentRole.WORKER])
                    if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0
                ),
                "Manager_Reports_Sent": lambda m: (
                    np.sum([a.reports_sent for a in m.schedule.agents if a.role == AgentRole.MANAGER])
                    if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0
                ),
                "Director_Reports_Sent": lambda m: (
                    np.sum([a.reports_sent for a in m.schedule.agents if a.role == AgentRole.DIRECTOR])
                    if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0
                ),
                "Reporter_Reports_Sent": lambda m: (
                    np.sum([a.reports_sent for a in m.schedule.agents if a.role == AgentRole.REPORTER])
                    if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0
                ),
                "Worker_Reports_Received": lambda m: (
                    np.sum([len(a.reports_received) for a in m.schedule.agents if a.role == AgentRole.WORKER])
                    if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0
                ),
                "Manager_Reports_Received": lambda m: (
                    np.sum([len(a.reports_received) for a in m.schedule.agents if a.role == AgentRole.MANAGER])
                    if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0
                ),
                "Director_Reports_Received": lambda m: (
                    np.sum([len(a.reports_received) for a in m.schedule.agents if a.role == AgentRole.DIRECTOR])
                    if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0
                ),
                "Reporter_Reports_Received": lambda m: (
                    np.sum([len(a.reports_received) for a in m.schedule.agents if a.role == AgentRole.REPORTER])
                    if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0
                ),
                "TotalTasks": lambda m: m.outcomes.total_tasks,
                "TasksCompletedOnTime": lambda m: m.outcomes.tasks_completed_on_time,
                "BudgetRemaining": lambda m: m.budget,
                "EquipmentAvailable": lambda m: m.equipment_available,
                "Worker_Act_Count": lambda m: (
                    np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.WORKER])
                    if any(a.role == AgentRole.WORKER for a in m.schedule.agents) else 0
                ),
                "Manager_Act_Count": lambda m: (
                    np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.MANAGER])
                    if any(a.role == AgentRole.MANAGER for a in m.schedule.agents) else 0
                ),
                "Director_Act_Count": lambda m: (
                    np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.DIRECTOR])
                    if any(a.role == AgentRole.DIRECTOR for a in m.schedule.agents) else 0
                ),
                "Reporter_Act_Count": lambda m: (
                    np.sum([a.actions_taken["act"] for a in m.schedule.agents if a.role == AgentRole.REPORTER])
                    if any(a.role == AgentRole.REPORTER for a in m.schedule.agents) else 0
                )
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
        
        self.metrics_log = []
        self.agent_sa_log = []
        self.configuration_log = []
        
        if not self._initialized:
            self.log_configuration()
            self.setup_excel_logging()
            self.initialize_agents()
            self._initialized = True
            logging.debug(f"Model initialized with simulation_id: {self.simulation_id}, excel_filepath: {self.excel_filepath}")
        else:
            logging.warning(f"Model reinitialization attempted for simulation_id: {self.simulation_id}")

    def setup_excel_logging(self):
        output_dir = os.path.join(os.getcwd(), "simulation_outputs")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logging.debug(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Failed to create output directory {output_dir}: {e}")
                logging.error(f"Directory creation error: {traceback.format_exc()}")

        # Create a unique filename for this simulation run
        filename = f"construction_simulation_{self.simulation_id}.xlsx"
        self.excel_filepath = os.path.join(output_dir, filename)
        self.csv_filepath = os.path.join(output_dir, filename.replace('.xlsx', '.csv'))

        # Initialize the Excel file only if it doesn't exist
        if not os.path.exists(self.excel_filepath):
            try:
                with pd.ExcelWriter(self.excel_filepath, engine='openpyxl', mode='w') as writer:
                    pd.DataFrame(self.configuration_log).to_excel(writer, sheet_name='Configuration', index=False)
                logging.debug(f"Initialized new Excel file: {self.excel_filepath}")
            except Exception as e:
                print(f"Error initializing Excel file {self.excel_filepath}: {e}")
                logging.error(f"Excel initialization error: {traceback.format_exc()}")
                # Write configuration to CSV only if Excel fails
                if self.configuration_log:
                    pd.DataFrame(self.configuration_log).to_csv(self.csv_filepath, index=False)
                    logging.debug(f"Configuration saved to fallback CSV: {self.csv_filepath}")
        else:
            logging.warning(f"Excel file {self.excel_filepath} already exists, skipping initialization")

    def get_current_comm_failure(self):
        base_comm_failure = (
            self.comm_failure_dedicated if self.reporting_structure == ReportingStructure.DEDICATED else
            self.comm_failure_self if self.reporting_structure == ReportingStructure.SELF else
            self.comm_failure_none
        )
        if self.outcomes.safety_incidents > 5:
            base_comm_failure *= 1.3
        if self.org_structure == OrgStructure.FLAT:
            return base_comm_failure * 0.8
        elif self.org_structure == OrgStructure.HIERARCHICAL:
            return base_comm_failure * 1.2
        return base_comm_failure

    def log_configuration(self):
        config_data = {
            "Simulation_ID": [self.simulation_id],
            "Step": [0],
            "Reporting_Structure": [self.reporting_structure.value],
            "Org_Structure": [self.org_structure.value],
            "Hazard_Probability": [self.base_hazard_prob],
            "Delay_Probability": [self.base_delay_prob],
            "Resource_Shortage_Probability": [self.resource_prob],
            "Comm_Failure_Dedicated": [self.comm_failure_dedicated],
            "Comm_Failure_Self": [self.comm_failure_self],
            "Comm_Failure_None": [self.comm_failure_none],
            "Worker_Detection": [self.worker_detection],
            "Manager_Detection": [self.manager_detection],
            "Reporter_Detection": [self.reporter_detection],
            "Director_Detection": [self.director_detection],
            "Worker_Reporting": [self.worker_reporting],
            "Manager_Reporting": [self.manager_reporting],
            "Reporter_Reporting": [self.reporter_reporting],
            "Director_Reporting": [self.director_reporting],
            "Initial_Budget": [self.budget],
            "Initial_Equipment": [self.equipment_available],
            "Grid_Width": [self.width],
            "Grid_Height": [self.height],
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        self.configuration_log = config_data
        logging.debug(f"Configuration logged for simulation_id: {self.simulation_id}")

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
        worker_fatigue = np.mean([a.fatigue for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0.5
        budget_factor = max(0.1, 1 - self.budget / 1000000)
        org_factor = 1.2 if self.org_structure == OrgStructure.HIERARCHICAL else 0.8 if self.org_structure == OrgStructure.FLAT else 1.0
        incident_factor = min(1.0, 1 + 0.05 * self.outcomes.safety_incidents)
        hazard_prob = self.base_hazard_prob * (1 + 0.5 * worker_fatigue) * (1 + budget_factor) * org_factor * incident_factor
        hazard_prob = min(hazard_prob, 0.5)
        delay_prob = self.base_delay_prob * (1 + 0.02 * budget_factor)
        effective_resource_prob = self.resource_prob

        if random.random() < hazard_prob:
            severity = random.uniform(0.5, 1.0)
            self.current_events.append({"type": EventType.HAZARD, "description": "Loose scaffold", "severity": severity})
            self.event_counts[EventType.HAZARD] += 1
            delay_prob += 0.05 * severity
            effective_resource_prob += 0.03 * severity
            print(f"Step {self.schedule.steps}: Generated HAZARD event (severity={severity:.2f}, prob={hazard_prob:.3f})")

        if random.random() < delay_prob:
            severity = random.uniform(0.3, 0.7)
            self.current_events.append({"type": EventType.DELAY, "description": "Supply chain delay", "severity": severity})
            self.event_counts[EventType.DELAY] += 1
            self.outcomes.total_tasks += 1
            effective_resource_prob += 0.02 * severity
            print(f"Step {self.schedule.steps}: Generated DELAY event (severity={severity:.2f}, prob={delay_prob:.3f}), Total Tasks={self.outcomes.total_tasks}")

        if random.random() < effective_resource_prob:
            severity = random.uniform(0.2, 0.5)
            self.current_events.append({"type": EventType.RESOURCE_SHORTAGE, "description": "Material unavailability", "severity": severity})
            self.event_counts[EventType.RESOURCE_SHORTAGE] += 1
            print(f"Step {self.schedule.steps}: Generated RESOURCE_SHORTAGE event (severity={severity:.2f}, prob={effective_resource_prob:.3f})")

        if not self.current_events:
            print(f"Step {self.schedule.steps}: No events generated")
        return self.current_events if self.current_events else [None]

    def send_report(self, sender: ConstructionAgent, report: Dict) -> bool:
        self.reports.append(report)
        comm_failure = self.get_current_comm_failure()
        success = False
        target_roles = []
        event = report.get("event")
        comm_radius = 5 if self.org_structure == OrgStructure.FLAT else 3 if self.org_structure == OrgStructure.FUNCTIONAL else 2

        if self.reporting_structure == ReportingStructure.DEDICATED:
            if sender.role == AgentRole.REPORTER:
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
                        print(f"Step {self.schedule.steps}: Reporter {sender.unique_id} broadcast report to Agent {agent.unique_id} ({agent.role.value})")
            else:
                neighbors = self.grid.get_neighbors(sender.pos, moore=True, include_center=False, radius=comm_radius)
                for agent in self.schedule.agents:
                    if agent in neighbors and agent.role == AgentRole.REPORTER and random.random() > comm_failure:
                        report["additional_aspects"] = {"severity_increase": random.uniform(0.1, 0.3)}
                        agent.reports_received.append(report)
                        agent.awareness.comprehension = min(agent.awareness.comprehension + self.reporter_detection * 15, 100)
                        success = True
                        print(f"Step {self.schedule.steps}: Agent {sender.unique_id} ({sender.role.value}) sent report with additional aspects to Reporter {agent.unique_id}")
        elif self.reporting_structure == ReportingStructure.SELF:
            if self.org_structure == OrgStructure.HIERARCHICAL:
                target_roles = [AgentRole.MANAGER if sender.role == AgentRole.WORKER else AgentRole.DIRECTOR]
            elif self.org_structure == OrgStructure.FLAT:
                target_roles = [AgentRole.WORKER, AgentRole.MANAGER, AgentRole.DIRECTOR]
            else:
                target_roles = [AgentRole.WORKER, AgentRole.MANAGER, AgentRole.DIRECTOR]
            neighbors = self.grid.get_neighbors(sender.pos, moore=True, include_center=False, radius=comm_radius)
            for agent in self.schedule.agents:
                if agent in neighbors and agent.role in target_roles and random.random() > comm_failure:
                    agent.reports_received.append(report)
                    success = True
                    if event and not report.get("acted_on", False):
                        follow_up_action = agent.decide_action(event, is_follow_up=True)
                        agent.execute_action(event, follow_up_action)
                        report["acted_on"] = True
                        print(f"Step {self.schedule.steps}: Agent {agent.unique_id} ({agent.role.value}) executed follow-up action {follow_up_action} for reported event {event['type'].value}")
                    print(f"Step {self.schedule.steps}: Agent {sender.unique_id} ({sender.role.value}) sent report to Agent {agent.unique_id} ({agent.role.value})")
        elif self.reporting_structure == ReportingStructure.NONE:
            neighbors = self.grid.get_neighbors(sender.pos, moore=True, include_center=False, radius=comm_radius)
            for agent in self.schedule.agents:
                if agent in neighbors and agent.role in [AgentRole.WORKER, AgentRole.MANAGER, AgentRole.DIRECTOR] and random.random() > comm_failure:
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
        schedule_adherence = (self.outcomes.tasks_completed_on_time / self.outcomes.total_tasks * 100
                             if self.outcomes.total_tasks > 0 else 0)
        if self.schedule.steps > 1 and abs(schedule_adherence - self.previous_adherence) > 0.01:
            print(f"Step {self.schedule.steps}: Schedule Adherence changed from {self.previous_adherence:.2f}% to {schedule_adherence:.2f}%")
        self.previous_adherence = schedule_adherence
        metrics_data = {
            "Simulation_ID": [self.simulation_id],
            "Step": [self.schedule.steps],
            "Reporting_Structure": [self.reporting_structure.value],
            "Org_Structure": [self.org_structure.value],
            "Worker_SA": [np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0],
            "Manager_SA": [np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0],
            "Director_SA": [np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0],
            "Reporter_SA": [np.mean([a.awareness.total_score() for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0],
            "Worker_Reports_Sent": [np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0],
            "Manager_Reports_Sent": [np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0],
            "Director_Reports_Sent": [np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0],
            "Reporter_Reports_Sent": [np.sum([a.reports_sent for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0],
            "Worker_Reports_Received": [np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0],
            "Manager_Reports_Received": [np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0],
            "Director_Reports_Received": [np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0],
            "Reporter_Reports_Received": [np.sum([len(a.reports_received) for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0],
            "Safety_Incidents": [self.outcomes.safety_incidents],
            "Incident_Points": [self.outcomes.incident_points],
            "Schedule_Adherence": [schedule_adherence],
            "Cost_Overruns": [self.outcomes.cost_overruns],
            "Comm_Failure_Rate": [comm_failure],
            "Hazard_Events": [self.event_counts[EventType.HAZARD]],
            "Delay_Events": [self.event_counts[EventType.DELAY]],
            "Resource_Shortage_Events": [self.event_counts[EventType.RESOURCE_SHORTAGE]],
            "Total_Tasks": [self.outcomes.total_tasks],
            "Tasks_Completed_On_Time": [self.outcomes.tasks_completed_on_time],
            "Budget_Remaining": [self.budget],
            "Equipment_Available": [self.equipment_available],
            "Worker_Act_Count": [np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.WORKER]) if any(a.role == AgentRole.WORKER for a in self.schedule.agents) else 0],
            "Manager_Act_Count": [np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.MANAGER]) if any(a.role == AgentRole.MANAGER for a in self.schedule.agents) else 0],
            "Director_Act_Count": [np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.DIRECTOR]) if any(a.role == AgentRole.DIRECTOR for a in self.schedule.agents) else 0],
            "Reporter_Act_Count": [np.sum([a.actions_taken["act"] for a in self.schedule.agents if a.role == AgentRole.REPORTER]) if any(a.role == AgentRole.REPORTER for a in self.schedule.agents) else 0]
        }
        self.metrics_log.append(metrics_data)
        print(f"Step {self.schedule.steps}: Schedule Adherence={schedule_adherence:.2f}%, Total Tasks={self.outcomes.total_tasks}, Completed On Time={self.outcomes.tasks_completed_on_time}, "
              f"Safety Incidents={self.outcomes.safety_incidents}, Worker Act={metrics_data['Worker_Act_Count'][0]}, Budget={self.budget:.2f}, Equipment={self.equipment_available}")

    def log_agent_situational_awareness(self):
        try:
            sa_data = []
            for agent in self.schedule.agents:
                sa_data.append({
                    "Simulation_ID": self.simulation_id,
                    "Step": self.schedule.steps,
                    "Agent_ID": agent.unique_id,
                    "Role": agent.role.value,
                    "SA_Score": agent.awareness.total_score(),
                    "Perception": agent.awareness.perception,
                    "Comprehension": agent.awareness.comprehension,
                    "Projection": agent.awareness.projection,
                    "Reports_Sent": agent.reports_sent,
                    "Reports_Received": len(agent.reports_received),
                    "Workload": agent.workload,
                    "Fatigue": agent.fatigue,
                    "Experience": agent.experience,
                    "Risk_Tolerance": agent.risk_tolerance
                })
            self.agent_sa_log.extend(sa_data)
            print(f"Step {self.schedule.steps}: Logged situational awareness for {len(sa_data)} agents")
        except Exception as e:
            print(f"Error logging agent situational awareness at step {self.schedule.steps}: {e}")
            logging.error(f"Agent SA logging error: {traceback.format_exc()}")

    def save_to_excel(self):
        try:
            metrics_df = pd.DataFrame(self.metrics_log) if self.metrics_log else pd.DataFrame()
            sa_df = pd.DataFrame(self.agent_sa_log) if self.agent_sa_log else pd.DataFrame()
            model_data = self.datacollector.get_model_vars_dataframe()
            agent_data = self.datacollector.get_agent_vars_dataframe()

            existing_metrics = pd.DataFrame()
            existing_sa = pd.DataFrame()
            existing_model_data = pd.DataFrame()
            existing_agent_data = pd.DataFrame()
            
            if os.path.exists(self.excel_filepath):
                try:
                    excel_file = pd.ExcelFile(self.excel_filepath, engine='openpyxl')
                    if 'Model_Metrics' in excel_file.sheet_names:
                        existing_metrics = pd.read_excel(self.excel_filepath, sheet_name='Model_Metrics', engine='openpyxl')
                    if 'Agent_SA' in excel_file.sheet_names:
                        existing_sa = pd.read_excel(self.excel_filepath, sheet_name='Agent_SA', engine='openpyxl')
                    if 'Mesa_Model_Data' in excel_file.sheet_names:
                        existing_model_data = pd.read_excel(self.excel_filepath, sheet_name='Mesa_Model_Data', engine='openpyxl')
                    if 'Mesa_Agent_Data' in excel_file.sheet_names:
                        existing_agent_data = pd.read_excel(self.excel_filepath, sheet_name='Mesa_Agent_Data', engine='openpyxl')
                except Exception as e:
                    print(f"Error reading existing Excel file {self.excel_filepath}: {e}")
                    logging.error(f"Excel read error: {traceback.format_exc()}")

            if not metrics_df.empty:
                metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True) if not existing_metrics.empty else metrics_df
            if not sa_df.empty:
                sa_df = pd.concat([existing_sa, sa_df], ignore_index=True) if not existing_sa.empty else sa_df
            if not model_data.empty:
                model_data = pd.concat([existing_model_data, model_data], ignore_index=True) if not existing_model_data.empty else model_data
            if not agent_data.empty:
                agent_data = pd.concat([existing_agent_data, agent_data], ignore_index=True) if not existing_agent_data.empty else agent_data

            with pd.ExcelWriter(self.excel_filepath, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                if not metrics_df.empty:
                    metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
                if not sa_df.empty:
                    sa_df.to_excel(writer, sheet_name='Agent_SA', index=False)
                if not model_data.empty:
                    model_data.to_excel(writer, sheet_name='Mesa_Model_Data', index=False)
                if not agent_data.empty:
                    agent_data.to_excel(writer, sheet_name='Mesa_Agent_Data', index=False)
            logging.debug(f"Data appended to {self.excel_filepath} at step {self.schedule.steps}")
            print(f"Data successfully appended to {self.excel_filepath} at step {self.schedule.steps}")

            self.metrics_log = []
            self.agent_sa_log = []

        except Exception as e:
            print(f"Error saving to Excel at step {self.schedule.steps}: {e}")
            logging.error(f"Excel save error: {traceback.format_exc()}")
            try:
                if not metrics_df.empty:
                    metrics_df.to_csv(self.csv_filepath.replace('.csv', '_metrics.csv'), mode='a', header=not os.path.exists(self.csv_filepath.replace('.csv', '_metrics.csv')), index=False)
                if not sa_df.empty:
                    sa_df.to_csv(self.csv_filepath.replace('.csv', '_agent_sa.csv'), mode='a', header=not os.path.exists(self.csv_filepath.replace('.csv', '_agent_sa.csv')), index=False)
                if not model_data.empty:
                    model_data.to_csv(self.csv_filepath.replace('.csv', '_model_data.csv'), mode='a', header=not os.path.exists(self.csv_filepath.replace('.csv', '_model_data.csv')), index=False)
                if not agent_data.empty:
                    agent_data.to_csv(self.csv_filepath.replace('.csv', '_agent_data.csv'), mode='a', header=not os.path.exists(self.csv_filepath.replace('.csv', '_agent_data.csv')), index=False)
                logging.debug(f"Data saved to CSV files at {self.csv_filepath.replace('.csv', '_*.csv')}")
                print(f"Data saved to CSV files at {self.csv_filepath.replace('.csv', '_*.csv')}")
            except Exception as e:
                print(f"Error saving to CSV at step {self.schedule.steps}: {e}")
                logging.error(f"CSV save error: {traceback.format_exc()}")

    def step(self):
        try:
            print(f"Executing step {self.schedule.steps + 1}")
            if self.schedule.steps % 5 == 0:
                if self.budget < 100000:
                    self.budget += 150000
                    print(f"Step {self.schedule.steps}: Replenished budget, new total={self.budget:.2f}")
                if self.equipment_available < 50:
                    self.equipment_available += 10
                    print(f"Step {self.schedule.steps}: Replenished equipment, new total={self.equipment_available}")
            self.schedule.step()
            self.datacollector.collect(self)
            self.log_metrics()
            if self.schedule.steps % 10 == 0:
                try:
                    self.log_agent_situational_awareness()
                except Exception as e:
                    print(f"Error in log_agent_situational_awareness at step {self.schedule.steps}: {e}")
                    logging.error(f"SA logging error: {traceback.format_exc()}")
            if self.schedule.steps % 5 == 0:
                self.save_to_excel()
            print(f"Completed step {self.schedule.steps}")
        except Exception as e:
            print(f"Error in model step {self.schedule.steps + 1}: {e}")
            logging.error(f"Step error: {traceback.format_exc()}")
            self.save_to_excel()

    def run_simulation(self, steps: int = 150):
        print(f"Starting simulation with {steps} steps...")
        print(f"Simulation ID: {self.simulation_id}")
        print(f"Reporting structure: {self.reporting_structure.value}, Org structure: {self.org_structure.value}")
        print(f"Excel output: {self.excel_filepath}")
        print(f"Agent counts: Workers={len([a for a in self.schedule.agents if a.role == AgentRole.WORKER])}, "
              f"Managers={len([a for a in self.schedule.agents if a.role == AgentRole.MANAGER])}, "
              f"Directors={len([a for a in self.schedule.agents if a.role == AgentRole.DIRECTOR])}, "
              f"Reporters={len([a for a in self.schedule.agents if a.role == AgentRole.REPORTER])}")
        
        for i in range(steps):
            try:
                self.step()
                if (i + 1) % 25 == 0:
                    print(f"Completed step {i + 1}/{steps}")
            except Exception as e:
                print(f"Simulation halted at step {i + 1}: {e}")
                self.save_to_excel()
                break
        
        self.save_to_excel()
        print(f"Simulation completed. Results saved to {self.excel_filepath}")
        self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        print(f"Simulation ID: {self.simulation_id}")
        print(f"Total Steps: {self.schedule.steps}")
        print(f"Reporting Structure: {self.reporting_structure.value}")
        print(f"Organizational Structure: {self.org_structure.value}")
        print(f"Communication Failure Rate: {self.get_current_comm_failure():.3f}")
        print(f"Budget Remaining: ${self.budget:,.2f}")
        print(f"Equipment Available: {self.equipment_available}")
        print("\nSafety Outcomes:")
        print(f"  Safety Incidents: {self.outcomes.safety_incidents}")
        print(f"  Incident Points: {self.outcomes.incident_points:.1f}")
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
