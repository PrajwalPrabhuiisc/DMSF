Construction Site Safety Simulation Documentation
Overview
This document provides comprehensive documentation for the Construction Site Safety Simulation, an Agent-Based Model (ABM) implemented using the Mesa framework in Python. The simulation models a construction site environment to analyze how different reporting structures and situational awareness affect safety outcomes, schedule adherence, and cost overruns. It incorporates machine learning for decision-making, tracks agent behaviors, and logs results to Excel for analysis.
Table of Contents

Purpose and Scope
System Requirements
Code Structure
Key Components
Enums
Data Classes
Decision Model
ConstructionAgent Class
ConstructionModel Class
Visualization


Simulation Workflow
Usage Instructions
Output and Logging
Extending the Model
Limitations and Assumptions
Troubleshooting


Purpose and Scope
The Construction Site Safety Simulation is designed to:

Simulate interactions among construction site agents (Workers, Managers, Directors, and Reporters) under different reporting structures (Dedicated, Self, None).
Model situational awareness (perception, comprehension, projection) and its impact on decision-making.
Use a logistic regression model to simulate agent decisions (ignore, report, act, escalate) based on workload, fatigue, and event severity.
Track safety incidents, schedule adherence, and cost overruns.
Visualize agent behaviors and metrics using a grid-based interface and charts.
Log detailed simulation data to Excel for post-analysis.

The simulation is intended for researchers, safety analysts, and project managers to study the effects of organizational structures on construction site outcomes.

System Requirements

Python Version: 3.7 or higher
Required Libraries:
mesa (Agent-Based Modeling framework)
numpy
pandas
scikit-learn (for LogisticRegression)
openpyxl (for Excel output)
random, enum, dataclasses, typing, datetime, os, traceback, logging


Optional for Visualization:
A web browser to view the Mesa visualization server (runs on localhost:8521).


Operating System: Compatible with Windows, macOS, or Linux.
Disk Space: Minimal, but ensure sufficient space for Excel output files (typically <10 MB per simulation).


Code Structure
The codebase is organized into several logical components:

Imports: Standard Python libraries and Mesa-specific modules for modeling and visualization.
Enums: Define ReportingStructure, AgentRole, and EventType for consistent categorization.
Data Classes: SituationalAwareness and ProjectOutcomes for structured data management.
Decision Model: A DecisionModel class using logistic regression for agent decision-making.
Agent Class: ConstructionAgent defines agent behaviors, roles, and decision logic.
Model Class: ConstructionModel manages the simulation environment, agents, events, and data collection.
Visualization: Functions for grid-based visualization and charts, with a modular server setup.
Main Execution: Initializes and launches the simulation server.


Key Components
Enums

ReportingStructure:
DEDICATED: Reports are sent to dedicated Reporter agents.
SELF: Workers report to Managers, Managers to Directors.
NONE: Reports are sent randomly to any agent.


AgentRole:
WORKER: Performs tasks, has lower detection and reporting probabilities.
MANAGER: Oversees workers, has higher detection and reporting probabilities.
DIRECTOR: Senior role, highest detection and reporting probabilities.
REPORTER: Specialized role in Dedicated structure for handling reports.


EventType:
HAZARD: Safety-related events (e.g., loose scaffold).
DELAY: Schedule-related issues (e.g., supply chain delays).
RESOURCE_SHORTAGE: Resource-related issues (e.g., material unavailability).



Data Classes

SituationalAwareness:
Attributes: perception, comprehension, projection (floats, 0-100).
Method: total_score() computes the average of the three components.


ProjectOutcomes:
Tracks safety_incidents, incident_points, tasks_completed_on_time, total_tasks, and cost_overruns.



Decision Model

Class: DecisionModel
Purpose: Uses logistic regression to predict agent actions (ignore, report, act, escalate) based on:
workload (1-5)
fatigue (0-1)
event_severity (0.5 for resource shortage, 0.7 for delay, 1.0 for hazard)


Training Data: Hard-coded dataset with 9 samples for simplicity.
Output: Probabilistic action selection using predict_proba.

ConstructionAgent Class

Attributes:
unique_id: Unique identifier.
role: From AgentRole enum.
pos: Grid position (tuple).
awareness: SituationalAwareness instance.
workload: Integer (1-5).
fatigue: Float (0-1).
reports_sent: Count of sent reports.
reports_received: List of received reports.
decision_model: Instance of DecisionModel.
actions_taken: Dictionary tracking actions (ignore, report, act, escalate).
detection_accuracy and reporting_chance: Role-specific probabilities.


Methods:
observe_event: Updates awareness based on event detection (affected by fatigue and workload).
decide_action: Uses DecisionModel to choose an action.
step: Observes events, decides actions, updates metrics, and moves on the grid.



ConstructionModel Class

Attributes:
width, height: Grid dimensions (default 20x20).
grid: Mesa MultiGrid for agent placement.
schedule: Mesa RandomActivation for agent stepping.
reporting_structure: From ReportingStructure enum.
outcomes: ProjectOutcomes instance.
reports: List of reports sent.
current_events and event_counts: Track events per step.
hazard_prob, delay_prob, resource_prob: Event probabilities.
comm_failure_dedicated, comm_failure_self, comm_failure_none: Communication failure rates.
worker_detection, manager_detection, reporter_detection: Role-specific detection probabilities.
worker_reporting, manager_reporting, reporter_reporting: Role-specific reporting probabilities.
metrics_log, agent_sa_log, configuration_log: Data for Excel output.
datacollector: Mesa DataCollector for model and agent metrics.


Methods:
setup_excel_logging: Configures Excel output file path.
get_current_comm_failure: Returns communication failure rate based on reporting structure.
log_configuration: Logs simulation parameters.
initialize_agents: Places agents on the grid based on reporting structure.
get_events: Generates random events per step.
send_report: Handles report transmission with potential communication failures.
log_metrics: Logs model-level metrics per step.
log_agent_situational_awareness: Logs agent-level data every 10 steps.
save_to_excel: Saves logs to Excel every 50 steps and at simulation end.
step: Executes one simulation step.
run_simulation: Runs the simulation for a specified number of steps.
print_summary: Outputs a summary of results.



Visualization

agent_portrayal: Defines visual representation of agents:
Colors: Blue (Workers), Green (Managers), Red (Directors), Orange (Reporters).
Size: Proportional to situational awareness score.


ModelLegend: Displays a legend for agent roles and sizes (console-based if TextElement is unavailable).
create_server: Sets up a Mesa ModularServer with:
A 20x20 grid (CanvasGrid).
Charts for situational awareness and safety/schedule metrics.
User-settable parameters for reporting structure, event probabilities, and communication failure rates.




Simulation Workflow

Initialization:
Create a ConstructionModel instance with specified parameters.
Initialize agents based on the reporting structure (50 Workers, 10 Managers, 3 Directors, 5 Reporters for Dedicated).
Set up Excel logging and data collection.


Step Execution:
Generate random events (HAZARD, DELAY, RESOURCE_SHORTAGE).
Agents observe events, update awareness, decide actions, and move.
Reports are sent based on the reporting structure, with possible communication failures.
Metrics are logged, and data is saved to Excel periodically.


Visualization:
Agents are displayed on a grid, with charts showing situational awareness and safety metrics.


Output:
Excel file with sheets for configuration, model metrics, agent situational awareness, and Mesa data.
Console summary of safety, schedule, and reporting outcomes.




Usage Instructions

Install Dependencies:pip install mesa numpy pandas scikit-learn openpyxl


Run the Simulation:
Save the code to a file (e.g., construction_abm.py).
Run the script:python construction_abm.py


Access the visualization server at http://localhost:8521.


Adjust Parameters:
Modify model_params in create_server to change default values for:
reporting_structure (dedicated, self, none)
hazard_prob (0.01-0.20)
delay_prob (0.01-0.30)
comm_failure_dedicated (0.01-0.50)
comm_failure_self (0.01-0.60)
comm_failure_none (0.10-0.90)


Use the web interface sliders to adjust parameters dynamically.


Run Without Visualization:
Comment out the server.launch() line and add:model = ConstructionModel()
model.run_simulation(steps=100)






Output and Logging

Excel Output:
Saved to simulation_outputs/construction_ABM_*.xlsx.
Sheets:
Configuration: Simulation parameters (e.g., reporting structure, probabilities).
Model_Metrics: Step-wise metrics (e.g., safety incidents, situational awareness).
Agent_SA: Agent-level data every 10 steps (e.g., awareness, workload, reports).
Mesa_Model_Data: Model-level data from DataCollector.
Mesa_Agent_Data: Agent-level data from DataCollector.




Console Output:
Simulation progress (step updates).
Final summary with safety, schedule, and reporting statistics.




Extending the Model

Add New Agent Roles:
Extend AgentRole enum and update initialize_agents to include new roles.
Adjust detection and reporting probabilities in ConstructionModel.


New Event Types:
Add to EventType enum and update get_events with new probabilities and effects.


Enhanced Decision Model:
Replace DecisionModel with a more complex model (e.g., neural network) or train on real-world data.


Custom Visualizations:
Add new charts to create_server for additional metrics (e.g., workload, fatigue).


Parameter Tuning:
Experiment with different agent counts, grid sizes, or event probabilities.




Limitations and Assumptions

Simplified Decision Model: Logistic regression uses a small, hard-coded dataset, which may not capture complex real-world behaviors.
Static Agent Counts: Fixed numbers of agents per role; real sites may have dynamic staffing.
Random Events: Event probabilities are static and do not model external factors (e.g., weather, supplier reliability).
No Agent Communication Network: Agents communicate based on proximity and role, not a detailed organizational hierarchy.
Simplified Metrics: Safety incidents and cost overruns use basic calculations (e.g., $25,000 per hazard).


Troubleshooting

Issue: "TextElement not available" warning.
Solution: Update Mesa to the latest version (pip install --upgrade mesa).


Issue: Excel file not saving.
Solution: Check write permissions in the simulation_outputs directory and ensure openpyxl is installed.


Issue: Visualization server not loading.
Solution: Ensure port 8521 is free and try a different browser or localhost address.


Issue: Simulation crashes during step.
Solution: Check console for traceback; common issues include division by zero (e.g., no tasks recorded) or invalid parameter values.




This documentation provides a comprehensive guide to understanding, running, and extending the Construction Site Safety Simulation. For further assistance, refer to the Mesa documentation or contact the development team.
