# Construction Site Safety Simulation Research Documentation

## Abstract

This research investigates how different reporting structures impact Situational Awareness (SA) in a simulated high-rise construction site environment. Using the MESA framework, the simulation models agents (workers, managers, directors, reporters) responding to events (hazards, delays, resource shortages) under varying organizational and reporting configurations. 

The primary research question examines how reporting structures (dedicated, self, none) affect SA, measured through perception, comprehension, and projection components. The simulation incorporates non-deterministic elements through stochastic event generation, agent decisions, and communication failures. Agent decision-making combines logistic regression with Q-learning, while key performance metrics track safety incidents, schedule adherence, and cost overruns.

## 1. Introduction

Construction sites, particularly high-rise building projects, represent dynamic environments where effective Situational Awareness is critical for safety, schedule adherence, and resource management. Situational Awareness comprises three components:

- **Perception**: Detecting events and environmental changes
- **Comprehension**: Understanding the implications of detected events  
- **Projection**: Anticipating future states and potential outcomes

Ineffective reporting structures can lead to missed hazards, project delays, and resource shortages. This study simulates a high-rise construction site to evaluate how different reporting structures influence SA and project outcomes, with particular focus on the non-deterministic behavior arising from random event generation, agent decisions, and communication failures.

## 2. Research Questions and Hypotheses

### 2.1 Research Questions

**Primary Question**: How do different reporting structures (dedicated, self, none) impact the Situational Awareness of agents in a high-rise construction site simulation?

**Secondary Questions**:
- How do organizational structures (functional, flat, hierarchical) interact with reporting structures to affect SA?
- How does SA correlate with safety incidents, schedule adherence, and cost overruns?
- How do agent roles (worker, manager, director, reporter) influence SA under different configurations?

### 2.2 Hypotheses

- **H1**: Dedicated reporting structures will yield higher SA scores due to specialized reporters enhancing information flow
- **H2**: Flat organizational structures will improve SA by reducing communication barriers compared to hierarchical structures
- **H3**: Higher SA scores will correlate with fewer safety incidents, better schedule adherence, and lower cost overruns
- **H4**: Reporters will exhibit the highest SA in dedicated structures due to their specialized detection and reporting roles

## 3. Simulation Scenario: High-Rise Construction Site

### 3.1 Scenario Description

The simulation models a 30-story commercial tower construction project in an urban setting. Multiple teams work on tasks including scaffolding, concrete pouring, and material handling. Key challenges include:

**Hazards**: Loose scaffolding, falling objects, structural failures
- Risk of injuries or fatalities if unaddressed
- Modeled with severity range 0.5-1.0

**Delays**: Supply chain disruptions, late material deliveries, labor shortages  
- Impact on construction milestones
- Modeled with severity range 0.3-0.7

**Resource Shortages**: Limited availability of materials (cement) or equipment (cranes)
- Constraint on project progress
- Modeled with severity range 0.2-0.5

The workforce operates under constraints of a $1,000,000 initial budget and 500 units of equipment, with tasks requiring completion within deadlines to avoid cost overruns.

### 3.2 Model Abstraction

**Agents**: Represent construction site personnel with specific attributes:
- **Workers**: Laborers, equipment operators (experience: 0-0.5)
- **Managers**: Site supervisors (experience: 0.5-1.0)  
- **Directors**: Project managers (experience: 0.5-1.0)
- **Reporters**: Safety inspectors (experience: 0.5-1.0)

Each agent has attributes:
- Workload (1-3): Current task load
- Fatigue (0-1.0): Physical/mental strain
- Experience (0-1.0): Skill level
- Risk tolerance (0-1.0): Willingness to ignore risks

**Environment**: 20×20 grid representing the construction site with agent movement simulating task distribution across different zones or floors.

**Reporting Structures**:
- **Dedicated**: Safety inspectors report hazards to all non-reporters
- **Self**: Workers report to supervisors or managers  
- **None**: Ad-hoc reporting to nearby agents

**Organizational Structures**:
- **Functional**: Role-based communication following traditional hierarchies
- **Flat**: Open communication enabling collaborative teams
- **Hierarchical**: Strict chain of command for large bureaucratic projects

### 3.3 Simplifications and Assumptions

- **Event Modeling**: Discrete events with fixed severity ranges rather than continuous, multifaceted real-world issues
- **Agent Behavior**: Decision-making based on logistic regression and Q-learning, abstracting complex human factors
- **Resource Constraints**: Budget and equipment as single scalars rather than specific resource types
- **Communication**: Probabilistic modeling with failure rates, abstracting interpretation delays and misunderstandings
- **Movement**: Random grid movement simplifying complex site mobility patterns

## 4. Methodology

### 4.1 Simulation Framework

Built using the MESA framework with a multi-agent system on a 20×20 grid. Agents respond to events using logistic regression-based decisions and Q-learning, with outcomes logged to Excel/CSV files.

### 4.2 Key Components

**Agents** (`construction_agent.py`): Four role types with SituationalAwareness objects tracking perception, comprehension, and projection.

**Events** (`construction_model.py`): Stochastic generation of hazards, delays, and resource shortages with probabilities modified by fatigue, budget, and organizational factors.

**Decision Model** (`decision_model.py`): Logistic regression predicting action probabilities (ignore, report, act, escalate) based on 9 input features.

**Visualization** (`visualization.py`): Grid display with role-specific colors, SA-based sizing, and real-time charts for SA and incidents.

### 4.3 Non-Deterministic Elements

The simulation incorporates multiple sources of randomness:

**Event Generation**: Stochastic probabilities for different event types based on current conditions

**Event Detection**: Probabilistic observation based on agent experience and role

**Action Selection**: Probabilistic choice among available actions weighted by model predictions

**Action Outcomes**: Success rates dependent on agent attributes and environmental factors

**Agent Movement**: Random neighboring cell selection each timestep

**Initialization**: Random starting values for all agent attributes

## 5. Mathematical Formulations

### 5.1 Situational Awareness Calculation

Total SA score:
```
SA_total = (perception + comprehension + projection) / 3
```

**Perception Update**:
```
perception = min(perception + detection_accuracy × k₁ × org_modifier × sa_reduction, 100)
```
where k₁ = 50 for reporters, 40×(1 + experience) for others

**Comprehension Update**:
```
comprehension = min(comprehension + detection_accuracy × k₂ × (1 - workload/5) × org_modifier × sa_reduction, 100)
```
where k₂ = 30 for reporters, 20 for others

**Projection Update**:
```
projection = min(projection + detection_accuracy × k₃ × (1 - fatigue) × org_modifier × sa_reduction, 100)
```
where k₃ = 15 for reporters, 10 for others

**Organizational Modifiers**:
```
org_modifier = {
  1.2  if flat
  0.8  if hierarchical  
  1.0  if functional
}
```

**SA Reduction Factor**:
```
sa_reduction = 0.7 if safety_incidents > 5, else 1.0
```

### 5.2 Event Probability Calculations

**Budget Impact Factor**:
```
budget_factor = max(0.1, 1 - budget/1,000,000)
```

**Organizational Impact Factor**:
```
org_factor = {
  1.2  if hierarchical
  0.8  if flat
  1.0  if functional
}
```

**Incident Escalation Factor**:
```
incident_factor = min(1.0, 1 + 0.05 × safety_incidents)
```

**Hazard Probability**:
```
P(hazard) = min(base_hazard_prob × (1 + 0.5 × worker_fatigue) × (1 + budget_factor) × org_factor × incident_factor, 0.5)
```

**Delay Probability**:
```
P(delay) = base_delay_prob × (1 + 0.02 × budget_factor)
```

**Resource Shortage Probability**:
```
P(resource_shortage) = base_resource_prob + 0.03 × hazard_severity + 0.02 × delay_severity
```

### 5.3 Action Success Rates

**Action Success (Hazard/Resource)**:
```
P(success) = min(0.9 + 0.2 × experience - 0.1 × fatigue - 0.05 × workload, 0.95)
```

**Delay Completion Success**:
```
P(completion) = max(0.9, 0.98 - 0.15 × fatigue - 0.05 × workload - 0.1 × (1 - experience))
```

**Incident Probability (Ignored Hazard)**:
```
P(incident) = 0.05 × (1 - experience) × severity
```

### 5.4 Communication Failure Rates

```
comm_failure = base_comm_failure × 1.3^(safety_incidents > 5) × org_multiplier
```

where:
```
org_multiplier = {
  0.8  if flat
  1.2  if hierarchical
  1.0  if functional
}
```

## 6. Agent Decision Flow

Each agent follows this process per simulation step:

### 6.1 Event Observation
- Check for events using detection probability based on experience and role
- Update SA components if event detected using formulas above

### 6.2 Action Decision
- Compute input features: workload, fatigue, event severity, experience, time pressure, resource availability, risk tolerance, stress, recent hazard history
- Apply logistic regression model to predict action probabilities
- Adjust probabilities using Q-table values and organizational memory
- Apply role-specific and situation-specific modifiers
- Select action via argmax of adjusted probabilities

### 6.3 Action Execution
- **Act**: Consume resources (cost = 1000 × severity, equipment = severity × multiplier)
- **Report/Escalate**: Send report with success probability (1 - comm_failure)
- **Ignore**: Risk hazard incident with calculated probability
- Update agent attributes (workload, fatigue, experience)
- Update Q-table using reward feedback

### 6.4 Q-Learning Update
```
Q[event_type][action] ← Q[event_type][action] + 0.2 × (reward - Q[event_type][action])
```

Reward structure:
- Successful hazard action: +3.0 × severity
- Ignored hazard leading to incident: -3.0 × severity
- Successful delay action: +2.0 × severity
- Other actions: scaled rewards based on outcome

## 7. Implementation Details

### 7.1 File Structure

**`enums.py`**: Defines ReportingStructure, OrgStructure, AgentRole, and EventType enumerations

**`data_classes.py`**: Contains SituationalAwareness and ProjectOutcomes data structures

**`decision_model.py`**: Implements DecisionModel with logistic regression using 30 training samples

**`construction_agent.py`**: Defines ConstructionAgent class with observation, decision, and execution methods

**`construction_model.py`**: Implements ConstructionModel coordinating the entire simulation

**`visualization.py`**: Configures MESA server with grid visualization and real-time charts

**`main.py`**: Launches simulation server on port 8521

### 7.2 Experimental Configuration

**Simulation Parameters**:
- Grid size: 20×20
- Simulation duration: 150 steps
- Resource replenishment: Every 50 steps
- Agent composition: 50 workers, 5-10 managers, 1-3 directors, 0-5 reporters

**Variable Parameters**:
- Hazard probability: 0.01-0.20 (default: 0.05)
- Delay probability: 0.01-0.30 (default: 0.10)  
- Resource shortage probability: 0.05
- Communication failure rates: 0.05-0.90

**Experimental Design**: Full factorial design testing all combinations of reporting structures (3) × organizational structures (3) = 9 configurations

## 8. Performance Metrics

### 8.1 Primary Metrics
- **SA Scores**: Per role and overall averages for perception, comprehension, projection
- **Safety Incidents**: Count and severity-weighted incident points
- **Schedule Adherence**: (tasks_completed_on_time / total_tasks) × 100
- **Cost Overruns**: Budget expenditure beyond initial allocation

### 8.2 Secondary Metrics
- **Communication Effectiveness**: Reports sent vs. received ratios
- **Action Distribution**: Frequency of ignore, report, act, escalate decisions
- **Agent Performance**: Individual agent SA progression over time
- **Resource Utilization**: Budget and equipment consumption patterns

## 9. Expected Results

### 9.1 Situational Awareness
- **Dedicated reporting** should produce highest SA scores due to specialized monitoring roles
- **Flat organizational structures** should enhance SA through improved communication flow
- **Hierarchical structures** may reduce SA due to communication bottlenecks and information filtering

### 9.2 Project Outcomes  
- **Higher SA scores** should correlate with fewer safety incidents and lower cost overruns
- **Better schedule adherence** expected in configurations with effective reporting and communication
- **Role-based performance** should show reporters achieving highest SA in dedicated structures

### 9.3 Interaction Effects
- **Flat + Dedicated** combination expected to optimize both communication and specialized monitoring
- **Hierarchical + Self** may create communication gaps reducing overall effectiveness
- **Budget constraints** should amplify differences between reporting structure effectiveness

## 10. Limitations and Future Work

### 10.1 Current Limitations

**Model Simplifications**:
- Discrete event modeling vs. continuous real-world complexity
- Static training data limiting decision model adaptability
- Simplified resource constraints ignoring resource type specificity
- Probabilistic communication modeling vs. nuanced human communication

**Methodological Constraints**:
- Non-deterministic elements may obscure subtle effects requiring extensive replication
- Grid-based movement abstracting complex site mobility
- Fixed simulation duration may not capture long-term adaptation effects

## References

1. MESA Framework Documentation: https://mesa.readthedocs.io/
2. Scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
3. Endsley, M. R. (1995). Toward a theory of situation awareness in dynamic systems. *Human Factors*, 37(1), 32-64.
