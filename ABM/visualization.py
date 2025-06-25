import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from enums import AgentRole
from construction_model import ConstructionModel

try:
    from mesa.visualization.TextVisualization import TextElement
    TEXT_ELEMENT_AVAILABLE = True
except ImportError:
    TEXT_ELEMENT_AVAILABLE = False
    print("Warning: TextElement not available in this version of MESA. Legend will be printed to console instead.")

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5
    }
    
    if agent.role == AgentRole.WORKER:
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0
    elif agent.role == AgentRole.MANAGER:
        portrayal["Color"] = "green"
        portrayal["Layer"] = 1
    elif agent.role == AgentRole.DIRECTOR:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 2
    elif agent.role == AgentRole.REPORTER:
        portrayal["Color"] = "orange"
        portrayal["Layer"] = 3
    
    sa_score = agent.awareness.total_score()
    portrayal["r"] = 0.3 + (sa_score / 100) * 0.7
    
    return portrayal

class ModelLegend:
    def __init__(self):
        pass
    
    def render(self, model):
        legend_text = """
        Agent Legend:
        🔵 Blue = Workers
        🟢 Green = Managers  
        🔴 Red = Directors
        🟠 Orange = Reporters
        
        Size = Situational Awareness Level
        """
        return legend_text

def create_server():
    grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
    
    chart_sa = ChartModule([
        {"Label": "Worker_SA", "Color": "Blue"},
        {"Label": "Manager_SA", "Color": "Green"},
        {"Label": "Director_SA", "Color": "Red"},
        {"Label": "Reporter_SA", "Color": "Orange"}
    ], data_collector_name='datacollector')
    
    chart_incidents = ChartModule([
        {"Label": "SafetyIncidents", "Color": "Red"},
        {"Label": "ScheduleAdherence", "Color": "Green"}
    ], data_collector_name='datacollector')
    
    model_params = {
        "reporting_structure": UserSettableParameter(
            "choice",
            "Reporting Structure",
            value="dedicated",
            choices=["dedicated", "self", "none"]
        ),
        "hazard_prob": UserSettableParameter(
            "slider",
            "Hazard Probability",
            0.05,
            0.01,
            0.20,
            0.01
        ),
        "delay_prob": UserSettableParameter(
            "slider",
            "Delay Probability", 
            0.10,
            0.01,
            0.30,
            0.01
        ),
        "comm_failure_dedicated": UserSettableParameter(
            "slider",
            "Communication Failure (Dedicated)",
            0.05,
            0.01,
            0.50,
            0.01
        ),
        "comm_failure_self": UserSettableParameter(
            "slider",
            "Communication Failure (Self)",
            0.10,
            0.01,
            0.60,
            0.01
        ),
        "comm_failure_none": UserSettableParameter(
            "slider",
            "Communication Failure (None)",
            0.50,
            0.10,
            0.90,
            0.01
        )
    }
    
    server = ModularServer(
        ConstructionModel,
        [grid, chart_sa, chart_incidents],
        "Construction Site Safety Simulation",
        model_params
    )
    
    return server