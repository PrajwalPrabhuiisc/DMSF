import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from enums import AgentRole
from construction_model import ConstructionModel

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
            value="centralized",
            choices=["centralized", "decentralized", "hybrid"]
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
            0.15,
            0.01,
            0.30,
            0.01
        ),
        "resource_prob": UserSettableParameter(
            "slider",
            "Resource Probability",
            0.03,
            0.01,
            0.20,
            0.01
        ),
        "num_workers": UserSettableParameter(
            "slider",
            "Number of Workers",
            50,
            20,
            100,
            5
        ),
        "num_managers": UserSettableParameter(
            "slider",
            "Number of Managers",
            10,
            5,
            20,
            1
        ),
        "num_directors": UserSettableParameter(
            "slider",
            "Number of Directors",
            3,
            1,
            5,
            1
        ),
        "num_reporters": UserSettableParameter(
            "slider",
            "Number of Reporters",
            5,
            0,
            10,
            1
        )
    }
    server = ModularServer(
        ConstructionModel,
        [grid, chart_sa, chart_incidents],
        "Construction Site Safety Simulation",
        model_params
    )
    return server