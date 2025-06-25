from enum import Enum

class ReportingStructure(Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HYBRID = "hybrid"

class AgentRole(Enum):
    WORKER = "worker"
    MANAGER = "manager"
    DIRECTOR = "director"
    REPORTER = "reporter"

class EventType(Enum):
    HAZARD = "hazard"
    DELAY = "delay"
    RESOURCE = "resource"
    QUALITY = "quality"
    ENVIRONMENTAL = "environmental"
    EQUIPMENT = "equipment"
    PERSONNEL = "personnel"

class EventSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4