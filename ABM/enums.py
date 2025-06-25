from enum import Enum

class ReportingStructure(Enum):
    DEDICATED = "dedicated"
    SELF = "self"
    NONE = "none"

class AgentRole(Enum):
    WORKER = "worker"
    MANAGER = "manager"
    DIRECTOR = "director"
    REPORTER = "reporter"

class EventType(Enum):
    HAZARD = "hazard"
    DELAY = "delay"
    RESOURCE_SHORTAGE = "resource_shortage"