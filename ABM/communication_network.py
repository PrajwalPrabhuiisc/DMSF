import networkx as nx
import random
from typing import List, Dict, Any
from enums import ReportingStructure, AgentRole

class CommunicationNetwork:
    def __init__(self, agents: List[Any], structure_type: ReportingStructure):
        self.agents = agents
        self.structure_type = structure_type
        self.graph = nx.DiGraph()
        self.initialize_network()

    def initialize_network(self):
        for agent in self.agents:
            self.graph.add_node(agent.unique_id, agent_type=agent.role.value)

        if self.structure_type == ReportingStructure.CENTRALIZED:
            managers = [a for a in self.agents if a.role == AgentRole.MANAGER]
            directors = [a for a in self.agents if a.role == AgentRole.DIRECTOR]
            workers = [a for a in self.agents if a.role == AgentRole.WORKER]
            for manager in managers:
                for worker in workers:
                    if worker in manager.subordinates:
                        self.graph.add_edge(worker.unique_id, manager.unique_id, weight=0.9)
                        self.graph.add_edge(manager.unique_id, worker.unique_id, weight=0.7)
                for director in directors:
                    self.graph.add_edge(manager.unique_id, director.unique_id, weight=0.95)
            for reporter in [a for a in self.agents if a.role == AgentRole.REPORTER]:
                for director in directors:
                    self.graph.add_edge(reporter.unique_id, director.unique_id, weight=0.95)

        elif self.structure_type == ReportingStructure.DECENTRALIZED:
            for agent1 in self.agents:
                for agent2 in self.agents:
                    if agent1 != agent2 and random.random() < 0.3:
                        weight = 0.6 if agent1.role == AgentRole.WORKER else 0.8
                        self.graph.add_edge(agent1.unique_id, agent2.unique_id, weight=weight)

        elif self.structure_type == ReportingStructure.HYBRID:
            self.initialize_centralized()
            workers = [a for a in self.agents if a.role == AgentRole.WORKER]
            for worker in workers:
                peers = [a for a in workers if a != worker]
                for peer in random.sample(peers, min(3, len(peers))):
                    self.graph.add_edge(worker.unique_id, peer.unique_id, weight=0.65)

    def initialize_centralized(self):
        managers = [a for a in self.agents if a.role == AgentRole.MANAGER]
        directors = [a for a in self.agents if a.role == AgentRole.DIRECTOR]
        workers = [a for a in self.agents if a.role == AgentRole.WORKER]
        for manager in managers:
            for worker in workers:
                if worker in manager.subordinates:
                    self.graph.add_edge(worker.unique_id, manager.unique_id, weight=0.9)
                    self.graph.add_edge(manager.unique_id, worker.unique_id, weight=0.7)
            for director in directors:
                self.graph.add_edge(manager.unique_id, director.unique_id, weight=0.95)

    def transmit_information(self, sender: Any, receiver: Any, information: Dict) -> bool:
        if self.graph.has_edge(sender.unique_id, receiver.unique_id):
            edge = self.graph[sender.unique_id][receiver.unique_id]
            transmission_efficiency = edge['weight']
            noise_factor = 0.95 if sender.role == AgentRole.REPORTER else random.uniform(0.8, 1.0)
            information_quality = information['quality'] * transmission_efficiency * noise_factor
            if random.random() < information_quality:
                receiver.receive_information(information)
                return True
        return False

    def broadcast_safety_alert(self, issuer: Any, alert: Dict):
        for agent in self.agents:
            if self.is_relevant_recipient(issuer, agent, alert):
                self.transmit_information(issuer, agent, alert)

    def is_relevant_recipient(self, issuer: Any, agent: Any, alert: Dict) -> bool:
        return agent.role != issuer.role or alert.get('severity', 1) >= 3

    def update_edges(self):
        for u, v, data in self.graph.edges(data=True):
            sender = next(a for a in self.agents if a.unique_id == u)
            receiver = next(a for a in self.agents if a.unique_id == v)
            trust_modifier = sender.trust_in_management if receiver.role in [AgentRole.MANAGER, AgentRole.DIRECTOR] else 1.0
            data['weight'] = min(1.0, data['weight'] * (0.95 + 0.05 * trust_modifier))

    def calculate_network_metrics(self) -> Dict:
        return {
            'clustering_coefficient': nx.average_clustering(self.graph),
            'average_path_length': nx.average_shortest_path_length(self.graph) if nx.is_strongly_connected(self.graph) else float('inf'),
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph)
        }