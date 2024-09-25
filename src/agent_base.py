from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def gm(self):
        ...
        
    def noise(self):
        ...
        
    def step(self):
        ...
        
    def infer(self):
        ...

class Environment(ABC):
    @abstractmethod
    def ge(self):
        ...
        
    @abstractmethod
    def noise(self):
        ...
    
    @abstractmethod
    def step(self):
        ...