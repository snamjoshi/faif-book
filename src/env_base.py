from abc import ABC, abstractmethod

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