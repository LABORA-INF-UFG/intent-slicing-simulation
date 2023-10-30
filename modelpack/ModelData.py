from .SliceData import SliceData
from pyomo import environ as pyo

class ModelData:
    def __init__(
            self,
            B: float, # Bandwidth (Hz)
            R: int, # Number of RBGs available on the cell (RBGs)
            PS: int, # Packet size (bits)
            e: float, # Small constant for approximations (e.g. 1e-5)
            l_max: int, # Maximum latency to drop a packet (TTIs, steps or ms)
            w_max: int # Maximum window size for calculating aggregated metrics (TTIs, steps or ms)
        ) -> None:
        self.B = B
        self.R = R
        self.PS = PS
        self.e = e
        self.l_max = l_max
        self.w_max = w_max
        
        # Initializing a dictionary for saving slices
        self.slices = dict()
        
        # Initializing the real window size, incremented by 1 at each step until reaches w_max
        self.w = 1

        # Initializing the step number
        self.n = 0
    
    def addSlice(self, s: SliceData):
        self.slices[s.id] = s
    
    def optimizeAndAdvanceStep(self, model: pyo.ConcreteModel):
        # Execute the model solution
        # TODO
        
        # Saving results
        # TODO

        # Incrementing step and window
        self.n += 1
        if w < self.w_max:
            w += 1