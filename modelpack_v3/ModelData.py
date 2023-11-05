from .SliceData import SliceData
from .UserData import UserData
from pyomo import environ as pyo
import numpy as np

class ModelData:
    def __init__(
            self,
            B: float, # Bandwidth (Hz)
            R: int, # Number of RBGs available on the cell (RBGs)
            PS: int, # Packet size (bits)
            e: float, # Small constant for approximations (e.g. 1e-5)
            b_max: int, # User's buffer packet capacity (packets)
            l_max: int, # Maximum latency to drop a packet (TTIs, steps or ms)
            w_max: int # Maximum window size for calculating aggregated metrics (TTIs, steps or ms)
        ) -> None:
        self.B = B
        self.R = R
        self.PS = PS
        self.e = e
        self.b_max = b_max
        self.l_max = l_max
        self.w_max = w_max
        
        # Initializing dictionaries for saving slices and users
        self.slices = dict()
        self.users = dict()

        # Initializing variables for saving results
        self.scheduling = dict()

        # Initializing the step number
        self.n = 0
    
    def addSlice(
        self,
        id, # The slice name
        ):
        self.slices[id] = SliceData(
            id=id,
            l_max=self.l_max,
        )
        self.scheduling[id] = np.array([])
    
    def addUser(
        self,
        id, # User id
        s, # User's slice name
        SE, # User's spectral efficiency list (one value for each step)
        ):
        self.users[id] = UserData(
            id=id,
            s=s,
            SE=SE,
            w_max=self.w_max,
            b_max=self.b_max,
            l_max=self.l_max
        )
    
    def associateUsersToSlices(self):
        for s in self.slices.values():
            s.disassociateUsers()
        for u in self.users.values():
            self.slices[u.s].addUser(u)

    def advanceStep(self):
        # Incrementing step and windows
        self.n += 1
        for u in self.users.values():
            u.incrementWindow()
    
    def saveResults(
        self,
        rrbs_per_slice: dict,
        ):
        for s in rrbs_per_slice.keys():
            self.scheduling[s] = np.append(self.scheduling[s], rrbs_per_slice[s])