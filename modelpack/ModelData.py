from .SliceData import SliceData
from pyomo import environ as pyo

class ModelData:
    def __init__(
            self,
            B: float,
            R: int,
            PS: int,
            e: float,
            l_max: int,
            b_max: int,
            w_max: int
        ) -> None:
        self.B = B
        self.R = R
        self.PS = PS
        self.e = e
        self.l_max = l_max
        self.b_max = b_max
        self.w_max = w_max
        self.slices = dict()
        self.w = 1
        self.n = 0
    
    def addSlice(self, s: SliceData):
        self.slices[s.id] = s
    
    def advanceStep(self, model: pyo.ConcreteModel):
        for s in model.S:
            # d = slices[s]
            total_sent = sum(model.sent_s_i[i].value for i in model.I) if (s in model.S_rlp) else 0
            b = sum(self.slices[s].buffer[i] for i in model.I) - total_sent
            request = self.slices[s].buffer[0] * self.PS
            r = sum(model.k_u[u] * self.PS for u in self.slices[s].users.keys())
            
            self.slices[s].saveHist(d,b,request,r,total_sent)


        for s in self.slices.values():
            s.saveHist(0,0,0,0,0) # TODO GET d, b AND request
        self.n += 1
        if w < self.w_max:
            w += 1