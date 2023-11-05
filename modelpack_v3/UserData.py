import numpy as np

class UserData:
    def __init__(
            self,
            id: int, # Id of the user
            s: str, # Name of the slice that the user belongs to
            SE: list, # Spectral efficiency (bits/s/Hz) for all steps
            w_max: int, # Maximum window size for calculating aggregated metrics (TTIs, steps or ms)
            b_max: int, # User buffer capacity (packets)
            l_max: int, # Maximum latency to drop a packet (TTIs, steps or ms)
            r_req: float = None, # Minimum required served throughput for each user (Mb/s or Kb/ms)
            l_req: float = None, # Maximum required average buffer latency (ms or TTIs or steps)
            p_req: float = None, # Maximum required packet loss rate (ratio)
            f_req: float = None, # Minimum required fifth-percentile served throughput (Mb/s or Kb/s)
            g_req: float = None # Minimum required long-term served throughput (Mb/s or Kb/s)
        ) -> None:
        self.id = id
        self.s = s
        self.SE = SE
        self.w_max = w_max
        self.b_max = b_max
        self.l_max = l_max
        self.r_req = r_req
        self.l_req = l_req
        self.p_req = p_req
        self.f_req = f_req
        self.g_req = g_req

        self.hist_r = np.array([]) # Historical served throughput (list of Mb/s or Kb/ms)
        self.hist_d = np.array([]) # Historical dropped packets (list of packets)
        self.hist_rcv = np.array([]) # Historical received packets (list of packets)
        self.hist_part = np.array([]) # Historical partially sent packets (list of part of packets)
        self.hist_buff = np.ndarray((0,l_max+1)) # Historical buffer packets (list of list of packets)
        self.hist_sent = np.ndarray((0,l_max+1)) # Historical sent packets (list of list of packets)
        
        # Historical requirements (may vary during the simulation)
        self.hist_r_req = np.array([])
        self.hist_l_req = np.array([])
        self.hist_p_req = np.array([])
        self.hist_g_req = np.array([])
        self.hist_f_req = np.array([])

        # Initializing the accumulated sent packets of previous steps (list of list of packets)
        self.acc = np.zeros(l_max+1)
        self.hist_acc = np.ndarray((0,l_max+1))
        
        # Initializing the number of packets on the buffer at the beggining of each step,
        # considering packets that tried to arrive the buffer but were dropped (list of packets)
        self.hist_b = np.array([])

        # Initializing the incremental window size
        self.w = 1 

        # Historical window size (may vary during the simulation)
        self.hist_w = np.array([])
    
    # Returns a sorted list of throughputs in a window that ends in step - 1
    def getSortedThroughputWindow (self, w, n):
        return sorted(self.hist_r[n-w+1:n])

    # Increments the window size
    def incrementWindow(self):
        if self.w < self.w_max:
            self.w += 1
        # If the slice is deactived, we reset the window size to 1
        if self.s == "be" and self.g_req == 0 and self.f_req == 0:
            self.w = 1
            
    # Updates user requirements
    def updateRequirements(
        self,
        r_req: float = None,
        l_req: float = None,
        p_req: float = None,
        f_req: float = None,
        g_req: float = None
        ):
        self.r_req = r_req
        self.l_req = l_req
        self.p_req = p_req
        self.f_req = f_req
        self.g_req = g_req

    # Updates the slice data before the simulation step and model soving
    def updateHistBefStep(
        self,
        d:int,
        rcv:int,
        part:float,
        buff:np.array,
        ):
        self.hist_d = np.append(self.hist_d, d)
        self.hist_rcv = np.append(self.hist_rcv, rcv)
        self.hist_part = np.append(self.hist_part, part)
        self.hist_buff = np.vstack([self.hist_buff, buff])
        self.hist_b = np.append(self.hist_b, rcv - buff[0] + sum(buff))
        self.hist_acc = np.vstack([self.hist_acc, self.acc])
        self.hist_w = np.append(self.hist_w, self.w)
        
        if self.r_req is not None:
            self.hist_r_req = np.append(self.hist_r_req, self.r_req)
        if self.l_req is not None:
            self.hist_l_req = np.append(self.hist_l_req, self.l_req)
        if self.p_req is not None:
            self.hist_p_req = np.append(self.hist_p_req, self.p_req)
        if self.g_req is not None:
            self.hist_g_req = np.append(self.hist_g_req, self.g_req)
        if self.f_req is not None:
            self.hist_f_req = np.append(self.hist_f_req, self.f_req)
    
    # Updates the slice data after the simulation step and model soving
    def updateHistAftStep(
        self,
        r: int,
        sent:np.array
        ):
        self.hist_r = np.append(self.hist_r, r)
        self.hist_sent=np.vstack([self.hist_sent, sent])
        self.acc += sent
    
    