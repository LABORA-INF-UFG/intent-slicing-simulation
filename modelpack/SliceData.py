class SliceData:
    def __init__(
            self,
            id: str, # Name of the slice
            b_s_max: int, # Maximum buffer size for the slice buffer, i.e. for all user buffers summed (packets) 
            hist_r: list, # Historical served throughput (list of Mb/s or Kb/ms)
            hist_d: list, # Historical dropped packets (list of packets)
            hist_rcv: list, # Historical received packets (list of packets)
            hist_buff: list, # Historical buffer packets (list of list of packets)
            hist_sent: list, # Historical sent packets (list of list of packets)
            hist_part: list, # Historical partially sent packets (list of part of packets)
            r_req: float = None, # Minimum required served throughput for each user (Mb/s or Kb/ms)
            l_req: float = None, # Maximum required average buffer latency (ms or TTIs or steps)
            p_req: float = None, # Maximum required packet loss rate (ratio)
            f_req: float = None, # Minimum required fifth-percentile served throughput (Mb/s or Kb/s)
            g_req: float = None # Minimum required long-term served throughput (Mb/s or Kb/s)
        ):
        self.id = id
        self.b_s_max = b_s_max
        self.hist_d = hist_d
        self.hist_buff = hist_buff
        self.hist_rcv = hist_rcv
        self.hist_r = hist_r
        self.hist_sent = hist_sent
        self.hist_part = hist_part
        self.r_req = r_req
        self.l_req = l_req
        self.p_req = p_req
        self.f_req = f_req
        self.g_req = g_req
        
        # Initializing a dictionary for saving users
        self.users = dict()

        # Calculating the accumulated sent packets of previous steps for each step n (list of list of packets)
        n_steps = len(self.hist_sent)
        l_max_plus_1 = len(self.hist_sent[0])
        self.hist_acc = [[]]*n_steps
        self.hist_acc[0] = [0]*l_max_plus_1
        for n in range(1, n_steps):
            self.hist_acc[n] = [0]*l_max_plus_1
            for i in range(0, l_max_plus_1):
                self.hist_acc[n][i] = self.hist_acc[n-1][i] + self.hist_sent[n-1][i]
        
        # Calculating the number of packets on the buffer at the beggining of each step,
        # considering packets that tried to arrive the buffer but were dropped (list of packets)
        self.hist_b_s = []*n_steps
        for n in range(len(self.hist_buff)):
            self.hist_b_s[n] = self.hist_rcv[n] - self.hist_buff[n][0] + sum(self.hist_buff[n])

    # Associates user with the slice
    def addUser (self, u) -> None:
        self.users[u.id] = u
    
    # Returns a sorted list of throughputs in a window that ends in step - 1
    def getSortedThroughputWindow (self, w, n):
        return sorted(self.hist_r[n-w+1:n])