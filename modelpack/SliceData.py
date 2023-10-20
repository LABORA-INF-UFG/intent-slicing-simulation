class SliceData:
    def __init__(
            self,
            id: str,
            b_max: int,
            l_max: int,
            r_req: float = None,
            l_req: float = None,
            p_req: float = None,
            f_req: float = None,
            g_req: float = None
        ):
        self.id = id
        self.b_max = b_max
        self.r_req = r_req
        self.l_req = l_req
        self.p_req = p_req
        self.f_req = f_req
        self.g_req = g_req
        self.users = dict()
        self.buffer = [0]*(l_max+1)
        self.hist_d = []
        self.hist_b = []
        self.hist_request = []
        self.hist_r = []
        self.hist_total_sent = []
        
    def addUser (self, u) -> None:
        self.users[u.id] = u
    
    def saveHist (self, d, b, request, r, total_sent):
        self.hist_d.append(d)
        self.hist_b.append(b)
        self.hist_request.append(request)
        self.hist_r.append(r)
        self.hist_total_sent.append(total_sent)
    
    def getSort (self):
        return sorted(self.hist_r)
    
    def addArrivingPackets(self, packets):
        self.buffer[0] += packets