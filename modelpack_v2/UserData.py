class UserData:
    def __init__(
            self,
            id: int, # Id of the user
            s: str, # Name of the slice that the user belongs to
            SE: list, # Spectral efficiency (bits/s/Hz) for all steps
            b_max: int # User's buffer packet capacity (packets) 
        ) -> None:
        self.id = id
        self.s = s
        self.SE = SE
        self.b_max = b_max