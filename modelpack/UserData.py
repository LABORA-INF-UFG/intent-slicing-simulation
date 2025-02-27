class UserData:
    def __init__(
            self,
            id: int, # Id of the user
            s: str, # Name of the slice that the user belongs to
            b_max: int, # Maximum packet capacity (packets) of the UE buffer
            SE: list # Spectral efficiency (bits/s/Hz) for all steps
        ) -> None:
        self.id = id
        self.s = s
        self.SE = SE