from typing import Dict, List, Tuple, Literal
from pydantic import BaseModel

class Surgery(BaseModel):
    id: int
    surgeon: str
    duration: int              # Duration in minutes
    deadline: int              # Day index
    infection_type: int        # 0=None, >0=Specific Type

class Schedule(BaseModel):
    id: str
    B_j: int                                       # Total duration/profit (Eq. 5)
    day: str
    surgeries: List[int]                           # Surgery IDs (for Solver)
    surgeries_data: List[Surgery]                  # Full objects (for Reporting)
    surgeon_work: Dict[str, int]                   # {Surgeon: Total Minutes}
    surgeon_busy_times: Dict[Tuple[str, int], int] # {(Surgeon, Time): 60} (Eq. 6 Coloring)
    start_times: Dict[int, int]                    # {Surgery ID: Start Minute (0-480)}

# Example usage:
if __name__ == "__main__":
    s1 = Surgery(id=1, surgeon="Dr_A", duration=120, deadline=1, infection_type=0)
    s2 = Surgery(id=2, surgeon="Dr_B", duration=90, deadline=2, infection_type=1)

    schedule = Schedule(
        id="sched_001",
        B_j=210,
        day="Mon",
        surgeries=[s1.id, s2.id],
        surgeries_data=[s1, s2],
        surgeon_work={"Dr_A": 120, "Dr_B": 90},
        surgeon_busy_times={("Dr_A", 0): 120, ("Dr_B", 120): 90},
        start_times={1: 0, 2: 120}
    )

    print(f"Schedule Cost: {schedule.B_j}")
    print(f"First Surgery Start: {schedule.start_times[s1.id]}")