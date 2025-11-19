from typing import Dict, List, Tuple, Literal
from pydantic import BaseModel

# Operational hours are from 9 AM to 5 PM (9 to 16)
Operational_time = Literal[9, 10, 11, 12, 13, 14, 15, 16]

# Priority levels for surgeries 0 (optional) and 1 (mandatory)
Priority = Literal[0, 1]

class Surgery(BaseModel):
    # Unique identifier for the surgery
    id: int
    # Surgeon assigned to this surgery
    surgeon: str
    # Duration of the surgery in minutes
    duration: int
    # Priority level of the surgery 
    priority: Priority
    # Day by which surgery must be completed
    deadline: int
    # Type of infection (0 for none)
    infection_type: int

class Schedule(BaseModel):
    # Unique identifier for the schedule 
    id: str
    # Total "profit" or total minutes of surgeries scheduled in this block (Eq. 5)
    B_j: int
    # Day of the week (e.g., "Mon", "Tue")
    day: str
    # List of surgeries (Surgery class) included in this schedule
    surgeries: List[int]
    # List of surgeries (Surgery class) included in this schedule
    surgeries_data: List[Surgery]
    # Mapping: surgeon → total minutes they work in this schedule (Eq. 5 data)
        # Example: {"Dr_A": 120, "Dr_B": 120}
    surgeon_work: Dict[str, int]
     # Mapping: (surgeon, day, time_slot) → 1 if surgeon is busy, else absent
        # Used for Eq. (6) constraints in scheduling optimization
        # Example key: ("Dr_A", "Mon", 9)
        # Represents that Dr_A is busy at 9 AM on Monday
    surgeon_busy_times: Dict[Tuple[str, str, int], int]


# Example usage:
if __name__ == "__main__":  
    surgery1 = Surgery(id=1, surgeon="Dr_A", duration=120, priority=1, deadline=1, infection_type=0)
    surgery2 = Surgery(id=2, surgeon="Dr_B", duration=90, priority=0, deadline=2, infection_type=1)

    schedule = Schedule(
        id="sched_001",
        B_j=210,
        day="Mon",
        surgeries=[surgery1.id, surgery2.id],
        surgeon_work={"Dr_A": 120, "Dr_B": 90},
        surgeon_busy_times={("Dr_A", "Mon", 9): 1, ("Dr_B", "Mon", 11): 1},
        surgeries_data =[surgery1, surgery2]
    )

    print(schedule)
    print(schedule.surgeries_data)

