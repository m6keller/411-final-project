from typing import Dict, List, Tuple, Literal

# OR is only open from Monday to Friday
Day = Literal["Mon", "Tue", "Wed", "Thu", "Fri"]

# There are 4 surgeons
Surgeon = Literal["Dr_A", "Dr_B", "Dr_C", "Dr_D"]

# Operational hours are from 9 AM to 5 PM (9 to 16)
Operational_time = Literal[9, 10, 11, 12, 13, 14, 15, 16]

# Priority levels for surgeries 0 (optional) and 1 (mandatory)
Priority = Literal[0, 1]

# Operating Rooms available
Operating_Room = Literal["OR_1", "OR_2", "OR_3"]

class Surgery:
    def __init__(
            self, 
            surgery_id: int, 
            surgeon: Surgeon,
            duration: int,
            priority: Priority):
        # Unique identifier for the surgery
        self.id = surgery_id

        # Surgeon assigned to this surgery
        self.surgeon = surgeon

        # Duration of the surgery in minutes
        self.duration = duration

        # Priority level of the surgery 
        self.priority = priority

    def __repr__(self):
        return f"Surgery(id={self.id}, surgeon={self.surgeon}, duration={self.duration}, priority={self.priority})"

class Schedule:
    def __init__(
        self,
        schedule_id: str,
        B_j: int,
        day: Day,
        surgeries: List[int],
        surgeon_work: Dict[Surgeon, int],
        surgeon_busy_times: Dict[Tuple[Surgeon, Day, int, Operating_Room], int]
    ):
        # Unique identifier for the schedule 
        self.id = schedule_id

        # Total "profit" or total minutes of surgeries scheduled in this block (Eq. 5)
        self.B_j = B_j

        # Day of the week (e.g., "Mon", "Tue")
        self.day = day

        # List of surgery IDs included in this schedule
        self.surgeries = surgeries

        # Mapping: surgeon → total minutes they work in this schedule (Eq. 5 data)
        # Example: {"Dr_A": 120, "Dr_B": 120}
        self.surgeon_work = surgeon_work

        # Mapping: (surgeon, day, time_slot) → 1 if surgeon is busy, else absent
        # Used for Eq. (6) constraints in scheduling optimization
        # Example key: ("Dr_A", "Mon", 9)
        # Represents that Dr_A is busy at 9 AM on Monday
        self.surgeon_busy_times = surgeon_busy_times

    def __repr__(self):
        return f"Schedule(id={self.id}, day={self.day})"

# Example usage:
if __name__ == "__main__":  
    surgery1 = Surgery(surgery_id=1, surgeon="Dr_A", duration=120, priority=1)
    surgery2 = Surgery(surgery_id=2, surgeon="Dr_B", duration=90, priority=0)

    schedule = Schedule(
        schedule_id="sched_001",
        B_j=210,
        day="Mon",
        surgeries=[surgery1, surgery2],
        surgeon_work={"Dr_A": 120, "Dr_B": 90},
        surgeon_busy_times={("Dr_A", "Mon", 9): 1, ("Dr_B", "Mon", 11): 1}
    )

    print(schedule)
    print(schedule.surgeries)

