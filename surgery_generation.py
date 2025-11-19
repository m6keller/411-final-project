import random
from schedule import Surgery

def generate_surgery_data(num_samples, num_days_planning_horizon=5):
    surgeries = []
    for i in range(num_samples):
        duration = random.choice([60, 120, 180])  # Surgery durations in minutes
        priority = random.choice([0, 1])  # Priority levels
        surgeon = random.choice(["Surgeon_A", "Surgeon_B", "Surgeon_C", "Surgeon_D"])  # Example surgeons
        deadline = random.randint(1, num_days_planning_horizon + 2) # Deadlines can be up to 2 days past the planning horizon
        infection_type = random.choice([0, 0, 0, 1, 2]) # Skew towards non-infectious
        
        surgery = Surgery(
            surgery_id=i,
            surgeon=surgeon,
            duration=duration,
            priority=priority,
            deadline=deadline,
            infection_type=infection_type
        )
        surgeries.append(surgery)
    return surgeries

if __name__ == "__main__":
    random.seed(48)
    num_samples = 20
    generated_surgeries = generate_surgery_data(num_samples)
    for surgery in generated_surgeries:
        print(surgery)
