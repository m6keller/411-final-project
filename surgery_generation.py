import random
from schedule import Surgery

def generate_surgery_data(num_samples, num_days_planning_horizon=5, num_surgeons=4):
    surgeries = []
    available_surgeons = [f"Surgeon_{chr(65+i)}" for i in range(num_surgeons)]
    
    for i in range(num_samples):
        duration = random.choice([60, 120, 180])  # Surgery durations in minutes
        surgeon = random.choice(available_surgeons)
        deadline = random.randint(1, num_days_planning_horizon + 2) # Deadlines can be up to 2 days past the planning horizon
        infection_type = random.choice([0, 0, 0, 1, 2]) # Skew towards non-infectious
        
        surgery = Surgery(
            id=i,
            surgeon=surgeon,
            duration=duration,
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
