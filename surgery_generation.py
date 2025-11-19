import random
from schedule import Surgery

def generate_surgery_data(num_samples):
    surgeries = []
    for i in range(num_samples):
        duration = random.choice([60, 120, 180])  # Surgery durations in minutes
        priority = random.choice([0, 1])  # Priority levels
        surgeon = random.choice(["Dr_A", "Dr_B", "Dr_C", "Dr_D"])  # Example surgeons
        surgery = Surgery(
            surgery_id=i,
            surgeon=surgeon,
            duration=duration,
            priority=priority
        )
        surgeries.append(surgery)
    return surgeries

if __name__ == "__main__":
    random.seed(48)
    num_samples = 20
    generated_surgeries = generate_surgery_data(num_samples)
    for surgery in generated_surgeries:
        print(surgery)
