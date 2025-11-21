import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pandas as pd
import seaborn as sns
import os
from collections import defaultdict

from schedule import Schedule, Surgery # Import Schedule and Surgery

def get_surgeon_colors(surgeons):
    """Creates a consistent color map for surgeons."""
    colors = {}
    # Generate visually distinct colors
    for i, surgeon in enumerate(surgeons):
        hue = i / len(surgeons)
        colors[surgeon] = plt.cm.get_cmap('hsv', len(surgeons))(i)
    return colors

def visualize_schedule(results, scenario_params, file_path):
    """
    Generates and saves a Gantt chart visualization of a schedule, with a separate subplot for each day.
    """
    selected_schedules = results.get("selected_schedules", [])
    
    num_surgeries = len(scenario_params["all_surgeries_data"])
    num_surgeons = len(scenario_params["all_surgeons"])
    num_days = len(scenario_params["all_days"])
    num_ors = next(iter(scenario_params["K_d"].values())) if scenario_params["K_d"] else 0

    all_days = scenario_params["all_days"]
    
    # --- Create Subplots for Each Day ---
    fig, axes = plt.subplots(1, num_days, figsize=(5 * num_days, 10), sharey=True)
    if num_days == 1: # If there's only one day, axes is not a list
        axes = [axes]

    fig.suptitle(
        f"Schedule for {results['scenario_name']} | {num_surgeries} surgeries, {num_surgeons} surgeons, {num_days} days, {num_ors} ORs",
        fontsize=16
    )

    if not selected_schedules:
        print(f"No schedules to visualize for {results['scenario_name']}.")
        for i, day in enumerate(all_days):
            ax = axes[i]
            ax.set_title(day)
            ax.text(0.5, 0.5, "No feasible schedule.", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel("Time (minutes)")
        axes[0].set_ylabel("Operating Rooms")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(file_path)
        plt.close()
        return

    all_surgeons = scenario_params["all_surgeons"]
    K_d = scenario_params["K_d"]
    mandatory_surgeries = set(scenario_params["mandatory_surgeries"])
    surgeon_colors = get_surgeon_colors(all_surgeons)

    # --- Group schedules by day ---
    schedules_by_day = defaultdict(list)
    for sched in selected_schedules:
        schedules_by_day[sched.day].append(sched)

    # --- Plot Each Day on a Separate Subplot ---
    for i, day in enumerate(all_days):
        ax = axes[i]
        ax.set_title(day)
        
        day_or_count = K_d.get(day, 0)
        y_labels = [f"OR {j+1}" for j in range(day_or_count)]
        y_ticks = [j * 10 + 5 for j in range(day_or_count)]
        
        ax.set_ylim(0, day_or_count * 10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlim(0, 480) # 8-hour day
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Time (minutes)")

        or_day_counter = 0
        for schedule in schedules_by_day[day]:
            if or_day_counter < day_or_count:
                y_base = or_day_counter * 10
                or_day_counter += 1
            else:
                print(f"Warning: Not enough ORs to visualize schedule {schedule.id} on {day}. Placing in last OR.")
                y_base = (day_or_count - 1) * 10

            for surg_obj in schedule.surgeries_data:
                surg_id = surg_obj.id
                start_time = schedule.start_times.get(surg_id, 0)
                duration = surg_obj.duration
                surgeon = surg_obj.surgeon
                is_mandatory = surg_id in mandatory_surgeries
                
                rect = patches.Rectangle(
                    (start_time, y_base + 1), 
                    duration, 
                    8,
                    edgecolor='black' if is_mandatory else 'gray',
                    facecolor=surgeon_colors.get(surgeon, 'gray'),
                    linewidth=1.5 if is_mandatory else 1,
                    linestyle='-' if is_mandatory else '--'
                )
                ax.add_patch(rect)
                
                ax.text(
                    start_time + duration / 2, 
                    y_base + 5, 
                    f"Surg {surg_id}\n({surgeon})",
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold'
                )

    axes[0].set_ylabel("Operating Rooms")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(file_path)
    plt.close()
    print(f"Saved schedule visualization to {file_path}")


def create_summary_visualizations(csv_path):
    """
    Generates summary visualizations from the experiment results CSV.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_dir = os.path.dirname(csv_path)

    # --- 1. Bar plot for scheduled vs. unscheduled surgeries ---
    df['unscheduled_surgeries'] = df['total_surgeries'] - df['successful_surgeries']
    
    plt.figure(figsize=(15, 8))
    # Sort by number of successful surgeries for better visualization
    df_plot = df.sort_values('successful_surgeries').set_index('scenario_name')
    df_plot[['successful_surgeries', 'unscheduled_surgeries']].plot(kind='bar', stacked=True, colormap='viridis', figsize=(15,8))
    
    plt.title('Number of Scheduled vs. Unscheduled Surgeries per Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Number of Surgeries')
    plt.xticks(rotation=75, ha='right')
    plt.legend(['Scheduled', 'Unscheduled'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_scheduled_vs_unscheduled.png'))
    plt.close()
    print(f"Saved summary bar plot to {output_dir}")

    # --- 2. Heatmaps for parameter pairs vs. successful surgeries ---
    param_pairs = [
        ('num_surgeons', 'num_days'),
        ('num_surgeons', 'num_ors'),
        ('num_days', 'num_ors')
    ]

    for x_param, y_param in param_pairs:
        plt.figure(figsize=(10, 8))
        
        pivot_table = df.pivot_table(
            values='successful_surgeries', 
            index=y_param, 
            columns=x_param,
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
        
        plt.title(f'Successful Surgeries by {x_param.replace("_", " ").title()} and {y_param.replace("_", " ").title()}')
        plt.xlabel(x_param.replace("_", " ").title())
        plt.ylabel(y_param.replace("_", " ").title())
        
        heatmap_path = os.path.join(output_dir, f'heatmap_{x_param}_vs_{y_param}.png')
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved heatmap to {heatmap_path}")