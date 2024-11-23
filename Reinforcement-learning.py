import matplotlib.pyplot as plt
import numpy as np
import random

num_teachers = 5
num_classes = 10
num_time_slots = 8
num_days = 5
num_rooms = 3

q_table = np.zeros((num_classes, num_days * num_time_slots * num_rooms))

alpha = 0.1 
gamma = 0.95  
epsilon = 1.0 
epsilon_decay = 0.99
min_epsilon = 0.1
episodes = 1000

teacher_availability = np.random.randint(2, size=(num_teachers, num_days, num_time_slots))
room_availability = np.random.randint(2, size=(num_rooms, num_days, num_time_slots))

def calculate_reward(class_id, day, time_slot, room):
    teacher = class_id % num_teachers 
    if teacher_availability[teacher, day, time_slot] == 0:
        return -10  
    if room_availability[room, day, time_slot] == 0:
        return -10  
    return 10  

def visualize_schedule(schedule, num_days=5, num_time_slots=8, num_rooms=3, num_teachers=5):
    teacher_colors = ['blue', 'green', 'red', 'purple', 'orange']
    visual_schedule = np.full((num_days, num_time_slots, num_rooms), -1)

    for class_id, (day, time_slot, room) in schedule.items():
        visual_schedule[day, time_slot, room] = class_id

    fig, axes = plt.subplots(num_days, 1, figsize=(12, 15))
    for day in range(num_days):
        ax = axes[day]
        ax.set_title(f'Day {day + 1}', fontsize=14)
        ax.set_xticks(range(num_time_slots))
        ax.set_yticks(range(num_rooms))
        ax.set_xticklabels([f'Time {i + 1}' for i in range(num_time_slots)])
        ax.set_yticklabels([f'Room {i + 1}' for i in range(num_rooms)])

        for time_slot in range(num_time_slots):
            for room in range(num_rooms):
                class_id = visual_schedule[day, time_slot, room]
                if class_id != -1:
                    teacher_id = class_id % num_teachers
                    ax.text(
                        time_slot, room, f'C{class_id}\nT{teacher_id}',
                        ha='center', va='center', fontsize=10, color='white',
                        bbox=dict(facecolor=teacher_colors[teacher_id], edgecolor='black')
                    )
        ax.imshow(visual_schedule[day], cmap='gray', aspect='auto', alpha=0.2, extent=(0, num_time_slots, num_rooms, 0))
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim(0, num_time_slots)
        ax.set_ylim(num_rooms, 0)
    plt.tight_layout()
    plt.show()

# RL trening
for episode in range(episodes):
    for class_id in range(num_classes):
        state = class_id
        action = (
            random.randint(0, num_days - 1), 
            random.randint(0, num_time_slots - 1), 
            random.randint(0, num_rooms - 1)
        ) if random.random() < epsilon else np.unravel_index(np.argmax(q_table[state]), (num_days, num_time_slots, num_rooms))

        day, time_slot, room = action
        reward = calculate_reward(class_id, day, time_slot, room)

        next_state = (state + 1) % num_classes
        max_future_q = np.max(q_table[next_state])
        current_q = q_table[state][np.ravel_multi_index(action, (num_days, num_time_slots, num_rooms))]

        q_table[state][np.ravel_multi_index(action, (num_days, num_time_slots, num_rooms))] = current_q + alpha * (
            reward + gamma * max_future_q - current_q
        )

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Trening završen!")

schedule = {}
for class_id in range(num_classes):
    best_action = np.unravel_index(np.argmax(q_table[class_id]), (num_days, num_time_slots, num_rooms))
    schedule[class_id] = best_action

visualize_schedule(schedule)
