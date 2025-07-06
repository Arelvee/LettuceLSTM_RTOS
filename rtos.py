import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# Configuration
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Arial',
    'axes.titlesize': 12,
    'axes.labelsize': 10
})

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [4, 3]})
fig.suptitle('FreeRTOS Task Trigger Times (Exact Seconds)\nHydroponic Monitoring System', 
             fontsize=14, fontweight='bold', y=0.98)

# Time parameters (20 seconds shown)
total_time = 20  # seconds

# CORE 0 TASKS (Sensor Tasks)
core0_tasks = [
    {'name': 'DHT22', 'period': 2.0, 'duration': 0.15, 'offset': 0.0, 'color': '#1f77b4'},
    {'name': 'Dallas', 'period': 2.0, 'duration': 0.15, 'offset': 0.2, 'color': '#ff7f0e'},
    {'name': 'TDS', 'period': 2.0, 'duration': 0.18, 'offset': 0.4, 'color': '#2ca02c'},
    {'name': 'pH', 'period': 2.0, 'duration': 0.18, 'offset': 0.6, 'color': '#d62728'},
    {'name': 'AS7341', 'period': 2.0, 'duration': 0.25, 'offset': 0.8, 'color': '#9467bd'},
    {'name': 'BH1750', 'period': 2.0, 'duration': 0.15, 'offset': 1.0, 'color': '#8c564b'}
]

# CORE 1 TASKS (Control Tasks)
core1_tasks = [
    {'name': 'Monitor', 'period': 5.0, 'duration': 0.4, 'color': '#17becf'},
    {'name': 'ML (TFLite)', 'period': 10.0, 'duration': 1.5, 'color': '#e377c2'},
    {'name': 'Pump Check', 'period': 60.0, 'duration': 0.1, 'color': '#7f7f7f'}  # Minute checks
]

# Draw CORE 0 Tasks with exact trigger points
for y_pos, task in enumerate(core0_tasks, start=1):
    trigger_times = [task['offset'] + n*task['period'] for n in range(int((total_time-task['offset'])/task['period']) + 1)]
    for t_time in trigger_times:
        if t_time <= total_time:
            ax0.add_patch(Rectangle((t_time, y_pos), 
                                  task['duration'], 0.8,
                                  facecolor=task['color'],
                                  edgecolor='black',
                                  alpha=0.8))
            # Mark trigger point
            ax0.plot(t_time, y_pos + 0.4, '|', color='black', markersize=8)
    ax0.text(-0.5, y_pos + 0.4, f"{task['name']}\n{task['period']}s", ha='right', va='center', fontsize=8)

# Draw CORE 1 Tasks with exact trigger points
for y_pos, task in enumerate(core1_tasks, start=1):
    trigger_times = [n*task['period'] for n in range(int(total_time/task['period']) + 1)]
    for t_time in trigger_times:
        if t_time <= total_time:
            ax1.add_patch(Rectangle((t_time, y_pos), 
                         task['duration'], 0.8,
                         facecolor=task['color'],
                         edgecolor='black',
                         alpha=0.8))
            # Mark trigger point
            ax1.plot(t_time, y_pos + 0.4, '|', color='black', markersize=8)
    ax1.text(-0.5, y_pos + 0.4, f"{task['name']}\n{task['period']}s", ha='right', va='center', fontsize=8)

# Add vertical grid at each second
for ax in (ax0, ax1):
    for t_sec in range(0, total_time+1):
        ax.axvline(x=t_sec, color='gray', linestyle=':', alpha=0.3)

# Formatting
for ax in (ax0, ax1):
    ax.set_xlim(0, total_time)
    ax.set_xticks(np.arange(0, total_time + 1, 1))
    ax.set_xlabel('Exact Trigger Time (seconds)', fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_yticks([])

ax0.set_ylim(0.5, len(core0_tasks) + 1)
ax1.set_ylim(0.5, len(core1_tasks) + 1)

ax0.set_title('CPU CORE 0: Sensor Tasks (All Priority 1)', fontsize=12, pad=10)
ax1.set_title('CPU CORE 1: Control Tasks (ML Priority 2)', fontsize=12, pad=10)

# Legend
legend_elements = [
    *[Patch(facecolor=task['color'], label=f"{task['name']} ({task['period']}s)") for task in core0_tasks],
    *[Patch(facecolor=task['color'], label=f"{task['name']} ({task['period']}s)") for task in core1_tasks],
    Line2D([0], [0], marker='|', color='black', label='Trigger Instant', markersize=10, linestyle='None')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('exact_trigger_times.png', dpi=300, bbox_inches='tight')
plt.show()