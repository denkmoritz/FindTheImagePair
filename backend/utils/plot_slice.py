import matplotlib.pyplot as plt
import numpy as np
import math
import random
import io
import base64


def plot_slice(inner_buffer, outer_buffer):
    heading_deg = random.uniform(30, 45)
    heading_rad = math.radians(heading_deg)

    theta = np.linspace(0, 2 * np.pi, 300)
    outer_x = outer_buffer * np.sin(theta)
    outer_y = outer_buffer * np.cos(theta)
    inner_x = inner_buffer * np.sin(theta)
    inner_y = inner_buffer * np.cos(theta)

    corridor_length = outer_buffer * 2
    corridor_width = inner_buffer * 2

    cx = np.array([-corridor_width / 2, corridor_width / 2, corridor_width / 2, -corridor_width / 2])
    cy = np.array([-corridor_length / 2, -corridor_length / 2, corridor_length / 2, corridor_length / 2])

    rot_x = cx * math.cos(-heading_rad) - cy * math.sin(-heading_rad)
    rot_y = cx * math.sin(-heading_rad) + cy * math.cos(-heading_rad)

    def random_point_in_slice():
        for _ in range(1000):
            r = random.uniform(inner_buffer, outer_buffer)
            angle = random.uniform(0, 2 * np.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)

            rx = x * math.cos(heading_rad) - y * math.sin(heading_rad)
            ry = x * math.sin(heading_rad) + y * math.cos(heading_rad)

            if not (-corridor_width / 2 <= rx <= corridor_width / 2 and -corridor_length / 2 <= ry <= corridor_length / 2):
                return x, y
        return None, None

    px, py = random_point_in_slice()

    # Plot setup
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(outer_x, outer_y, color='gray', linestyle='--', linewidth=0.8)
    ax.plot(inner_x, inner_y, color='gray', linestyle='--',  linewidth=0.8)
    ax.fill(rot_x, rot_y, facecolor='none', edgecolor='black', hatch='//', linestyle='--', linewidth=0.7)

    # Annotations – use angles where there’s no corridor (e.g. 135° and 315°)
    def place_annotation(buffer, angle_deg, label):
        angle = math.radians(angle_deg)
        x = buffer * math.cos(angle)
        y = buffer * math.sin(angle)
        label_x = x + 3 * math.cos(angle)
        label_y = y + 3 * math.sin(angle)

        ax.annotate(f"{label} m",
                    xy=(x, y),
                    xytext=(label_x, label_y),
                    arrowprops=dict(arrowstyle="->", linewidth=1),
                    fontsize=9)

    place_annotation(inner_buffer, 135, inner_buffer)
    place_annotation(outer_buffer, 315, outer_buffer)

    # Center and random point
    ax.scatter(0, 0, color='black', label='Center')

    arrow_length = 2
    if px is not None:
        ax.scatter(px, py, color='red', label='Possible Point in Slice')
        ax.arrow(px, py,
                 arrow_length * math.sin(heading_rad),
                 arrow_length * math.cos(heading_rad),
                 head_width=0.3, head_length=0.5, fc='red', ec='red')

    # Heading arrow from center
    ax.arrow(0, 0,
             arrow_length * math.sin(heading_rad),
             arrow_length * math.cos(heading_rad),
             head_width=0.5, head_length=0.8, fc='black', ec='black')

    # Clean visuals
    ax.set_aspect('equal')
    ax.set_title(f'Slice Zone (Heading {heading_deg:.1f}°)', pad=20)

    # Hide spines/ticks
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend below plot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=600, bbox_inches='tight')
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')