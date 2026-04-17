from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import animation


EARTH_RADIUS_KM = 6371.0
MU_KM3_S2 = 3.986004418e5


@st.cache_data(show_spinner=False)
def simulate_constellation(
    num_satellites: int,
    display_mode: str,
    steps: int,
    dt_seconds: int,
    altitude_km: float,
    inclination_deg: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    radius_km = EARTH_RADIUS_KM + altitude_km
    inclination_rad = np.radians(inclination_deg)
    base_motion = np.sqrt(MU_KM3_S2 / radius_km ** 3)

    if display_mode == "line":
        nominal_gap = np.radians(8.3) / max(num_satellites - 1, 1)
        base_phases = np.arange(num_satellites, dtype=np.float64) * nominal_gap
        base_phases += rng.uniform(-0.015 * nominal_gap, 0.015 * nominal_gap, size=num_satellites)
        altitude_offsets = rng.normal(0.0, 0.5, size=num_satellites)
    else:
        base_phases = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=num_satellites))
        altitude_offsets = rng.normal(0.0, 8.0, size=num_satellites)

    radii = radius_km + altitude_offsets
    mean_motion = np.sqrt(MU_KM3_S2 / np.clip(radii, EARTH_RADIUS_KM + 100.0, None) ** 3)

    trajectories = np.zeros((steps, num_satellites, 3), dtype=np.float32)
    phase_history = np.zeros((steps, num_satellites), dtype=np.float32)

    for step_index in range(steps):
        phases = base_phases + mean_motion * step_index * dt_seconds
        phase_history[step_index] = np.mod(phases, 2.0 * np.pi)

        x = radii * np.cos(phases)
        y = radii * np.sin(phases) * np.cos(inclination_rad)
        z = radii * np.sin(phases) * np.sin(inclination_rad)
        trajectories[step_index] = np.stack([x, y, z], axis=1).astype(np.float32)

    phase_spread_deg = np.degrees(np.ptp(np.unwrap(base_phases))) if num_satellites > 1 else 0.0
    altitude_span_km = float(np.max(radii) - np.min(radii)) if num_satellites > 1 else 0.0

    return {
        "trajectories": trajectories,
        "phase_history": phase_history,
        "radius_km": float(radius_km),
        "phase_spread_deg": float(phase_spread_deg),
        "altitude_span_km": altitude_span_km,
    }


def draw_earth(ax, radius_km: float = EARTH_RADIUS_KM) -> None:
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="#2f6c9f", alpha=0.18, linewidth=0)


def build_frame_figure(trajectories: np.ndarray, frame_index: int, display_mode: str) -> plt.Figure:
    positions = trajectories[frame_index]
    max_range = float(np.max(np.linalg.norm(trajectories.reshape(-1, 3), axis=1))) * 1.12

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_earth(ax)

    if display_mode == "line":
        order = np.argsort(np.arctan2(positions[:, 1], positions[:, 0]))
        ordered = positions[order]
        ax.plot(
            ordered[:, 0],
            ordered[:, 1],
            ordered[:, 2],
            color="#e3a008",
            linewidth=1.4,
            alpha=0.75,
        )

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=22, c="#d9480f", alpha=0.95)

    if positions.shape[0] > 0:
        ax.scatter(*positions[0], s=60, c="#0f766e")

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(f"Simulation frame {frame_index + 1}/{trajectories.shape[0]}")
    ax.view_init(elev=26, azim=35 + frame_index * 2)

    fig.tight_layout()
    return fig


def build_gif_bytes(trajectories: np.ndarray, display_mode: str, fps: int = 12) -> bytes:
    max_range = float(np.max(np.linalg.norm(trajectories.reshape(-1, 3), axis=1))) * 1.12
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_earth(ax)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    scatter = ax.scatter([], [], [], s=18, c="#d9480f", alpha=0.95)
    line = None
    if display_mode == "line":
        (line,) = ax.plot([], [], [], color="#e3a008", linewidth=1.2, alpha=0.75)

    def update(frame_index: int):
        positions = trajectories[frame_index]
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        if line is not None:
            order = np.argsort(np.arctan2(positions[:, 1], positions[:, 0]))
            ordered = positions[order]
            line.set_data(ordered[:, 0], ordered[:, 1])
            line.set_3d_properties(ordered[:, 2])
        ax.set_title(f"Simulation frame {frame_index + 1}/{trajectories.shape[0]}")
        ax.view_init(elev=24, azim=35 + frame_index * 2)
        artists = [scatter]
        if line is not None:
            artists.append(line)
        return tuple(artists)

    ani = animation.FuncAnimation(fig, update, frames=trajectories.shape[0], interval=1000 // fps, blit=False)
    buffer = BytesIO()
    ani.save(buffer, writer=animation.PillowWriter(fps=fps), format="gif")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_spacing_series(phase_history: np.ndarray, display_mode: str) -> np.ndarray:
    if phase_history.shape[1] < 2:
        return np.zeros(phase_history.shape[0], dtype=np.float32)

    if display_mode == "line":
        ordered = np.sort(np.unwrap(phase_history, axis=1), axis=1)
        gaps = np.diff(ordered, axis=1)
    else:
        sorted_phases = np.sort(phase_history, axis=1)
        wrapped = np.concatenate([sorted_phases, sorted_phases[:, :1] + 2.0 * np.pi], axis=1)
        gaps = np.diff(wrapped, axis=1)

    return gaps.std(axis=1).astype(np.float32)


st.set_page_config(page_title="Constellation Simulation Viewer", layout="wide")

st.title("Constellation Simulation Viewer")
st.write("Select the number of satellites, choose a line or random display, and inspect the simulated orbital motion.")

with st.sidebar:
    st.header("Controls")
    display_mode = st.selectbox("Display", options=["line", "random"], index=0)
    default_satellites = 30 if display_mode == "line" else 100
    num_satellites = st.slider("Number of satellites", min_value=2, max_value=300, value=default_satellites)
    steps = st.slider("Simulation steps", min_value=20, max_value=180, value=90, step=10)
    dt_seconds = st.slider("Step duration (seconds)", min_value=10, max_value=300, value=60, step=10)
    altitude_km = st.slider("Altitude (km)", min_value=300.0, max_value=1200.0, value=550.0, step=10.0)
    inclination_deg = st.slider("Inclination (deg)", min_value=0.0, max_value=98.0, value=53.0, step=1.0)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    show_animation = st.toggle("Generate GIF preview", value=False)

simulation = simulate_constellation(
    num_satellites=num_satellites,
    display_mode=display_mode,
    steps=steps,
    dt_seconds=dt_seconds,
    altitude_km=altitude_km,
    inclination_deg=inclination_deg,
    seed=int(seed),
)

trajectories = simulation["trajectories"]
phase_history = simulation["phase_history"]
frame_index = st.slider("Frame", min_value=0, max_value=steps - 1, value=0)

metric_cols = st.columns(4)
metric_cols[0].metric("Satellites", f"{num_satellites}")
metric_cols[1].metric("Altitude", f"{altitude_km:.0f} km")
metric_cols[2].metric("Initial spread", f"{float(simulation['phase_spread_deg']):.1f} deg")
metric_cols[3].metric("Altitude span", f"{float(simulation['altitude_span_km']):.2f} km")

left_col, right_col = st.columns([1.7, 1.0])

with left_col:
    frame_figure = build_frame_figure(trajectories, frame_index, display_mode)
    st.pyplot(frame_figure, clear_figure=True)

with right_col:
    spacing_series = build_spacing_series(phase_history, display_mode)
    st.line_chart(
        {
            "orbit irregularity": spacing_series,
        },
        height=260,
    )
    altitude_offsets = np.linalg.norm(trajectories[0], axis=1) - float(simulation["radius_km"])
    st.caption("Altitude offsets at frame 0")
    st.bar_chart(altitude_offsets)

if show_animation:
    with st.spinner("Rendering GIF preview..."):
        gif_bytes = build_gif_bytes(trajectories, display_mode)
    st.image(gif_bytes, caption="Simulation preview", use_container_width=False)
    st.download_button(
        "Download GIF",
        data=gif_bytes,
        file_name=f"constellation_{display_mode}_{num_satellites}.gif",
        mime="image/gif",
    )
