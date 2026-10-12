"""Portable flight rendering; positions and body geometry are in metres."""

import warnings
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from .rotations import quat_to_euler, quat_to_matrix

BODY_LENGTH = 0.025
SCENE_BOUNDS = np.array([[1.5, -0.15, 0.35], [2.5, 0.15, 0.65]])
SCENE_SIZE = (1280, 640)
BLUE, ORANGE, GREEN = (0, 114, 188), (226, 126, 34), (0, 132, 94)
DELAY_LABEL = "Задержка"


@lru_cache(maxsize=16)
def font(size, bold=False):
    properties = font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"
    )
    return ImageFont.truetype(font_manager.findfont(properties), size)


def camera_basis(view="perspective"):
    if view == "top":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if view != "perspective":
        raise ValueError("view must be 'perspective' or 'top'")
    azimuth, elevation = np.deg2rad([-55, 20])
    forward = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    return right, np.cross(forward, right)


def make_projector(bounds, width, height, margin=(105, 80, 75, 95), view="perspective"):
    right, up = camera_basis(view)
    corners = np.asarray(list(product(*zip(*bounds))))
    uv = np.column_stack([corners @ right, corners @ up])
    center = (uv.min(axis=0) + uv.max(axis=0)) / 2
    left, right_margin, top, bottom = margin
    available = np.array([width - left - right_margin, height - top - bottom])
    scale = 0.94 * np.min(available / np.maximum(np.ptp(uv, axis=0), 1e-12))
    screen_center = np.array([left, top]) + available / 2

    def project(points):
        points = np.asarray(points, dtype=float)
        return screen_center + (np.column_stack([points @ right, points @ up]) - center) * (
            scale * np.array([1.0, -1.0])
        )

    return project


def convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points))))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for sequence, hull in ((points, lower), (reversed(points), upper)):
        for point in sequence:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
    return lower[:-1] + upper[:-1]


def ellipsoid(center, radii):
    azimuth, elevation = np.meshgrid(
        np.linspace(0, 2 * np.pi, 24, endpoint=False), np.linspace(-np.pi / 2, np.pi / 2, 12)
    )
    unit = np.column_stack(
        [
            (np.cos(elevation) * np.cos(azimuth)).ravel(),
            (np.cos(elevation) * np.sin(azimuth)).ravel(),
            np.sin(elevation).ravel(),
        ]
    )
    return np.asarray(center) + unit * radii


@lru_cache(maxsize=1)
def body_geometry():
    length = BODY_LENGTH
    parts = tuple(
        ellipsoid(np.array(center) * length, np.array(radii) * length)
        for center, radii in (
            ([0, 0, -0.18], [0.15, 0.18, 0.34]),
            ([0, 0, -0.29], [0.145, 0.175, 0.055]),
            ([0, 0, -0.12], [0.165, 0.195, 0.055]),
            ([0, 0, 0.10], [0.22, 0.23, 0.22]),
            ([0, 0, 0.20], [0.21, 0.22, 0.065]),
            ([0, 0, 0.34], [0.15, 0.16, 0.14]),
        )
    )
    wing = (
        np.array(
            [
                [0, 0.05, 0.14],
                [0.02, 0.45, 0.30],
                [0, 0.90, 0.12],
                [-0.02, 0.95, -0.15],
                [-0.02, 0.50, -0.20],
                [0, 0.08, 0.02],
            ]
        )
        * length
    )
    antennae = (
        np.array(
            [[[0.05, 0.05, 0.43], [0.08, 0.13, 0.64]], [[-0.05, -0.05, 0.43], [-0.08, -0.13, 0.64]]]
        )
        * length
    )
    legs = np.array(
        [
            [[0, 0.08, 0.17], [0.05, 0.38, 0.10], [0.08, 0.52, 0.02]],
            [[0, 0.10, 0.07], [0.02, 0.43, -0.04], [0.06, 0.54, -0.16]],
            [[0, 0.08, -0.03], [-0.04, 0.36, -0.17], [-0.08, 0.47, -0.31]],
        ]
    )
    legs = np.concatenate([legs, legs * [1, -1, 1]]) * length
    head_axis = np.array([[0, 0, 0.43], [0, 0, 0.72]]) * length
    return parts, wing, antennae, legs, head_axis


def polyline(draw, points, color, width=2):
    if len(points) > 1:
        draw.line(list(map(tuple, points)), fill=color, width=width, joint="curve")


def check_canvas(points, size, what):
    if np.any(np.min(points, axis=0) < 2) or np.any(np.max(points, axis=0) > np.array(size) - 3):
        raise ValueError(f"{what} leaves the fixed canvas; scale will not be changed")


def arrow(draw, endpoints, color, width=2, head=6):
    start, end = endpoints
    delta = end - start
    norm = np.linalg.norm(delta)
    if norm < 1e-6:
        x, y = end
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        return
    direction = delta / norm
    normal = np.array([-direction[1], direction[0]])
    head = min(head, norm * 0.45)
    base = end - head * direction
    polyline(draw, endpoints, color, width)
    draw.polygon(
        [tuple(end), tuple(base + 0.45 * head * normal), tuple(base - 0.45 * head * normal)],
        fill=color,
    )


def draw_body(image, position, quat, project):
    rotation = quat_to_matrix(quat)
    parts, wing, antennae, legs, head_axis = body_geometry()

    def xy(points):
        return project(points @ rotation.T + position)

    extent = np.vstack(
        [*parts, wing, wing * [1, -1, 1], antennae.reshape(-1, 3), legs.reshape(-1, 3), head_axis]
    )
    check_canvas(xy(extent), image.size, "Bumblebee geometry")
    # Stroke widths follow the projected physical body, including enlarged strip panels.
    pixels = BODY_LENGTH * np.linalg.norm(project(np.eye(3)) - project(np.zeros((1, 3))))
    stroke = max(1, round(pixels / (np.sqrt(2) * 40)))
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    for sign in (-1, 1):
        draw.polygon(
            list(map(tuple, xy(wing * [1, sign, 1]))),
            fill=(154, 207, 226, 125),
            outline=(67, 116, 137, 185),
        )
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    for segment in legs:
        polyline(draw, xy(segment), (42, 36, 27), stroke)
    colors = ("#372b1b", "#efb01d", "#efb01d", "#2c2419", "#e5a31b", "#171817")
    for points, color in zip(parts, colors):
        draw.polygon(convex_hull(xy(points)), fill=color)
    for segment in antennae:
        polyline(draw, xy(segment), (18, 18, 18), stroke)
    axis = xy(head_axis)
    arrow(draw, axis, (202, 54, 43), stroke, np.linalg.norm(axis[1] - axis[0]) * 0.4)
    return image


def force_segments(points, wind_start, wind_end):
    """Clip each edge to the open x interval; never join separate gust passages."""
    segments, current = [], []
    for p, q in zip(points[:-1], points[1:]):
        dx = q[0] - p[0]
        if abs(dx) < 1e-14:
            a, b = (0.0, 1.0) if wind_start < p[0] < wind_end else (1.0, 0.0)
        else:
            cuts = sorted(((wind_start - p[0]) / dx, (wind_end - p[0]) / dx))
            a, b = max(0.0, cuts[0]), min(1.0, cuts[1])
        if b > a:
            start, end = p + a * (q - p), p + b * (q - p)
            if current and a == 0 and np.allclose(current[-1], start, rtol=0, atol=1e-12):
                current.append(end)
            else:
                if current:
                    segments.append(np.asarray(current))
                current = [start, end]
        elif current:
            segments.append(np.asarray(current))
            current = []
    if current:
        segments.append(np.asarray(current))
    return segments


def draw_gust(image, project, history, points, direction, wind_start, wind_end, show_vector=True):
    draw = ImageDraw.Draw(image)
    for segment in force_segments(history, wind_start, wind_end):
        polyline(draw, project(segment), ORANGE, 3)
    segments = force_segments(points, wind_start, wind_end)
    if show_vector and segments:
        center = (segments[0][0] + segments[0][-1]) / 2 + [0, 0, 0.025]
        endpoints = project([center - 0.12 * direction, center + 0.12 * direction])
        check_canvas(endpoints, image.size, "Gust vector")
        arrow(draw, endpoints, GREEN, 3, 10)
        label = tuple(endpoints[1] + [8, -12])
        box = draw.textbbox(label, "F_w", font=font(16, True))
        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
            warnings.warn("Gust vector label extends beyond the fixed canvas", stacklevel=2)
        draw.text(label, "F_w", font=font(16, True), fill=GREEN)


def draw_grid(draw, project, view):
    low, high = SCENE_BOUNDS
    origin = low.copy()
    axes = (0, 1) if view == "top" else (0, 1, 2)
    ticks = (np.linspace(1.5, 2.5, 5), np.array([-0.15, 0, 0.15]), np.array([0.35, 0.5, 0.65]))
    for axis in axes:
        for value in ticks[axis]:
            point = origin.copy()
            point[axis] = value
            for other in axes:
                if other == axis:
                    continue
                end = point.copy()
                end[other] = high[other]
                polyline(draw, project([point, end]), (218, 222, 226), 1)
            x, y = project([point])[0]
            offset = (0, 10) if axis == 0 else (-18, 15 if axis == 1 and view != "top" else 0)
            anchor = "mt" if axis == 0 else "rm"
            draw.text(
                (x + offset[0], y + offset[1]),
                f"{value:g}",
                font=font(14),
                fill=(58, 63, 68),
                anchor=anchor,
            )
        end = origin.copy()
        end[axis] = high[axis]
        polyline(draw, project([origin, end]), (92, 99, 106), 2)
        x, y = project([end])[0]
        offset = (12, 32) if axis == 0 else (-10, -24)
        draw.text(
            (x + offset[0], y + offset[1]),
            f"{'xyz'[axis]}, м",
            font=font(18, True),
            fill=(58, 63, 68),
            anchor="mt",
        )


def labels(image, delay_ms, time=None):
    draw = ImageDraw.Draw(image)
    if time is not None:
        draw.text((28, 20), f"t = {time:+.2f} с", font=font(24), fill=(30, 30, 30))
    draw.text(
        (image.width - 28, 20),
        f"{DELAY_LABEL}: {delay_ms:g} мс",
        font=font(24, True),
        fill=(30, 30, 30),
        anchor="rt",
    )


def prepare_result(result, wind_start, wind_end):
    try:
        states = np.array(result["states"], dtype=float, copy=True)
        time = np.asarray(result["state_time"], dtype=float)
        active = np.asarray(result["wind_active"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("result must contain states, state_time and wind_active arrays") from exc
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 7:
        raise ValueError("Too little flight data: states must have shape (N >= 2, M >= 7)")
    if time.shape != (len(states),) or active.shape != time.shape:
        raise ValueError("state_time and wind_active must have one entry per state")
    if not np.isfinite(states).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("States and times must be finite, with strictly increasing state_time")
    if not np.isin(active, [False, True]).all():
        raise ValueError("wind_active must contain boolean or 0/1 values")
    norms = np.linalg.norm(states[:, 3:7], axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("Flight data contains a zero or unnormalizable quaternion")
    states[:, 3:7] /= norms[:, None]
    inside = np.flatnonzero((states[:, 0] >= 1.5) & (states[:, 0] <= 2.5))
    if len(inside) < 2:
        raise ValueError("Too little flight data inside x = [1.5, 2.5] m")
    if np.any(np.diff(inside) != 1):
        raise ValueError("Flight leaves and re-enters the x window; provide one continuous passage")
    first, last = inside[0], inside[-1]
    selected = states[first : last + 1].copy()
    selected_time = time[first : last + 1].copy()
    selected_active = active[first : last + 1].astype(bool)
    # Add exact x-window intersections without interpolating across quaternion sign flips.
    for before, i, j in ((True, first - 1, first), (False, last, last + 1)):
        if i < 0 or j >= len(states):
            continue
        boundary = float(np.clip(states[i if before else j, 0], 1.5, 2.5))
        fraction = (boundary - states[i, 0]) / (states[j, 0] - states[i, 0])
        if (before and fraction == 1) or (not before and fraction == 0):
            continue
        p, q = states[i].copy(), states[j].copy()
        if np.dot(p[3:7], q[3:7]) < 0:
            q[3:7] *= -1
        state = p + fraction * (q - p)
        state[0] = boundary
        state[3:7] /= np.linalg.norm(state[3:7])
        t = time[i] + fraction * (time[j] - time[i])
        selected = np.vstack([state, selected] if before else [selected, state])
        selected_time = np.r_[t, selected_time] if before else np.r_[selected_time, t]
        selected_active = np.r_[False, selected_active] if before else np.r_[selected_active, False]
    points = selected[:, :3]
    if np.any(points < SCENE_BOUNDS[0] - 1e-12) or np.any(points > SCENE_BOUNDS[1] + 1e-12):
        raise ValueError(
            "Trajectory exceeds fixed scene x=[1.5,2.5], y=[-.15,.15], z=[.35,.65] m; "
            f"observed min={points.min(axis=0)}, max={points.max(axis=0)}"
        )
    spatial_wind = (points[:, 0] > wind_start) & (points[:, 0] < wind_end)
    if not np.any(selected_active & spatial_wind):
        raise ValueError("No active gust in the x window; check wind_active and wind boundaries")
    if np.any(selected_active & ~spatial_wind):
        raise ValueError("wind_active disagrees with the supplied open wind interval")
    segments = force_segments(np.column_stack([points, selected_time]), wind_start, wind_end)
    return selected, selected_time - segments[0][0, 3], selected_active


def make_video(
    path,
    states,
    relative_time,
    active,
    frame_indices,
    fps,
    delay_ms,
    direction,
    view,
    wind_start,
    wind_end,
):
    import av

    points = states[:, :3]
    project = make_projector(SCENE_BOUNDS, *SCENE_SIZE, view=view)
    base = Image.new("RGBA", SCENE_SIZE, "white")
    draw_grid(ImageDraw.Draw(base), project, view)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = SCENE_SIZE
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for index in frame_indices:
            image = base.copy()
            history = points[: index + 1]
            polyline(ImageDraw.Draw(image), project(history), BLUE, 2)
            draw_gust(
                image,
                project,
                history,
                points,
                direction,
                wind_start,
                wind_end,
                show_vector=bool(active[index]),
            )
            image = draw_body(image, points[index], states[index, 3:7], project)
            labels(image, delay_ms, relative_time[index])
            frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB")), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_strip(path, states, relative_time, indices, delay_ms, view):
    panel_width, height = 240, 360
    image = Image.new("RGBA", (panel_width * len(indices), height + 55), "white")
    right, up = camera_basis(view)
    scale = (panel_width - 32) / (2 * BODY_LENGTH)

    def project(points):
        return np.array([panel_width / 2, height / 2]) + np.column_stack(
            [points @ right, points @ up]
        ) * [scale, -scale]

    for order, index in enumerate(indices):
        panel = Image.new("RGBA", (panel_width, height), (246, 246, 244) if order % 2 else "white")
        panel = draw_body(panel, np.zeros(3), states[index, 3:7], project)
        draw = ImageDraw.Draw(panel)
        draw.text(
            (panel_width / 2, 16),
            f"t = {relative_time[index]:+.2f} с",
            font=font(18, True),
            fill=(25, 25, 25),
            anchor="mt",
        )
        roll, pitch, yaw = np.rad2deg(quat_to_euler(states[index, 3:7]))
        angles = f"φ={roll:+.0f}°  θ={pitch:+.0f}°  ψ={yaw:+.0f}°"
        draw.text(
            (panel_width / 2, height - 30), angles, font=font(14), fill=(65, 65, 65), anchor="mt"
        )
        image.paste(panel, (order * panel_width, 55))
    labels(image, delay_ms)
    image.convert("RGB").save(path)


def make_stroboscopic(
    path, states, relative_time, indices, delay_ms, direction, view, wind_start, wind_end
):
    points = states[:, :3]
    project = make_projector(SCENE_BOUNDS, *SCENE_SIZE, view=view)
    image = Image.new("RGBA", SCENE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw_grid(draw, project, view)
    polyline(draw, project(points), BLUE, 2)
    draw_gust(image, project, points, points, direction, wind_start, wind_end)
    for order, index in enumerate(np.unique(indices)):
        image = draw_body(image, points[index], states[index, 3:7], project)
        x, y = project([points[index]])[0]
        y += -34 if order % 2 == 0 else 26
        ImageDraw.Draw(image).text(
            (x, y),
            f"{relative_time[index]:+.2f} с",
            font=font(14),
            fill=(30, 30, 30),
            stroke_width=2,
            stroke_fill="white",
            anchor="mm",
        )
    labels(image, delay_ms)
    image.convert("RGB").save(path)


def render_flight(
    result,
    output_dir,
    delay_ms,
    direction=(0, 1, 0),
    view="perspective",
    fps=20,
    frames=120,
    wind_start=2.0,
    wind_end=2.05,
):
    """Return [MP4, attitude-strip PNG, stroboscopic PNG] as pathlib Paths.

    Quaternions are scalar-first; local +z points toward the head. Time labels
    are relative to the first gust entry. All scene renders use identical bounds
    and a 0.025 m body; only attitude panels use a fixed enlarged metre scale.
    The force arrow denotes direction, not a force magnitude supplied by result.
    """
    camera_basis(view)
    if (
        not isinstance(fps, (int, np.integer))
        or isinstance(fps, (bool, np.bool_))
        or fps < 1
        or not isinstance(frames, (int, np.integer))
        or isinstance(frames, (bool, np.bool_))
        or frames < 2
    ):
        raise ValueError("fps must be a positive integer and frames an integer >= 2")
    if not np.isfinite(delay_ms) or delay_ms < 0:
        raise ValueError("delay_ms must be finite and non-negative")
    if not np.isfinite([wind_start, wind_end]).all() or not 1.5 <= wind_start < wind_end <= 2.5:
        raise ValueError("Wind boundaries must satisfy 1.5 <= wind_start < wind_end <= 2.5 m")
    direction = np.array(direction, dtype=float, copy=True)
    norm = np.linalg.norm(direction)
    if (
        direction.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isfinite(norm)
        or norm < 1e-12
    ):
        raise ValueError("direction must be a finite, nonzero 3-vector")
    direction /= norm
    states, relative_time, active = prepare_result(result, wind_start, wind_end)
    frame_indices = [
        int(np.argmin(abs(relative_time - t)))
        for t in np.linspace(relative_time[0], relative_time[-1], frames)
    ]
    pose_indices = [
        int(np.argmin(abs(relative_time - t)))
        for t in np.linspace(relative_time[0], relative_time[-1], 8)
    ]
    output_dir = Path(output_dir)
    stem = f"flight_delay_{delay_ms:g}ms_{view}"
    paths = [
        output_dir / f"{stem}.mp4",
        output_dir / f"{stem}_strip.png",
        output_dir / f"{stem}_stroboscopic.png",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    make_video(
        paths[0],
        states,
        relative_time,
        active,
        frame_indices,
        int(fps),
        delay_ms,
        direction,
        view,
        wind_start,
        wind_end,
    )
    make_strip(paths[1], states, relative_time, pose_indices, delay_ms, view)
    make_stroboscopic(
        paths[2],
        states,
        relative_time,
        pose_indices,
        delay_ms,
        direction,
        view,
        wind_start,
        wind_end,
    )
    return paths
