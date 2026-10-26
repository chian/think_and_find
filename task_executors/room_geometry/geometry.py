"""Small geometry helpers for room-scale outbreak calculations."""

from __future__ import annotations

from .._validation import positive


def room_volume(
    floor_area_m2: float,
    room_height_m: float,
) -> float:
    """Return the volume of a constant-height room."""

    return positive(floor_area_m2, "floor_area_m2") * positive(
        room_height_m,
        "room_height_m",
    )


def room_height(
    room_volume_m3: float,
    floor_area_m2: float,
) -> float:
    """Return the effective height of a constant-area room."""

    return positive(room_volume_m3, "room_volume_m3") / positive(
        floor_area_m2,
        "floor_area_m2",
    )
