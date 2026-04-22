"""
logos.py
--------
Team logo loading and figure-placement helpers.

Logo files are stored in:
  italy-serie-a-2025-2026.football-logos.cc/{size}x{size}/<slug>.football-logos.cc.png

Naming quirks handled:
  - "Como"         → "como-1907"
  - "Hellas Verona" → "verona"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

LOGO_BASE_DIR = Path(__file__).parent / "images" / "logos"

# Lowercase team name → logo slug  (None = no logo available)
_NAME_TO_SLUG: dict[str, str | None] = {
    "como":              "como-1907",
    "hellas verona":     "verona",
    "borussia dortmund": None,
    "as roma":           "roma",
    "inter milan":       "inter",
    "ac milan":          "milan",
}


def _get_slug(team_name: str) -> str | None:
    key = team_name.lower().strip()
    if key in _NAME_TO_SLUG:
        return _NAME_TO_SLUG[key]
    return key.replace(" ", "-")


def get_logo_path(team_name: str, size: int = 256) -> Path | None:
    """Return Path to team PNG logo at the given pixel size, or None."""
    slug = _get_slug(team_name)
    if slug is None:
        return None
    path = LOGO_BASE_DIR / f"{slug}.football-logos.cc.png"
    return path if path.exists() else None


def get_logo_image(team_name: str, size: int = 256) -> Optional[np.ndarray]:
    """Load and return team logo as a numpy RGBA array, or None if unavailable."""
    path = get_logo_path(team_name, size)
    if path is None:
        return None
    try:
        import matplotlib.pyplot as plt
        return plt.imread(str(path))
    except Exception:
        return None


def add_team_logos(
    fig,
    home_name: str,
    away_name: str,
    y_bottom: float = 0.957,
    logo_height_in: float = 0.65,
    x_margin: float = 0.30,
) -> None:
    """
    Add home (left) and away (right) logos to the figure, flanking the suptitle.

    Parameters
    ----------
    fig            : matplotlib Figure
    home_name      : home team name → left logo
    away_name      : away team name → right logo
    y_bottom       : bottom edge of logo axes in figure fractions
    logo_height_in : desired logo height in inches (logos are square)
    x_margin       : horizontal distance (figure fraction) from centre toward edge
                     where the logo is centred. Default 0.12 places logos close to
                     the title text rather than at the extreme corners.
    """
    fig_w, fig_h = fig.get_size_inches()
    lw = logo_height_in / fig_w   # width as figure fraction
    lh = logo_height_in / fig_h   # height as figure fraction

    home_img = get_logo_image(home_name, size=256)
    away_img = get_logo_image(away_name, size=256)

    # Centre the logos at (0.5 - x_margin) and (0.5 + x_margin)
    x_left  = 0.5 - x_margin - lw / 2
    x_right = 0.5 + x_margin - lw / 2

    if home_img is not None:
        ax_home = fig.add_axes([x_left, y_bottom, lw, lh])
        ax_home.imshow(home_img)
        ax_home.axis("off")

    if away_img is not None:
        ax_away = fig.add_axes([x_right, y_bottom, lw, lh])
        ax_away.imshow(away_img)
        ax_away.axis("off")
