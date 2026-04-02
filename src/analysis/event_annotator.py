"""
Event annotator: overlay known macro and sector events on drift timelines.

Purpose
-------
Drift z-score charts can be hard to interpret in isolation. This module
annotates Plotly figures with vertical lines marking known macro and sector
events, helping analysts quickly answer: "was this drift flag driven by a
genuine company-specific risk change, or was it a market-wide shock that
every company in this sector mentioned?"

The `flag_event_driven_drift` function adds an `event_proximate` boolean
column to drift score DataFrames. True means a drift flag occurred within
±1 year of a known event, which helps separate boilerplate event mentions
(e.g., every company suddenly adding "pandemic" language in 2020) from
idiosyncratic risk developments that deserve deeper analyst scrutiny.

Usage
-----
    from src.analysis.event_annotator import annotate_drift_chart, MACRO_EVENTS

    # In Streamlit dashboard:
    annotate_drift_chart(fig, years=ticker_df["year"].tolist(), sector="Energy")

    # In analysis notebook:
    events = get_events_for_range(2018.0, 2023.0, sector="Financials")
    df = flag_event_driven_drift(drift_df)
    company_specific = df[df["drift_flag"] & ~df["event_proximate"]]
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event databases
# ---------------------------------------------------------------------------

# Macro events: year_float → description
# year_float uses 0.0 for start-of-year events, 0.5 for mid-year, etc.
MACRO_EVENTS: dict[float, str] = {
    2008.5: "Global financial crisis peak",
    2016.0: "Brexit vote; US election uncertainty",
    2018.0: "TCJA tax reform; US tariff escalation begins",
    2019.0: "US-China trade war escalation; Boeing 737 MAX grounding",
    2020.0: "COVID-19 pandemic (global lockdowns, supply chain disruption)",
    2022.0: "Russia-Ukraine war; Fed rate hiking cycle begins",
    2023.0: "US regional banking stress (SVB, Signature); AI boom begins",
}


class SectorEvent(NamedTuple):
    """Immutable record for a sector-specific event."""

    year_float: float
    description: str


# Sector events keyed by GICS sector name
SECTOR_EVENTS: dict[str, list[SectorEvent]] = {
    "Energy": [
        SectorEvent(2014.5, "Oil price crash — Brent drops from $115 to $55"),
        SectorEvent(2016.0, "OPEC output freeze negotiations; US shale cost compression"),
        SectorEvent(2020.25, "COVID demand collapse; WTI briefly goes negative (Apr 2020)"),
        SectorEvent(2022.0, "Russia-Ukraine war drives energy price spike; EU sanctions on Russian gas"),
    ],
    "Financials": [
        SectorEvent(2008.5, "Global financial crisis — Lehman collapse, TARP bailout"),
        SectorEvent(2010.5, "Dodd-Frank Wall Street Reform signed into law"),
        SectorEvent(2018.5, "US bank stress tests; Basel III phased implementation"),
        SectorEvent(2023.25, "SVB, Signature Bank, First Republic failures; FDIC interventions"),
    ],
    "Health Care": [
        SectorEvent(2010.0, "ACA (Affordable Care Act) signed — major reimbursement changes"),
        SectorEvent(2017.0, "ACA repeal attempts; drug pricing scrutiny intensifies"),
        SectorEvent(2020.0, "COVID-19: surge demand for diagnostics, PPE; telehealth acceleration"),
        SectorEvent(2020.75, "FDA Emergency Use Authorizations for COVID vaccines (Pfizer, Moderna)"),
        SectorEvent(2022.5, "Inflation Reduction Act — Medicare drug price negotiation provisions"),
    ],
    "Information Technology": [
        SectorEvent(2018.0, "GDPR enforcement begins (EU); Cambridge Analytica / Facebook scandal"),
        SectorEvent(2019.5, "US antitrust investigations into Big Tech (DOJ, FTC)"),
        SectorEvent(2020.0, "COVID accelerates cloud, remote work, e-commerce adoption"),
        SectorEvent(2022.0, "Global semiconductor shortage peaks; US CHIPS Act proposed"),
        SectorEvent(2023.0, "ChatGPT / generative AI boom reshapes competitive landscape"),
    ],
    "Consumer Discretionary": [
        SectorEvent(2017.0, "Amazon-Whole Foods acquisition; retail disruption narrative peaks"),
        SectorEvent(2019.0, "US-China tariffs raise import costs for consumer goods"),
        SectorEvent(2020.0, "COVID-19 lockdowns crush in-store retail; e-commerce surge"),
        SectorEvent(2022.0, "Supply chain normalisation; consumer spending shifts to services"),
    ],
    "Consumer Staples": [
        SectorEvent(2018.0, "TCJA impact on supply chain; tariffs raise input costs"),
        SectorEvent(2021.0, "Supply chain disruptions; commodity input cost inflation"),
        SectorEvent(2022.5, "Food price inflation peaks globally; shelf-price elasticity debates"),
    ],
    "Industrials": [
        SectorEvent(2018.0, "US steel/aluminium tariffs (Section 232); supply chain reshoring begins"),
        SectorEvent(2019.0, "Boeing 737 MAX grounding — largest single manufacturer impact"),
        SectorEvent(2020.0, "COVID disrupts global manufacturing; aerospace demand collapses"),
        SectorEvent(2021.5, "Port congestion, freight cost spike, semiconductor shortage"),
        SectorEvent(2022.0, "Russia-Ukraine war: titanium, neon supply disruptions; energy cost spike"),
    ],
    "Materials": [
        SectorEvent(2015.0, "China slowdown; commodity supercycle deflation"),
        SectorEvent(2018.0, "US Section 232 steel/aluminium tariffs"),
        SectorEvent(2020.0, "COVID demand shock to industrial metals; gold surges"),
        SectorEvent(2022.0, "Ukraine war disrupts fertiliser and metals supply chains"),
    ],
    "Real Estate": [
        SectorEvent(2020.0, "COVID collapses office/retail REIT valuations; remote work structural shift"),
        SectorEvent(2022.0, "Fed rate hiking cycle raises cap rates; REIT valuations compress"),
        SectorEvent(2023.0, "Regional bank stress tightens commercial real estate lending"),
    ],
    "Utilities": [
        SectorEvent(2020.0, "COVID reduces commercial electricity demand"),
        SectorEvent(2022.0, "European energy crisis raises gas-fired generation costs globally"),
        SectorEvent(2022.5, "US Inflation Reduction Act — large renewable energy tax credits"),
    ],
    "Communication Services": [
        SectorEvent(2018.0, "GDPR enforcement; online advertising regulation scrutiny"),
        SectorEvent(2020.0, "COVID drives streaming, gaming, digital advertising surge"),
        SectorEvent(2021.0, "Apple ATT privacy changes reduce mobile ad targeting precision"),
        SectorEvent(2023.0, "Generative AI threatens search advertising incumbency"),
    ],
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_events_for_range(
    start_year: float,
    end_year: float,
    sector: str | None = None,
) -> list[tuple[float, str]]:
    """Return macro and (optionally) sector events within a year range.

    Parameters
    ----------
    start_year:
        Inclusive lower bound (e.g. 2018.0).
    end_year:
        Inclusive upper bound (e.g. 2023.0).
    sector:
        GICS sector name. If provided, sector-specific events are appended
        after macro events. Pass None to return only macro events.

    Returns
    -------
    list of (year_float, description) tuples, sorted by year_float.
    """
    events: list[tuple[float, str]] = [
        (yr, desc)
        for yr, desc in MACRO_EVENTS.items()
        if start_year <= yr <= end_year
    ]

    if sector and sector in SECTOR_EVENTS:
        events += [
            (ev.year_float, ev.description)
            for ev in SECTOR_EVENTS[sector]
            if start_year <= ev.year_float <= end_year
        ]

    return sorted(events, key=lambda x: x[0])


def annotate_drift_chart(
    fig: go.Figure,
    years: list[float | int],
    sector: str | None = None,
) -> go.Figure:
    """Add vertical event lines and annotations to a Plotly drift timeline.

    Modifies `fig` in-place and also returns it for convenience.

    Each event gets:
    - A dashed vertical line spanning the full chart height
    - A short rotated annotation label near the top of the chart

    Macro events use a grey dashed line; sector-specific events use an
    orange dashed line so analysts can distinguish them at a glance.

    Parameters
    ----------
    fig:
        A Plotly Figure (typically a go.Figure with Scatter traces).
    years:
        List of fiscal years shown on the x-axis. Used to determine the
        visible range for filtering relevant events.
    sector:
        GICS sector name for sector-specific event overlay. Pass None for
        macro events only.

    Returns
    -------
    go.Figure
        The annotated figure (same object as input).
    """
    if not years:
        return fig

    start_year = float(min(years)) - 0.5
    end_year = float(max(years)) + 0.5

    macro_in_range = [
        (yr, desc)
        for yr, desc in MACRO_EVENTS.items()
        if start_year <= yr <= end_year
    ]

    sector_in_range: list[tuple[float, str]] = []
    if sector and sector in SECTOR_EVENTS:
        sector_in_range = [
            (ev.year_float, ev.description)
            for ev in SECTOR_EVENTS[sector]
            if start_year <= ev.year_float <= end_year
        ]

    # Avoid duplicate labels when macro and sector events coincide at same year
    added_years: set[float] = set()

    for yr, desc in macro_in_range:
        fig.add_vline(
            x=yr,
            line_dash="dash",
            line_color="rgba(120, 120, 120, 0.6)",
            line_width=1,
            annotation_text=_truncate_label(desc, max_chars=35),
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="grey",
            annotation_textangle=-60,
        )
        added_years.add(yr)

    for yr, desc in sector_in_range:
        if yr in added_years:
            # Offset slightly to avoid overlapping with macro annotation
            yr_display = yr + 0.05
        else:
            yr_display = yr

        fig.add_vline(
            x=yr_display,
            line_dash="dot",
            line_color="rgba(210, 140, 0, 0.7)",
            line_width=1,
            annotation_text=_truncate_label(desc, max_chars=35),
            annotation_position="top right",
            annotation_font_size=9,
            annotation_font_color="darkorange",
            annotation_textangle=-60,
        )
        added_years.add(yr)

    return fig


def flag_event_driven_drift(
    drift_scores_df: pd.DataFrame,
    proximity_years: float = 1.0,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Add an `event_proximate` column flagging drift near known events.

    A drift flag is considered event-proximate if it occurs within
    ±proximity_years of any macro event (or sector event, if sector info
    is present). This helps distinguish:

    - **Event-proximate flags**: language changes driven by a broad shock
      (e.g., every company adding COVID language in 2020). These are
      expected and may not signal idiosyncratic deterioration.
    - **Non-proximate flags**: language changes outside known events.
      These deserve closer analyst scrutiny as potential leading indicators
      of company-specific risk escalation.

    The column does NOT filter out event-proximate rows — analysts may
    still want to examine them. It is an additional signal for triage.

    Parameters
    ----------
    drift_scores_df:
        DataFrame with at minimum columns: year, drift_flag.
        Optionally a sector column (name controlled by sector_col).
    proximity_years:
        Half-width of the proximity window in years (default: 1.0).
    sector_col:
        Name of the sector column in the DataFrame (default: "sector").

    Returns
    -------
    pd.DataFrame
        New DataFrame (original is not mutated) with an additional
        `event_proximate` boolean column.
    """
    df = drift_scores_df.copy()

    if "year" not in df.columns or "drift_flag" not in df.columns:
        logger.warning("flag_event_driven_drift: required columns 'year' and 'drift_flag' not found")
        df["event_proximate"] = False
        return df

    all_event_years = list(MACRO_EVENTS.keys())
    if sector_col in df.columns:
        for sector in df[sector_col].dropna().unique():
            if sector in SECTOR_EVENTS:
                all_event_years += [ev.year_float for ev in SECTOR_EVENTS[sector]]

    unique_event_years = sorted(set(all_event_years))

    def _is_proximate(row_year: float) -> bool:
        return any(
            abs(row_year - event_year) <= proximity_years
            for event_year in unique_event_years
        )

    df["event_proximate"] = df["year"].apply(lambda y: _is_proximate(float(y)))

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate_label(text: str, max_chars: int) -> str:
    """Truncate annotation text to max_chars, appending '…' if needed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
