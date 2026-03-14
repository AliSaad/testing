"""
Energy Margin Optimizer
=======================
Standalone Streamlit app — works on Streamlit Cloud or Databricks Apps.

Features:
  • Real day-ahead wholesale prices via ENTSO-E Transparency Platform API
  • Simulated fallback curve if no API key is provided
  • Multi-client portfolio management (add / edit / remove)
  • Per-client and aggregate margin analysis
  • Optimal procurement window finder
  • CSV export of results

Setup:
  pip install streamlit pandas plotly requests

Run:
  streamlit run energy_margin_app.py

ENTSO-E API key (free):
  Register at https://transparency.entsoe.eu → My Account → Security Token
  Paste the token in the sidebar when the app loads.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
import json
import io


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Energy Margin Optimizer",
    page_icon="⚡",
    layout="wide",
)

# ── ENTSO-E bidding zones (most common) ───────────────────────────────────────

BIDDING_ZONES = {
    "Great Britain":    "10YGB----------A",
    "Germany/Lux":     "10Y1001A1001A82H",
    "France":          "10YFR-RTE------C",
    "Netherlands":     "10YNL----------L",
    "Belgium":         "10YBE----------2",
    "Spain":           "10YES-REE------0",
    "Italy (North)":   "10Y1001A1001A73I",
    "Norway (NO1)":    "10YNO-1--------2",
    "Sweden (SE3)":    "10Y1001A1001A46L",
    "Denmark (DK1)":   "10YDK-1--------W",
}

CURRENCY_SYMBOL = {"Great Britain": "£", "default": "€"}

# ── Session state defaults ────────────────────────────────────────────────────

def init_state():
    if "clients" not in st.session_state:
        st.session_state.clients = [
            {"name": "Acme Manufacturing",  "rate_p_kwh": 15.5, "consumption_mwh": 320},
            {"name": "Greenfields Logistics","rate_p_kwh": 14.0, "consumption_mwh": 180},
            {"name": "Northgate Coldstore",  "rate_p_kwh": 16.2, "consumption_mwh": 500},
        ]
    if "prices" not in st.session_state:
        st.session_state.prices = None
    if "price_date" not in st.session_state:
        st.session_state.price_date = None
    if "zone" not in st.session_state:
        st.session_state.zone = "Great Britain"

init_state()


# ── ENTSO-E fetch ─────────────────────────────────────────────────────────────

def fetch_entso_prices(api_key: str, zone_eic: str, target_date: date) -> pd.Series | None:
    """Fetch day-ahead prices from ENTSO-E for a given date and bidding zone.
    Returns a pd.Series indexed 0–23 (hourly p/kWh) or None on failure."""
    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0)
    end   = start + timedelta(days=1)
    fmt   = "%Y%m%d%H%M"
    url   = "https://web-api.tp.entsoe.eu/api"
    params = {
        "securityToken": api_key,
        "documentType":  "A44",
        "in_Domain":     zone_eic,
        "out_Domain":    zone_eic,
        "periodStart":   start.strftime(fmt),
        "periodEnd":     end.strftime(fmt),
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return parse_entso_xml(r.text)
    except Exception as e:
        st.error(f"ENTSO-E fetch failed: {e}")
        return None


def parse_entso_xml(xml_text: str) -> pd.Series | None:
    """Parse ENTSO-E XML response → hourly prices in p/kWh (converted from €/MWh)."""
    try:
        root = ET.fromstring(xml_text)
        ns   = {"ns": root.tag.split("}")[0].lstrip("{")} if "}" in root.tag else {}
        tag  = lambda t: f"ns:{t}" if ns else t

        prices_raw = []
        for ts in root.findall(f".//{tag('TimeSeries')}", ns):
            res_tag = ts.find(f".//{tag('resolution')}", ns)
            if res_tag is None or res_tag.text != "PT60M":
                continue
            for pt in ts.findall(f".//{tag('Point')}", ns):
                pos   = int(pt.find(tag("position"), ns).text)
                price = float(pt.find(tag("price.amount"), ns).text)
                prices_raw.append((pos, price))

        if not prices_raw:
            return None

        prices_raw.sort(key=lambda x: x[0])
        # ENTSO-E returns €/MWh — convert to p/kWh  (×100 / 1000 = ×0.1)
        # For GB, approximate EUR→GBP at 0.86
        hourly = [p * 0.1 * 0.86 for _, p in prices_raw[:24]]
        while len(hourly) < 24:
            hourly.append(hourly[-1])
        return pd.Series(hourly, name="price_p_kwh")

    except Exception as e:
        st.error(f"XML parse error: {e}")
        return None


# ── Simulated price curve ─────────────────────────────────────────────────────

def simulated_prices() -> pd.Series:
    """Realistic UK industrial wholesale price shape (p/kWh)."""
    base = [
        8.2, 7.8, 7.4, 7.1, 7.3, 8.5,
        11.2, 14.8, 16.3, 13.1, 12.4, 11.8,
        11.2, 10.8, 10.5, 10.9, 12.1, 15.4,
        18.7, 19.2, 16.8, 13.4, 11.2, 9.8,
    ]
    return pd.Series(base, name="price_p_kwh")


# ── Maths ─────────────────────────────────────────────────────────────────────

def best_window(prices: pd.Series, duration_h: int) -> int:
    """Return start hour of the lowest-cost window of given duration."""
    best_start, best_avg = 0, float("inf")
    for s in range(24 - duration_h + 1):
        avg = prices.iloc[s:s + duration_h].mean()
        if avg < best_avg:
            best_avg = avg
            best_start = s
    return best_start


def window_stats(prices: pd.Series, start: int, dur: int, rate_p_kwh: float, consumption_mwh: float):
    end       = min(start + dur, 24)
    avg_w     = prices.iloc[start:end].mean()
    wh_cost   = avg_w * consumption_mwh * 10       # p/kWh × MWh × 10 = £
    revenue   = rate_p_kwh * consumption_mwh * 10
    margin    = revenue - wh_cost
    margin_pc = (margin / revenue * 100) if revenue else 0
    return dict(
        avg_wholesale=avg_w,
        wholesale_cost=wh_cost,
        revenue=revenue,
        margin=margin,
        margin_pct=margin_pc,
    )


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(prices: pd.Series, sel_start: int, sel_dur: int,
                sweet_start: int, client_rate: float, currency: str) -> go.Figure:
    hours  = [f"{h:02d}:00" for h in range(24)]
    colors = []
    for i in range(24):
        in_sel = sel_start <= i < sel_start + sel_dur
        in_sw  = sweet_start <= i < sweet_start + sel_dur
        if in_sel and in_sw:
            colors.append("#1D9E75")
        elif in_sel:
            colors.append("#EF9F27")
        elif in_sw:
            colors.append("rgba(29,158,117,0.3)")
        else:
            colors.append("#D3D1C7")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours, y=prices.round(2),
        marker_color=colors,
        hovertemplate="%{x}: %{y:.1f}p/kWh<extra></extra>",
        name="Wholesale price",
    ))
    fig.add_hline(
        y=client_rate,
        line_dash="dash", line_color="#378ADD", line_width=1.5,
        annotation_text=f"Client rate {client_rate:.1f}p",
        annotation_position="top right",
        annotation_font_color="#378ADD",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="p/kWh", gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(tickangle=-45),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ Settings")

    api_key = st.text_input(
        "ENTSO-E API key",
        type="password",
        help="Free at transparency.entsoe.eu → My Account → Security Token",
    )

    zone_name = st.selectbox("Bidding zone", list(BIDDING_ZONES.keys()), index=0)
    st.session_state.zone = zone_name
    currency = CURRENCY_SYMBOL.get(zone_name, CURRENCY_SYMBOL["default"])

    target_date = st.date_input("Price date", value=date.today())

    if st.button("Fetch day-ahead prices", use_container_width=True):
        with st.spinner("Fetching from ENTSO-E…"):
            result = fetch_entso_prices(api_key, BIDDING_ZONES[zone_name], target_date)
            if result is not None:
                st.session_state.prices    = result
                st.session_state.price_date = target_date
                st.success(f"Loaded {len(result)} hours for {target_date}")
            else:
                st.warning("Using simulated curve instead.")
                st.session_state.prices = simulated_prices()

    if st.session_state.prices is None:
        st.info("No API key? Using a simulated UK price curve.")
        st.session_state.prices = simulated_prices()

    st.divider()
    st.markdown("### Clients")

    with st.expander("Add client"):
        new_name = st.text_input("Name")
        new_rate = st.number_input("Contracted rate (p/kWh)", 5.0, 30.0, 15.0, 0.5)
        new_cons = st.number_input("Daily consumption (MWh)", 10, 5000, 200, 10)
        if st.button("Add", use_container_width=True):
            if new_name:
                st.session_state.clients.append({
                    "name": new_name,
                    "rate_p_kwh": new_rate,
                    "consumption_mwh": new_cons,
                })
                st.rerun()

    for i, c in enumerate(st.session_state.clients):
        cols = st.columns([4, 1])
        cols[0].markdown(f"**{c['name']}**")
        if cols[1].button("✕", key=f"del_{i}"):
            st.session_state.clients.pop(i)
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

prices = st.session_state.prices

st.markdown("## Energy Margin Optimizer")

price_label = (
    f"Live — {st.session_state.price_date} · {zone_name}"
    if st.session_state.price_date else f"Simulated · {zone_name}"
)
st.caption(price_label)

tab_portfolio, tab_client, tab_export = st.tabs(["Portfolio overview", "Per-client detail", "Export"])


# ── Tab 1: Portfolio overview ─────────────────────────────────────────────────

with tab_portfolio:
    if not st.session_state.clients:
        st.info("Add clients in the sidebar to get started.")
    else:
        dur_h = st.slider("Procurement window (hours)", 1, 12, 6)

        rows = []
        for c in st.session_state.clients:
            sw  = best_window(prices, dur_h)
            st = window_stats(prices, sw, dur_h, c["rate_p_kwh"], c["consumption_mwh"])
            rows.append({
                "Client":            c["name"],
                f"Rate (p/kWh)":     c["rate_p_kwh"],
                "Consumption (MWh)": c["consumption_mwh"],
                "Sweet-spot window": f"{sw:02d}:00–{sw+dur_h:02d}:00",
                f"Wholesale cost ({currency})":  round(st["wholesale_cost"]),
                f"Revenue ({currency})":         round(st["revenue"]),
                f"Gross margin ({currency})":    round(st["margin"]),
                "Margin %":          round(st["margin_pct"], 1),
            })

        df = pd.DataFrame(rows)

        # KPI strip
        k1, k2, k3, k4 = st.columns(4)
        total_rev  = df[f"Revenue ({currency})"].sum()
        total_cost = df[f"Wholesale cost ({currency})"].sum()
        total_marg = df[f"Gross margin ({currency})"].sum()
        avg_pct    = round(total_marg / total_rev * 100, 1) if total_rev else 0

        k1.metric("Total revenue",      f"{currency}{total_rev:,.0f}")
        k2.metric("Total wholesale cost", f"{currency}{total_cost:,.0f}")
        k3.metric("Gross margin",        f"{currency}{total_marg:,.0f}")
        k4.metric("Avg margin %",        f"{avg_pct}%")

        st.divider()

        # Styled table
        def colour_margin(val):
            return "color: #1D9E75" if val >= 0 else "color: #E24B4A"

        st.dataframe(
            df.style.applymap(colour_margin, subset=[f"Gross margin ({currency})", "Margin %"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Bar chart — margin by client
        fig_bar = px.bar(
            df, x="Client", y=f"Gross margin ({currency})",
            color="Margin %",
            color_continuous_scale=["#E24B4A", "#EF9F27", "#1D9E75"],
            labels={f"Gross margin ({currency})": f"Margin ({currency})"},
        )
        fig_bar.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 2: Per-client detail ──────────────────────────────────────────────────

with tab_client:
    if not st.session_state.clients:
        st.info("Add clients in the sidebar to get started.")
    else:
        client_names = [c["name"] for c in st.session_state.clients]
        sel_name     = st.selectbox("Select client", client_names)
        client       = next(c for c in st.session_state.clients if c["name"] == sel_name)

        c1, c2, c3 = st.columns(3)
        with c1:
            rate_override = st.number_input(
                "Contracted rate (p/kWh)", 5.0, 30.0,
                float(client["rate_p_kwh"]), 0.5, key="rate_ov"
            )
        with c2:
            cons_override = st.number_input(
                "Daily consumption (MWh)", 10, 5000,
                int(client["consumption_mwh"]), 10, key="cons_ov"
            )
        with c3:
            dur_client = st.slider("Window (hours)", 1, 12, 6, key="dur_cl")

        start_h = st.slider("Start hour", 0, 23, 0, format="%02d:00", key="start_cl")

        sw         = best_window(prices, dur_client)
        sel_stats  = window_stats(prices, start_h, dur_client, rate_override, cons_override)
        sw_stats   = window_stats(prices, sw,      dur_client, rate_override, cons_override)
        extra_cost = sel_stats["wholesale_cost"] - sw_stats["wholesale_cost"]
        margin_gap = sw_stats["margin"] - sel_stats["margin"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Wholesale cost",  f"{currency}{sel_stats['wholesale_cost']:,.0f}")
        m2.metric("Revenue",         f"{currency}{sel_stats['revenue']:,.0f}")
        m3.metric("Gross margin",    f"{currency}{sel_stats['margin']:,.0f}",
                  delta=f"{sel_stats['margin_pct']:.1f}%")
        m4.metric("Optimal window",  f"{sw:02d}:00–{sw+dur_client:02d}:00",
                  delta=f"{currency}{margin_gap:,.0f} uplift" if margin_gap > 0 else None)

        if start_h != sw:
            st.warning(
                f"Shifting to the sweet spot ({sw:02d}:00–{sw+dur_client:02d}:00) saves "
                f"{currency}{extra_cost:,.0f} in procurement and adds {currency}{margin_gap:,.0f} margin."
            )
        else:
            st.success("You're already in the optimal procurement window for this client.")

        fig_detail = build_chart(prices, start_h, dur_client, sw, rate_override, currency)
        st.plotly_chart(fig_detail, use_container_width=True)

        # Save edits back to client list
        for c in st.session_state.clients:
            if c["name"] == sel_name:
                c["rate_p_kwh"]       = rate_override
                c["consumption_mwh"]  = cons_override


# ── Tab 3: Export ─────────────────────────────────────────────────────────────

with tab_export:
    st.markdown("#### Export results")
    dur_exp = st.slider("Window duration for export (hours)", 1, 12, 6, key="dur_exp")

    rows_exp = []
    for c in st.session_state.clients:
        sw  = best_window(prices, dur_exp)
        st2 = window_stats(prices, sw, dur_exp, c["rate_p_kwh"], c["consumption_mwh"])
        rows_exp.append({
            "Client":              c["name"],
            "Rate (p/kWh)":        c["rate_p_kwh"],
            "Consumption (MWh)":   c["consumption_mwh"],
            "Optimal window":      f"{sw:02d}:00–{sw+dur_exp:02d}:00",
            "Avg wholesale (p/kWh)": round(st2["avg_wholesale"], 2),
            f"Wholesale cost ({currency})": round(st2["wholesale_cost"]),
            f"Revenue ({currency})":        round(st2["revenue"]),
            f"Gross margin ({currency})":   round(st2["margin"]),
            "Margin %":            round(st2["margin_pct"], 1),
            "Price date":          str(st.session_state.price_date or "simulated"),
            "Bidding zone":        zone_name,
        })

    df_exp = pd.DataFrame(rows_exp)
    st.dataframe(df_exp, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    df_exp.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue(),
        file_name=f"energy_margins_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### Raw hourly prices")
    prices_df = pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in range(24)],
        "Wholesale (p/kWh)": prices.round(2).values,
    })
    st.dataframe(prices_df, use_container_width=True, hide_index=True)
    price_csv = prices_df.to_csv(index=False)
    st.download_button(
        "Download price curve CSV",
        data=price_csv,
        file_name=f"prices_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
