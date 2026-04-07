"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       Simulador de Modulación Digital M-aria (M-QAM y M-PSK)                 ║
║       Autor: wendy cardenas villalobos                              ║
║       Stack: Python · Streamlit · NumPy · Plotly                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import math  # Usamos math nativo en lugar de SciPy para evitar crashes de CPU

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DE LA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador M-QAM / M-PSK",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

  :root {
    --bg:        #0a0e1a;
    --panel:     #111827;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #ff6b35;
    --accent3:   #7cf36a;
    --text:      #ccd6f6;
    --muted:     #6b7fa3;
    --card-bg:   #0d1526;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Barlow', sans-serif;
  }

  [data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .main-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 0;
  }
  .sub-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    margin-top: 2px;
  }
  .header-bar {
    border-left: 4px solid var(--accent);
    padding-left: 16px;
    margin-bottom: 28px;
  }

  .kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }
  .kpi-card.green  { border-top-color: var(--accent3); }
  .kpi-card.orange { border-top-color: var(--accent2); }

  .kpi-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .kpi-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
  }
  .kpi-value.green  { color: var(--accent3); }
  .kpi-value.orange { color: var(--accent2); }
  .kpi-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
  }

  .badge {
    display: inline-block;
    background: #0a1929;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 3px 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    margin-right: 8px;
  }

  .js-plotly-plot { background: transparent !important; }
  hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1: GENERACIÓN DE CONSTELACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_qam_constellation(M: int) -> np.ndarray:
    m = int(np.sqrt(M))
    axis = np.arange(-(m - 1), m, 2, dtype=float)
    I, Q = np.meshgrid(axis, axis)
    symbols = (I + 1j * Q).flatten()
    energy = np.mean(np.abs(symbols) ** 2)
    return symbols / np.sqrt(energy)

def generate_psk_constellation(M: int) -> np.ndarray:
    k = np.arange(M)
    angles = 2 * np.pi * k / M + np.pi / M
    return np.exp(1j * angles)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2: SIMULACIÓN MONTE CARLO (BATCHING EXTREMO)
# ═══════════════════════════════════════════════════════════════════════════════

def add_awgn_noise(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    snr_lin = 10 ** (snr_db / 10.0)
    Es = np.mean(np.abs(symbols) ** 2)
    sigma2 = Es / (2.0 * snr_lin)
    sigma = np.sqrt(sigma2)
    noise = sigma * (np.random.randn(*symbols.shape) +
                     1j * np.random.randn(*symbols.shape))
    return symbols + noise

def simulate_ber(
    constellation: np.ndarray,
    n_symbols: int,
    snr_db: float,
    mod_type: str,
    M: int,
) -> tuple[float, float, int, int]:
    
    k = int(np.log2(M))
    tx_indices = np.random.randint(0, M, size=n_symbols)
    tx_symbols = constellation[tx_indices]
    rx_symbols = add_awgn_noise(tx_symbols, snr_db)

    rx_indices = np.zeros(n_symbols, dtype=int)
    batch_size = 100  # Lote pequeño para uso cero de RAM
    
    for i in range(0, n_symbols, batch_size):
        batch_rx = rx_symbols[i : i + batch_size]
        distances = np.abs(batch_rx[:, np.newaxis] - constellation[np.newaxis, :]) ** 2
        rx_indices[i : i + batch_size] = np.argmin(distances, axis=1)
        del distances

    symbol_errors = np.sum(rx_indices != tx_indices)
    SER_sim = symbol_errors / n_symbols
    BER_sim = SER_sim / k

    return BER_sim, SER_sim, int(BER_sim * n_symbols * k), symbol_errors


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3: FÓRMULAS TEÓRICAS DE BER / SER (SIN SCIPY)
# ═══════════════════════════════════════════════════════════════════════════════

def ebn0_to_snr_db(ebn0_db: float, M: int) -> float:
    k = np.log2(M)
    return ebn0_db + 10 * np.log10(k)

def ber_theory_qam(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    k = np.log2(M)
    ebn0_lin = 10 ** (ebn0_db / 10.0)
    arg = np.sqrt(3 * k * ebn0_lin / (2 * (M - 1)))
    coeff = (4.0 / k) * (1.0 - 1.0 / np.sqrt(M))
    
    # Vectorización nativa segura: iteramos con list comprehension y math nativo
    erfc_vals = np.array([math.erfc(x) for x in arg])
    ber = coeff * 0.5 * erfc_vals
    return np.clip(ber, 1e-10, 1.0)

def ber_theory_psk(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    k = np.log2(M)
    ebn0_lin = 10 ** (ebn0_db / 10.0)
    if M == 2:
        arg_bpsk = np.sqrt(ebn0_lin)
        erfc_vals = np.array([math.erfc(x) for x in arg_bpsk])
        return 0.5 * erfc_vals
        
    arg = np.sqrt(k * ebn0_lin) * np.sin(np.pi / M)
    erfc_vals = np.array([math.erfc(x) for x in arg])
    ber = (2.0 / k) * 0.5 * erfc_vals
    return np.clip(ber, 1e-10, 1.0)

def spectral_efficiency(M: int) -> float:
    return np.log2(M)

def energy_efficiency_required(M: int, mod_type: str) -> float:
    ebn0_range = np.linspace(-5, 60, 5000)
    if mod_type == "QAM":
        ber_curve = ber_theory_qam(ebn0_range, M)
    else:
        ber_curve = ber_theory_psk(ebn0_range, M)
        
    idx = np.where(ber_curve <= 1e-5)[0]
    if len(idx) == 0:
        return float("nan")
    return ebn0_range[idx[0]]


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4: GRÁFICAS PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0d1526",
    font=dict(family="Share Tech Mono, monospace", color="#ccd6f6", size=11),
    margin=dict(l=50, r=20, t=50, b=50),
    xaxis=dict(gridcolor="#1e2d45", gridwidth=1, zerolinecolor="#2a3f5f", zerolinewidth=1, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d45", gridwidth=1, zerolinecolor="#2a3f5f", zerolinewidth=1, tickfont=dict(size=10)),
)

def plot_constellation(constellation, rx_symbols, tx_indices, mod_type, M, snr_db):
    fig = go.Figure()
    if rx_symbols is not None and len(rx_symbols) > 0:
        n_plot = min(4000, len(rx_symbols))
        idx_plot = np.random.choice(len(rx_symbols), n_plot, replace=False)
        fig.add_trace(go.Scatter(
            x=rx_symbols[idx_plot].real, y=rx_symbols[idx_plot].imag,
            mode="markers", marker=dict(size=3, color="rgba(0, 212, 255, 0.18)", line=dict(width=0)),
            name="Rx (con ruido)", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=constellation.real, y=constellation.imag,
        mode="markers", marker=dict(size=10, color="#ff6b35", symbol="circle", line=dict(color="#ffffff", width=1.2)),
        name="Constelación ideal", hovertemplate="I=%{x:.3f}<br>Q=%{y:.3f}<extra></extra>",
    ))

    if mod_type == "PSK":
        theta = np.linspace(0, 2 * np.pi, 300)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode="lines",
            line=dict(color="#1e2d45", width=1, dash="dot"), showlegend=False, hoverinfo="skip",
        ))

    layout = PLOT_LAYOUT.copy()
    layout.update(dict(
        title=dict(text=f"<b>Constelación {M}-{mod_type}</b>  |  SNR = {snr_db:.1f} dB", font=dict(size=13, color="#00d4ff"), x=0.02),
        xaxis_title="In-Phase (I)", yaxis_title="Quadrature (Q)", xaxis=dict(**PLOT_LAYOUT["xaxis"], scaleanchor="y"),
        legend=dict(bgcolor="#0d1526", bordercolor="#1e2d45", borderwidth=1, font=dict(size=10), x=0.01, y=0.99),
        height=500,
    ))
    fig.update_layout(**layout)
    return fig

def plot_ber_curve(mod_type, M, ebn0_current, ber_simulated):
    ebn0_range = np.linspace(-2, 35, 800)
    fig = go.Figure()
    colors = {4: "#7cf36a", 16: "#00d4ff", 64: "#bf7aff", 256: "#ff6b35", 1024: "#ffd700", 4096: "#ff4d8d", 16384: "#a0c4ff"}

    if mod_type == "QAM":
        ber_th = ber_theory_qam(ebn0_range, M)
    else:
        ber_th = ber_theory_psk(ebn0_range, M)

    color = colors.get(M, "#00d4ff")
    fig.add_trace(go.Scatter(
        x=ebn0_range, y=ber_th, mode="lines", name=f"{M}-{mod_type} (teórico)",
        line=dict(color=color, width=2.5), hovertemplate="Eb/N0=%{x:.1f} dB<br>BER=%{y:.2e}<extra></extra>",
    ))

    ref_orders = [o for o in [4, 16, 64, 256, 1024] if o != M]
    for ref_M in ref_orders:
        ber_ref = ber_theory_qam(ebn0_range, ref_M) if mod_type == "QAM" else ber_theory_psk(ebn0_range, ref_M)
        fig.add_trace(go.Scatter(
            x=ebn0_range, y=ber_ref, mode="lines", name=f"{ref_M}-{mod_type}",
            line=dict(color=colors.get(ref_M, "#888"), width=1, dash="dot"), opacity=0.35, hoverinfo="skip",
        ))

    if ber_simulated > 0:
        fig.add_trace(go.Scatter(
            x=[ebn0_current], y=[ber_simulated], mode="markers", name="Punto actual (sim.)",
            marker=dict(size=14, color="#ffffff", symbol="star", line=dict(color=color, width=2)),
            hovertemplate=(f"Eb/N0 = {ebn0_current:.1f} dB<br>BER sim. = {ber_simulated:.2e}<extra></extra>"),
        ))
        fig.add_shape(type="line", x0=ebn0_current, x1=ebn0_current, y0=1e-7, y1=ber_simulated, line=dict(color=color, width=1, dash="dash"))

    layout = PLOT_LAYOUT.copy()
    layout.update(dict(
        title=dict(text=f"<b>BER vs Eb/N0</b>  |  Curva Waterfall {M}-{mod_type}", font=dict(size=13, color="#00d4ff"), x=0.02),
        xaxis_title="Eb/N0 (dB)", yaxis_title="Bit Error Rate (BER)",
        yaxis=dict(**PLOT_LAYOUT["yaxis"], type="log", range=[-7, 0], tickformat=".0e"),
        legend=dict(bgcolor="#0d1526", bordercolor="#1e2d45", borderwidth=1, font=dict(size=10), x=0.01, y=0.01),
        height=500,
    ))
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: HELPERS DE RENDERIZADO UI
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str = "", color: str = "blue") -> str:
    cls = {"blue": "", "green": "green", "orange": "orange"}.get(color, "")
    val_cls = f"kpi-value {cls}".strip()
    return f"""<div class="kpi-card {cls}"><div class="kpi-label">{label}</div><div class="{val_cls}">{value}</div><div class="kpi-sub">{sub}</div></div>"""

def format_ber(ber: float) -> str:
    if ber <= 0 or np.isnan(ber): return "< 10⁻⁷"
    exp = int(np.floor(np.log10(ber)))
    mant = ber / (10 ** exp)
    return f"{mant:.2f} × 10<sup>{exp}</sup>"

def quality_label(ber: float) -> tuple[str, str]:
    if np.isnan(ber) or ber <= 0: return "EXCELENTE", "green"
    if ber < 1e-5: return "BUENO", "green"
    if ber < 1e-3: return "ACEPTABLE", "blue"
    if ber < 1e-2: return "MARGINAL", "orange"
    return "DEGRADADO", "orange"


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6: APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""<div class="header-bar"><div class="main-header"> Simulador M-aria</div><div class="sub-header">M-QAM &amp; M-PSK · AWGN Channel · Monte Carlo BER Analysis</div></div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""<div style='font-family: "Barlow Condensed", sans-serif; font-size:1.1rem; font-weight:700; color:#00d4ff; letter-spacing:0.1em; text-transform:uppercase; border-bottom:1px solid #1e2d45; padding-bottom:8px; margin-bottom:16px;'>⚙ Panel de Control</div>""", unsafe_allow_html=True)

        mod_type = st.selectbox("TIPO DE MODULACIÓN", options=["QAM", "PSK"], index=0)
        M_options = [4, 16, 64, 256, 1024, 4096, 16384]
        M = st.selectbox("ORDEN DE MODULACIÓN (M)", options=M_options, index=1, format_func=lambda x: f"{x}-{mod_type}  ({int(np.log2(x))} bits/símbolo)")
        k_bits = int(np.log2(M))

        st.markdown("---")
        st.markdown("""<div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; color:#6b7fa3; letter-spacing:0.1em; margin-bottom:8px;'>PARÁMETROS DE RUIDO / CANAL</div>""", unsafe_allow_html=True)
        snr_db = st.slider("SNR (dB)  — Por símbolo", min_value=0.0, max_value=40.0, value=15.0, step=0.5)
        ebn0_db = snr_db - 10 * np.log10(k_bits)
        st.markdown(f"<div style='font-family:\"Share Tech Mono\",monospace; font-size:0.72rem; color:#00d4ff;'>Eb/N0 equivalente: <b>{ebn0_db:.2f} dB</b></div>", unsafe_allow_html=True)

        st.markdown("---")
        max_sym = 50_000 if M <= 256 else 20_000 if M <= 4096 else 10_000
        n_symbols = st.number_input("SÍMBOLOS A SIMULAR (Monte Carlo)", min_value=1_000, max_value=max_sym, value=min(10_000, max_sym), step=1_000)

        st.markdown("---")
        run_sim = st.button("▶  EJECUTAR SIMULACIÓN", use_container_width=True)

        st.markdown(f"""<div style='margin-top:16px; font-family:"Share Tech Mono",monospace; font-size:0.68rem; color:#6b7fa3; line-height:1.6;'><div class="badge">k</div>{k_bits} bits/símbolo<br><div class="badge">M</div>{M} símbolos<br><div class="badge">η</div>{k_bits:.0f} bits/s/Hz (Nyquist)</div>""", unsafe_allow_html=True)

    if mod_type == "QAM":
        constellation = generate_qam_constellation(M)
    else:
        constellation = generate_psk_constellation(M)

    eta       = spectral_efficiency(M)
    ber_th_pt = (ber_theory_qam if mod_type == "QAM" else ber_theory_psk)(np.array([ebn0_db]), M)[0]
    e_req     = energy_efficiency_required(M, mod_type)

    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None

    if run_sim:
        with st.spinner("Simulando canal AWGN..."):
            ber_sim, ser_sim, bit_errors, sym_errors = simulate_ber(constellation, n_symbols, snr_db, mod_type, M)
            tx_indices = np.random.randint(0, M, size=n_symbols)
            tx_symbols = constellation[tx_indices]
            rx_symbols = add_awgn_noise(tx_symbols, snr_db)

            st.session_state.sim_results = {
                "rx_symbols": rx_symbols, "tx_indices": tx_indices, "ber_sim": ber_sim,
                "ser_sim": ser_sim, "bit_errors": bit_errors, "sym_errors": sym_errors,
                "snr_db": snr_db, "ebn0_db": ebn0_db, "M": M, "mod_type": mod_type, "n_symbols": n_symbols,
            }

    sim = st.session_state.sim_results
    sim_valid = (sim is not None and sim["M"] == M and sim["mod_type"] == mod_type)
    ber_sim_val = sim["ber_sim"] if sim_valid else float("nan")
    ser_sim_val = sim["ser_sim"] if sim_valid else float("nan")

    st.markdown('<div class="section-title"> Métricas de Desempeño</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1: st.markdown(kpi_card("Eficiencia Espectral", f"{eta:.0f} b/s/Hz", f"log₂({M}) = {eta:.0f} | Nyquist", "blue"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_card("BER Teórica", format_ber(ber_th_pt), f"Eb/N0 = {ebn0_db:.1f} dB", "blue"), unsafe_allow_html=True)
    with col3:
        if sim_valid and ber_sim_val > 0: st.markdown(kpi_card("BER Simulada", format_ber(ber_sim_val), quality_label(ber_sim_val)[0], quality_label(ber_sim_val)[1]), unsafe_allow_html=True)
        else: st.markdown(kpi_card("BER Simulada", "—", "Sin simular", "blue"), unsafe_allow_html=True)
    with col4:
        if sim_valid: st.markdown(kpi_card("SER Simulada", format_ber(ser_sim_val), f"{sim['sym_errors']:,} errores de {sim['n_symbols']:,}", "orange"), unsafe_allow_html=True)
        else: st.markdown(kpi_card("SER Simulada", "—", "Ejecuta la simulación", "orange"), unsafe_allow_html=True)
    with col5:
        if np.isnan(e_req): st.markdown(kpi_card("Eb/N0 Req. (BER=10⁻⁵)", "> 35 dB", "No viable con curva disponible", "orange"), unsafe_allow_html=True)
        else: st.markdown(kpi_card("Eb/N0 Req. (BER=10⁻⁵)", f"{e_req:.1f} dB", f"Eb/N0 mín. para BER < 10⁻⁵", "green" if e_req < 25 else "orange"), unsafe_allow_html=True)

    st.markdown("")
    info_cols = st.columns(4)
    info_data = [
        ("Bits por símbolo (k)", f"{k_bits}", f"log₂({M})"),
        ("Tasa relativa de datos", f"×{k_bits} vs BPSK", "Incremento de throughput"),
        ("SNR actual", f"{snr_db:.1f} dB", f"≡ {10**(snr_db/10):.1f} lineal"),
        ("Símbolos en constelación", f"{M:,}", f"{'Cuadrada' if mod_type=='QAM' else 'Circular'} {int(np.sqrt(M))}×{int(np.sqrt(M))} " if mod_type=="QAM" else f"Radio unitario"),
    ]
    for col, (lbl, val, sub) in zip(info_cols, info_data):
        with col: st.markdown(kpi_card(lbl, val, sub, "blue"), unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Visualización</div>', unsafe_allow_html=True)
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        rx_plot = sim["rx_symbols"] if sim_valid else None
        tx_idx_plot = sim["tx_indices"] if sim_valid else None
        fig_const = plot_constellation(constellation, rx_plot, tx_idx_plot, mod_type, M, snr_db)
        st.plotly_chart(fig_const, use_container_width=True, config={"displayModeBar": False})
        if not sim_valid: st.info(" Presiona **EJECUTAR SIMULACIÓN** para ver la nube de ruido AWGN.")

    with plot_col2:
        fig_ber = plot_ber_curve(mod_type, M, ebn0_db, ber_sim_val if sim_valid else 0.0)
        st.plotly_chart(fig_ber, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title"> Tabla de Referencia Teórica</div>', unsafe_allow_html=True)
    ref_snr = 20.0
    ref_ebn0 = ref_snr - 10 * np.log10(np.log2(np.array(M_options)))
    summary_rows = []
    for i, m in enumerate(M_options):
        ki = int(np.log2(m))
        eb = ref_ebn0[i]
        ber_r = ber_theory_qam(np.array([eb]), m)[0] if mod_type == "QAM" else ber_theory_psk(np.array([eb]), m)[0]
        e_r = energy_efficiency_required(m, mod_type)
        summary_rows.append({
            "Modulación": f"{m}-{mod_type}", "k (bits/símbolo)": ki, "η (bits/s/Hz)": ki,
            f"BER teórica @ SNR={ref_snr:.0f}dB": f"{ber_r:.2e}",
            "Eb/N0 req. (BER=10⁻⁵)": f"{e_r:.1f} dB" if not np.isnan(e_r) else "> 35 dB", "Estado": " ACTIVO" if m == M else "",
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""<div style='text-align:center; font-family:"Share Tech Mono",monospace; font-size:0.68rem; color:#6b7fa3; padding: 8px 0;'>Simulador M-QAM / M-PSK · Canal AWGN · Monte Carlo BER · NumPy · Plotly · Streamlit<br>Fórmulas: Proakis &amp; Salehi — "Digital Communications", 5ª ed.</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()