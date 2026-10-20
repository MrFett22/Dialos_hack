"""Figuras Plotly de la página (reciben datos ya calculados; no leen archivos)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ENV_COLORS = {"No Altitude": "#4c78a8", "Medium": "#f2a541", "Extreme": "#d1495b",
              "Medium Altitude": "#f2a541", "Extreme Altitude": "#d1495b"}
SHORT = {"No Altitude": "No Altitude", "Medium Altitude": "Medium", "Extreme Altitude": "Extreme"}
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1d2433")


def _layout(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), font=FONT, plot_bgcolor="white",
                      paper_bgcolor="white", **kw)
    fig.update_xaxes(gridcolor="#eef1f6", zerolinecolor="#c9cfdb")
    fig.update_yaxes(gridcolor="#eef1f6", zerolinecolor="#c9cfdb")
    return fig


def effects_bars(eff: dict, key: str, title: str, scale: float = 100, unit: str = "pp") -> go.Figure:
    e = eff[key]
    vals = [e["medium"] * scale, e["extreme"] * scale]
    lo = [e["medium_ic95"][0] * scale, e["extreme_ic95"][0] * scale]
    hi = [e["medium_ic95"][1] * scale, e["extreme_ic95"][1] * scale]
    fig = go.Figure(go.Bar(x=["Medium", "Extreme"], y=vals, marker_color=[ENV_COLORS["Medium"], ENV_COLORS["Extreme"]],
                           error_y=dict(type="data", symmetric=False, array=np.subtract(hi, vals), arrayminus=np.subtract(vals, lo)),
                           text=[f"{v:+.1f} {unit}" if unit == "pp" else f"{v:+.3f}" for v in vals], textposition="inside",
                           insidetextanchor="middle", textfont=dict(color="white")))
    fig.add_hline(y=0, line_color="#8892a6")
    return _layout(fig, 300, title=dict(text=title, font=dict(size=14)), yaxis_title=f"diferencia vs. No Altitude ({unit})", showlegend=False)


def consistency_strip(detail: dict, cats: list[str]) -> go.Figure:
    """β por estadio-temporada (anónimo) de cada variable; la línea vertical es 0."""
    labels = {"whiff por swing": "Whiff/Swing (pp)", "ponche por turno": "K/PA (pp)", "bola en juego por swing": "BIP/Swing (pp)",
              "run value RE24 por bola en juego": "RV por BIP (×100)"}
    fig = go.Figure()
    for cat in cats:
        for i, (var, lab) in enumerate(labels.items()):
            betas = list(detail[cat][var]["betas"].values())
            names = list(detail[cat][var]["betas"].keys())
            fig.add_trace(go.Scatter(x=[100 * b for b in betas], y=[f"{lab} · {SHORT[cat]}"] * len(betas), mode="markers", name=SHORT[cat],
                                     marker=dict(color=ENV_COLORS[cat], size=9, opacity=0.75, line=dict(width=0.5, color="white")),
                                     text=names, hovertemplate="%{text}<br>%{x:+.2f}<extra></extra>", showlegend=(i == 0)))
    fig.add_vline(x=0, line_color="#8892a6")
    return _layout(fig, 420, title=dict(text="Cada punto = un estadio anónimo en una temporada", font=dict(size=14)),
                   xaxis_title="diferencia contra No Altitude (efectos fijos de pitcher y bateador)",
                   legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


def damage_panels(cat: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    order = ["No Altitude", "Medium", "Extreme"]
    c = cat.set_index("ambiente").loc[order]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Probabilidad de bola en juego<br>por pitch", "Daño esperado si hay bola en juego<br>(carreras RE24)",
                                                         "Daño de contacto por 100 pitches<br>= P(BIP) × E[RV | BIP]"))
    for j, (col, k, fmt) in enumerate((("p_bip_por_pitch", 100, "{:.1f}%"), ("rv_por_bip", 1, "{:+.3f}"), ("dano_bip_por_pitch", 100, "{:.2f}")), start=1):
        y = c[col] * k
        fig.add_trace(go.Bar(x=order, y=y, marker_color=[ENV_COLORS[o] for o in order], text=[fmt.format(v) for v in y],
                             textposition="outside", error_y=dict(type="data", symmetric=False, array=c[f"{col}_hi"] * k - y,
                                                                  arrayminus=y - c[f"{col}_lo"] * k), showlegend=False), row=1, col=j)
    fig.update_yaxes(rangemode="tozero")
    return _layout(fig, 360)


def decomposition_donut(frac_damage: float, frac_prob: float, frac_inter: float) -> go.Figure:
    fig = go.Figure(go.Pie(labels=["cada bola en juego hace más daño", "hay más bolas en juego", "interacción"],
                           values=[frac_damage, frac_prob, frac_inter], hole=0.62, sort=False,
                           marker=dict(colors=["#d1495b", "#4c78a8", "#c9cfdb"]), textinfo="percent"))
    fig.update_layout(annotations=[dict(text="aumento del<br>daño en Extreme", showarrow=False, font=dict(size=13))])
    return _layout(fig, 320, legend=dict(orientation="h", y=-0.1))


def stadium_scatter(st: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in ["No Altitude", "Medium", "Extreme"]:
        g = st[st["categoria"] == c]
        fig.add_trace(go.Scatter(x=100 * g["p_bip_por_pitch"], y=g["rv_por_bip"], mode="markers", name=c, text=g["ambiente"],
                                 marker=dict(size=8 + g["juegos"] / 12, color=ENV_COLORS[c], opacity=0.8, line=dict(width=0.6, color="white")),
                                 hovertemplate="%{text}<br>P(BIP) %{x:.1f}%<br>RV/BIP %{y:+.3f}<extra></extra>"))
    return _layout(fig, 360, title=dict(text="Estadios anónimos: cuánto contacto y cuánto cuesta", font=dict(size=14)),
                   xaxis_title="P(bola en juego) por pitch (%)", yaxis_title="carreras por bola en juego",
                   legend=dict(orientation="h", y=-0.22, x=0))


def k_vs_bip(lw: pd.DataFrame) -> go.Figure:
    order = ["No Altitude", "Medium", "Extreme"]
    d = lw.set_index("ambiente").loc[order]
    y = -d["K_menos_BIP"]
    fig = go.Figure(go.Bar(x=order, y=y, marker_color=[ENV_COLORS[o] for o in order], text=[f"{v:.3f}" for v in y], textposition="outside",
                           error_y=dict(type="data", symmetric=False, array=-d["K_menos_BIP_lo"] - y, arrayminus=y + d["K_menos_BIP_hi"])))
    return _layout(fig, 320, title=dict(text="Carreras que ahorra un ponche en lugar de una bola en juego", font=dict(size=14)),
                   yaxis_title="carreras por turno", showlegend=False)


def characteristic_forest(cv: pd.DataFrame, labels: dict) -> go.Figure:
    fig = go.Figure()
    for k, c in enumerate(["No Altitude", "Medium", "Extreme"]):
        fig.add_trace(go.Scatter(x=cv[f"pend_{c}"], y=[labels[m] for m in cv["caracteristica"]], mode="markers", name=c,
                                 marker=dict(color=ENV_COLORS[c], size=10),
                                 error_x=dict(type="data", symmetric=False, array=cv[f"pend_{c}_hi"] - cv[f"pend_{c}"],
                                              arrayminus=cv[f"pend_{c}"] - cv[f"pend_{c}_lo"], thickness=1.2), offsetgroup=c))
    fig.add_vline(x=0, line_color="#8892a6")
    return _layout(fig, 430, scattermode="group", xaxis_title="carreras evitadas por 100 turnos por +1 desviación estándar (IC 95%)",
                   legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))


def bars_simple(x, y, title, colors=None, fmt="{:.2f}", height=300, yaxis_title=""):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors or "#1f3a63", text=[fmt.format(v) for v in y], textposition="outside"))
    return _layout(fig, height, title=dict(text=title, font=dict(size=14)), yaxis_title=yaxis_title, showlegend=False)


def heatmap(grid: pd.DataFrame, title: str) -> go.Figure:
    piv = grid.pivot(index="PlateLocHeight", columns="loc_in", values="carreras_por_100")
    vmax = float(np.nanpercentile(np.abs(piv.to_numpy()), 98)) if piv.notna().any().any() else 1
    fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=piv.columns, y=piv.index, colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                               colorbar=dict(title="carreras/100<br>vs. típica"), hovertemplate="lado %{x:.1f} · altura %{y:.1f}<br>%{z:+.1f}<extra></extra>"))
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="black", width=2))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _layout(fig, 430, title=dict(text=title, font=dict(size=14)), xaxis_title="afuera ←  → adentro (pies)", yaxis_title="altura (pies)")


def sankey(probs: dict, labels: dict) -> go.Figure:
    take = probs["ball"] + probs["called_strike"] + probs["hbp"]
    contact = probs["foul"] + sum(probs[e] for e in ["out", "single", "double", "triple", "home_run"])
    bip = contact - probs["foul"]
    nodes = ["Lanzamiento", "Take", "Swing", "Contacto", "Bola en juego"] + [labels[e] for e in
            ["ball", "called_strike", "hbp", "whiff", "foul", "out", "single", "double", "triple", "home_run"]]
    idx = {n: i for i, n in enumerate(nodes)}
    links = [("Lanzamiento", "Take", take), ("Lanzamiento", "Swing", 1 - take), ("Swing", labels["whiff"], probs["whiff"]),
             ("Swing", "Contacto", contact), ("Contacto", labels["foul"], probs["foul"]), ("Contacto", "Bola en juego", bip)]
    links += [("Take", labels[e], probs[e]) for e in ["ball", "called_strike", "hbp"]]
    links += [("Bola en juego", labels[e], probs[e]) for e in ["out", "single", "double", "triple", "home_run"]]
    good = {labels[e] for e in ["called_strike", "whiff", "out", "foul"]}
    colors = ["#1f3a63", "#8892a6", "#1f3a63", "#1f3a63", "#d1495b"] + ["#4c78a8" if n in good else "#d1495b" for n in nodes[5:]]
    fig = go.Figure(go.Sankey(node=dict(label=[f"{n}" for n in nodes], color=colors, pad=14, thickness=16),
                              link=dict(source=[idx[a] for a, _, _ in links], target=[idx[b] for _, b, _ in links],
                                        value=[max(v, 1e-6) for _, _, v in links],
                                        customdata=[f"{v:.1%}" for _, _, v in links], hovertemplate="%{customdata}<extra></extra>")))
    return _layout(fig, 380)


def prob_bars(probs: dict, labels: dict, base: dict | None = None) -> go.Figure:
    evs = list(probs)
    fig = go.Figure(go.Bar(y=[labels[e] for e in evs], x=[probs[e] for e in evs], orientation="h", name="este lanzamiento",
                           marker_color="#1f3a63", text=[f"{probs[e]:.1%}" for e in evs], textposition="outside"))
    if base:
        fig.add_trace(go.Scatter(y=[labels[e] for e in evs], x=[base[e] for e in evs], mode="markers", name="mediano del tipo",
                                 marker=dict(symbol="line-ns-open", size=18, color="#d1495b", line=dict(width=3))))
    fig.update_yaxes(autorange="reversed")
    top = max(list(probs.values()) + (list(base.values()) if base else []))
    fig.update_xaxes(tickformat=".0%", range=[0, top * 1.22])
    return _layout(fig, 400, legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))


def zone_picker(surface: pd.DataFrame, px: float, pz: float, b_side: str) -> go.Figure:
    """Cuadros de Location+ seleccionables (celdas con pocos pitches reales en gris claro)."""
    s = surface.copy()
    ok = s["pitches_en_celda"] >= 20
    color = np.where(ok, s["location_plus"], np.nan)
    fig = go.Figure(go.Scatter(x=s["plate_x"], y=s["plate_z"], mode="markers", name="zona",
                               marker=dict(symbol="square", size=17, color=s["location_plus"].where(ok, 100.0), colorscale="RdBu", cmid=100,
                                           cmin=80, cmax=120, opacity=np.where(ok, 1.0, 0.18).tolist(),
                                           colorbar=dict(title="Location+", orientation="h", y=-0.3, thickness=10, len=0.8)),
                               customdata=np.c_[s["location_plus"], s["pitches_en_celda"]],
                               hovertemplate="lado %{x:.1f} · altura %{y:.1f}<br>Location+ %{customdata[0]:.0f}<br>"
                                             "pitches reales en la celda: %{customdata[1]}<extra></extra>"))
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="black", width=2))
    fig.add_shape(type="circle", x0=px - 0.12, x1=px + 0.12, y0=pz - 0.12, y1=pz + 0.12, line=dict(color="black", width=3), fillcolor="white")
    fig.update_yaxes(range=[0.45, 4.55], fixedrange=True)
    fig.update_xaxes(range=[-1.95, 1.95], fixedrange=True)
    who = "derecho (lado +)" if b_side == "Right" else "zurdo (lado −)"
    return _layout(fig, 560, showlegend=False, title=dict(text=f"Vista del catcher · bateador {who}", font=dict(size=13)),
                   xaxis_title="PlateLocSide (pies)", yaxis_title="PlateLocHeight (pies)", clickmode="event+select", dragmode="select")


def env_compare(by_env: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    order = list(by_env["ambiente"])
    labs = [SHORT[o] for o in order]
    cols = [ENV_COLORS[o] for o in order]
    fig = make_subplots(rows=1, cols=4, subplot_titles=("E[Δ] (carreras/100 pitches)<br>menor = mejor", "P(whiff | swing)", "P(bola en juego)",
                                                         "IVB (pulg.)"))
    for j, (col, fmt, k) in enumerate((("xrv_per_100", "{:+.2f}", 1), ("p_whiff_swing_model", "{:.1%}", 1), ("p_bip", "{:.1%}", 1), ("ivb", "{:.1f}", 1)), start=1):
        y = by_env[col] * k
        fig.add_trace(go.Bar(x=labs, y=y, marker_color=cols, text=[fmt.format(v) for v in y], textposition="outside", showlegend=False), row=1, col=j)
    return _layout(fig, 340)


def trajectories(paths: pd.DataFrame, hand_sign: float = 1.0) -> go.Figure:
    """Diferencia de la trayectoria contra No Altitude (pulgadas) con la misma salida, y punto de llegada (vista del catcher)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, column_widths=[0.38, 0.38, 0.24],
                        subplot_titles=("Altura vs. No Altitude (pulg.)", "Lado vs. No Altitude (pulg.)", "Dónde cruza el plato"))
    for cat, g in paths.groupby("ambiente", sort=False):
        col = ENV_COLORS[cat]
        fig.add_trace(go.Scatter(x=g["distancia"], y=g["dif_altura_pulg"], mode="lines", name=SHORT[cat], line=dict(color=col, width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["distancia"], y=g["dif_lado_pulg"], mode="lines", line=dict(color=col, width=3), showlegend=False), row=1, col=2)
        last = g.iloc[-1]
        fig.add_trace(go.Scatter(x=[last["lado"]], y=[last["altura"]], mode="markers", marker=dict(color=col, size=14, line=dict(color="white", width=1)),
                                 showlegend=False, hovertemplate=f"{SHORT[cat]}<br>lado %{{x:.2f}} · altura %{{y:.2f}}<extra></extra>"), row=1, col=3)
    fig.update_xaxes(autorange="reversed", title_text="distancia al plato (pies)", row=1, col=1)
    fig.update_xaxes(autorange="reversed", title_text="distancia al plato (pies)", row=1, col=2)
    fig.update_xaxes(title_text="lado (pies)", row=1, col=3)
    fig.update_yaxes(title_text="altura (pies)", row=1, col=3)
    return _layout(fig, 380, legend=dict(orientation="h", y=1.18, x=1, xanchor="right"))


def ranking_forest(top: pd.DataFrame) -> go.Figure:
    t = top.iloc[::-1]
    fig = go.Figure(go.Scatter(x=t["aepp"], y=[f"#{int(r)} {p}" for r, p in zip(t["rank"], t["pitcher_anon_id"])], mode="markers",
                               marker=dict(color="#d1495b", size=10),
                               error_x=dict(type="data", symmetric=False, array=t["aepp_hi80"] - t["aepp"], arrayminus=t["aepp"] - t["aepp_lo80"],
                                            color="#8892a6"),
                               hovertemplate="%{y}<br>AEPP %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#8892a6")
    return _layout(fig, max(320, 22 * len(t) + 80), xaxis_title="AEPP: carreras evitadas por 100 turnos vs. pitcher promedio en altitud (intervalo 80%)")


def mix_donut(mix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(labels=mix["AutoPitchType"], values=mix["uso"], hole=0.55, sort=False, textinfo="percent", textposition="inside"))
    return _layout(fig, 320, title=dict(text="Pitch mix", font=dict(size=14)), legend=dict(orientation="h", y=-0.1))


def pitcher_type_bars(mix: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    m = mix[mix["muestra_suficiente"]]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Stuff+ por tipo<br>(100 = promedio del tipo)", "Location+ por tipo<br>(100 = promedio del tipo)",
                                                         "Whiff por swing"))
    for j, (col, fmt) in enumerate((("stuff_plus_tipo", "{:.0f}"), ("location_plus_tipo", "{:.0f}"), ("whiff_swing", "{:.0%}")), start=1):
        fig.add_trace(go.Bar(x=m["AutoPitchType"], y=m[col], marker_color="#1f3a63", text=[fmt.format(v) for v in m[col]], textposition="outside",
                             showlegend=False), row=1, col=j)
    for j in (1, 2):
        fig.add_hline(y=100, line_dash="dot", line_color="#8892a6", row=1, col=j)
    return _layout(fig, 320)


def sensitivity_scatter(s: pd.DataFrame, r: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=s["s_mod_rel"], y=s["s_emp"], mode="markers", marker=dict(color="#8892a6", size=6, opacity=0.6),
                               error_y=dict(type="data", array=1.645 * s["s_emp_se"], color="#e3e7ef", thickness=1),
                               text=s["pitcher_anon_id"], hovertemplate="%{text}<br>modelo %{x:+.2f} · observado %{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#8892a6")
    fig.add_vline(x=0, line_color="#8892a6")
    return _layout(fig, 380, title=dict(text=f"Sensibilidad modelada vs. observada: r = {r:+.2f}", font=dict(size=14)),
                   xaxis_title="según el modelo físico (carreras/100, relativa a la liga)", yaxis_title="observada (carreras/100)")


def stadium_physics(st: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in ["No Altitude", "Medium Altitude", "Extreme Altitude"]:
        g = st[st["categoria"] == c]
        fig.add_trace(go.Scatter(x=g["drag_ratio"], y=g["magnus_ratio"], mode="markers", name=SHORT[c], text=g["stadium_id"],
                                 marker=dict(color=ENV_COLORS[c], size=12, line=dict(width=0.6, color="white")),
                                 error_x=dict(type="data", symmetric=False, array=g["drag_hi"] - g["drag_ratio"], arrayminus=g["drag_ratio"] - g["drag_lo"],
                                              thickness=1), error_y=dict(type="data", symmetric=False, array=g["magnus_hi"] - g["magnus_ratio"],
                                                                         arrayminus=g["magnus_ratio"] - g["magnus_lo"], thickness=1),
                                 hovertemplate="%{text}<br>drag %{x:.3f} · Magnus %{y:.3f}<extra></extra>"))
    return _layout(fig, 400, title=dict(text="Cada estadio anónimo: fuerza del aire relativa a No Altitude", font=dict(size=14)),
                   xaxis_title="ratio de drag (1 = promedio de No Altitude)", yaxis_title="ratio de Magnus",
                   legend=dict(orientation="h", y=-0.22, x=0))


# ---------------------------------------------------------------------------
# Ampliación: tipos de pitch, regiones, estabilidad, contrafactual
# ---------------------------------------------------------------------------

# métrica del CSV de tipos → (etiqueta, factor, unidad)
TYPE_METRICS = {"whiff_swing": ("Δ Whiff/Swing", 100, "pp"), "bip_swing": ("Δ BIP/Swing", 100, "pp"),
                "rv_bip": ("Δ Run Value/BIP", 1, "carreras"), "ivb": ("Δ IVB", 1, "pulg."), "hb_abs": ("Δ movimiento horizontal |HB|", 1, "pulg."),
                "delta_por_100": ("Costo total por 100 pitches", 1, "carreras/100")}


# posición de la etiqueta de cada tipo (solo estética: evita que se encimen)
TYPE_TEXTPOS = {"Four-Seam": "bottom center", "Sinker": "middle right", "Cutter": "middle right", "Slider": "bottom center",
                "Curveball": "top center", "Changeup": "middle left", "Splitter": "top center"}


def pitch_type_scatter(t: pd.DataFrame, min_pitches: int) -> go.Figure:
    """X = Δ whiff/swing en Extreme, Y = costo total en Extreme; tamaño = pitches. Tipos con poca muestra en hueco."""
    fig = go.Figure()
    for ok, name in ((True, "muestra razonable"), (False, f"muestra pequeña (< {min_pitches:,} pitches)")):
        g = t[(t["pitches"] >= min_pitches) == ok]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=100 * g["whiff_swing_E"], y=g["delta_por_100_E"], mode="markers+text", text=g["tipo"], name=name,
            textposition=[TYPE_TEXTPOS.get(tp, "top center") for tp in g["tipo"]],
            marker=dict(size=np.sqrt(g["pitches"]) / 30 + 7, color="#d1495b" if ok else "white", line=dict(width=2, color="#d1495b"), opacity=0.85),
            error_x=dict(type="data", array=196 * g["whiff_swing_E_se"], color="#c9cfdb", thickness=1),
            error_y=dict(type="data", array=1.96 * g["delta_por_100_E_se"], color="#c9cfdb", thickness=1),
            customdata=np.c_[g["pitches"]],
            hovertemplate="%{text}<br>Δ whiff/swing %{x:+.1f} pp<br>costo %{y:+.2f} carreras/100<br>%{customdata[0]:,} pitches<extra></extra>"))
    fig.add_vline(x=0, line_color="#8892a6")
    return _layout(fig, 460, title=dict(text="Extreme vs. No Altitude por tipo (barras = IC 95%; tamaño = pitches)", font=dict(size=14)),
                   xaxis_title="Δ whiff por swing (pp) · ← pierde más whiff", yaxis_title="costo (carreras/100) · ↑ pierde más valor",
                   legend=dict(orientation="h", y=-0.2, x=0))


def pitch_type_compare(t: pd.DataFrame, metric: str, highlight: str | None = None) -> go.Figure:
    lab, k, unit = TYPE_METRICS[metric]
    fig = go.Figure()
    for env, key in (("Medium", "M"), ("Extreme", "E")):
        y = k * t[f"{metric}_{key}"]
        fig.add_trace(go.Bar(x=t["tipo"], y=y, name=env, marker_color=ENV_COLORS[env],
                             marker_line=dict(width=[3 if tp == highlight else 0 for tp in t["tipo"]], color="#13213c"),
                             error_y=dict(type="data", array=1.96 * k * t[f"{metric}_{key}_se"], color="#5b6577", thickness=1),
                             text=[f"{v:+.2f}" if unit.startswith("carreras") else f"{v:+.1f}" for v in y], textposition="outside"))
    fig.add_hline(y=0, line_color="#8892a6")
    return _layout(fig, 360, barmode="group", title=dict(text=f"{lab} vs. No Altitude ({unit})", font=dict(size=14)),
                   yaxis_title=unit, legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


def pitch_type_detail(row: pd.Series) -> go.Figure:
    """Cambio del tipo elegido en cada métrica, Medium y Extreme, con IC 95%."""
    from plotly.subplots import make_subplots
    keys = list(TYPE_METRICS)
    short = {"whiff_swing": "Δ whiff/sw", "bip_swing": "Δ BIP/sw", "rv_bip": "Δ RV/BIP", "ivb": "Δ IVB", "hb_abs": "Δ |HB|", "delta_por_100": "costo"}
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[f"{short[m]}<br>({TYPE_METRICS[m][2]})" for m in keys], horizontal_spacing=0.06)
    for j, m in enumerate(keys, start=1):
        _, k, unit = TYPE_METRICS[m]
        vals = [k * row[f"{m}_M"], k * row[f"{m}_E"]]
        err = [1.96 * k * row[f"{m}_M_se"], 1.96 * k * row[f"{m}_E_se"]]
        fig.add_trace(go.Bar(x=["Medium", "Extreme"], y=vals, marker_color=[ENV_COLORS["Medium"], ENV_COLORS["Extreme"]],
                             error_y=dict(type="data", array=err, color="#5b6577", thickness=1), showlegend=False,
                             text=[f"{v:+.2f}" if unit.startswith("carreras") else f"{v:+.1f}" for v in vals], textposition="outside"),
                      row=1, col=j)
        fig.add_hline(y=0, line_color="#8892a6", row=1, col=j)
    fig.update_annotations(font_size=11)
    return _layout(fig, 330)


def region_bars(reg: pd.DataFrame) -> go.Figure:
    """Cambio del valor de cada región (carreras por 100 pitches; positivo = el pitcher pierde valor)."""
    fig = go.Figure()
    for env, col, se in (("Medium", "medium_menos_no_por_100", "ee_medium"), ("Extreme", "extreme_menos_no_por_100", "ee_extreme")):
        fig.add_trace(go.Bar(x=reg["etiqueta"], y=reg[col], name=f"{env} − No Altitude", marker_color=ENV_COLORS[env],
                             error_y=dict(type="data", array=1.96 * reg[se], color="#5b6577", thickness=1),
                             text=[f"{v:+.2f}" for v in reg[col]], textposition="outside"))
    fig.add_hline(y=0, line_color="#8892a6")
    return _layout(fig, 360, barmode="group", title=dict(text="¿Cuánto valor pierde cada región en altitud? (IC 95%)", font=dict(size=14)),
                   yaxis_title="carreras/100 (+ = peor para el pitcher)", legend=dict(orientation="h", y=-0.15, x=0))


def stability_bars(rows: pd.DataFrame, highlight: dict) -> go.Figure:
    """Estabilidad año a año de cada métrica (r entre temporadas, IC 95%), coloreada por grupo."""
    r = rows.sort_values("estabilidad_cruda")
    colors = [highlight.get(m, "#c9cfdb") for m in r["metrica"]]
    fig = go.Figure(go.Bar(y=r["etiqueta"], x=r["estabilidad_cruda"], orientation="h", marker_color=colors,
                           error_x=dict(type="data", symmetric=False, array=r["estabilidad_hi"] - r["estabilidad_cruda"],
                                        arrayminus=r["estabilidad_cruda"] - r["estabilidad_lo"], color="#5b6577", thickness=1),
                           text=[f"{v:.2f}" for v in r["estabilidad_cruda"]], textposition="outside"))
    fig.add_vline(x=0, line_color="#8892a6")
    return _layout(fig, 430, title=dict(text="¿Se repite de un año al siguiente? (r entre temporadas, IC 95%)", font=dict(size=14)),
                   xaxis_title="correlación año a año del mismo pitcher", showlegend=False)


def counterfactual_check(sh: dict, labels: dict) -> go.Figure:
    """Validación split-half del contrafactual físico (Extreme): cambio observado vs. predicho (mediana entre grupos)."""
    keys = [k for k in labels if k in sh]
    fig = go.Figure()
    for name, col, color in (("observado", "observado", "#5b6577"), ("predicho por el modelo físico", "predicho", "#d1495b")):
        fig.add_trace(go.Bar(x=[labels[k] for k in keys], y=[sh[k][col] for k in keys], name=name, marker_color=color,
                             text=[f"{sh[k][col]:+.2f}" for k in keys], textposition="outside"))
    fig.add_hline(y=0, line_color="#8892a6")
    return _layout(fig, 340, barmode="group", title=dict(text="Validación split-half: cambio en Extreme, mitad de pitchers no usada", font=dict(size=14)),
                   legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


# ---------------------------------------------------------------------------
# Pitching+ general vs. altitud y lanzamiento óptimo por ambiente
# ---------------------------------------------------------------------------


def relative_gain_bars(tab: pd.DataFrame) -> go.Figure:
    """Ganancia de valor relativo en Extreme (frente al típico del mismo conteo) por quintil de P(whiff), antes y con dos strikes."""
    from plotly.subplots import make_subplots
    fases = ["0 o 1 strike", "dos strikes"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{f}" for f in fases], shared_yaxes=True)
    for j, f in enumerate(fases, start=1):
        t = tab[tab["fase"] == f]
        for col, name, color in (("gana_vs_general", "vs. General", "#1f3a63"), ("gana_vs_no", "vs. No Altitude", ENV_COLORS["Extreme"])):
            fig.add_trace(go.Bar(x=t["quintil_whiff"], y=t[col], name=name, marker_color=color, showlegend=(j == 1),
                                 text=[f"{v:+.2f}" for v in t[col]], textposition="outside"), row=1, col=j)
        fig.add_hline(y=0, line_color="#8892a6", row=1, col=j)
    fig.update_xaxes(title_text="quintil de P(whiff) (Q5 = más whiff)")
    fig.update_yaxes(title_text="ganancia en Extreme (carreras/100)", row=1, col=1)
    return _layout(fig, 360, barmode="group", legend=dict(orientation="h", y=1.18, x=1, xanchor="right"))


def contribution_bars(contrib: dict) -> go.Figure:
    """Por qué cambia el valor relativo: contribución de cada grupo de resultados (+ = gana valor en el ambiente alterno)."""
    keys = list(contrib)
    vals = [-100 * contrib[k] for k in keys]
    fig = go.Figure(go.Bar(y=keys, x=vals, orientation="h", marker_color=["#2e8b57" if v > 0 else "#d1495b" for v in vals],
                           text=[f"{v:+.2f}" for v in vals], textposition="outside"))
    fig.add_vline(x=0, line_color="#8892a6")
    fig = _layout(fig, 230, xaxis_title="carreras por 100 pitches frente al típico del conteo (+ = gana valor)", showlegend=False)
    lim = 1.6 * max(abs(v) for v in vals) if any(vals) else 1
    fig.update_xaxes(range=[-lim, lim])
    fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
    return fig


def profile_radar(rows: dict, labels: dict) -> go.Figure:
    """Perfil óptimo en percentiles del tipo (0–100) para cada objetivo."""
    cats = list(labels.values())
    colors = {"General": "#1f3a63", "Extreme": ENV_COLORS["Extreme"], "Extreme (física aprox.)": ENV_COLORS["Medium"], "Mejor Stuff+": "#8892a6"}
    fig = go.Figure()
    for name, pct in rows.items():
        r = [pct[c] for c in labels] + [pct[list(labels)[0]]]
        fig.add_trace(go.Scatterpolar(r=r, theta=cats + [cats[0]], name=name, line=dict(color=colors.get(name, "#5b6577"), width=2),
                                      fill="toself" if name == "General" else None, opacity=0.85))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], ticksuffix="")))
    fig = _layout(fig, 380, legend=dict(orientation="h", y=-0.12, x=0), title=dict(text="Perfil óptimo en percentiles del tipo", font=dict(size=14)))
    fig.update_layout(margin=dict(l=70, r=70, t=50, b=10))
    return fig


def location_pair(h: pd.DataFrame, title_a: str = "Pitching+ general", title_b: str = "Pitching+ Extreme (costos)",
                  col_b: str = "pp_extreme") -> go.Figure:
    """Pitching+ según la ubicación para un lanzamiento fijo: general vs. Extreme, misma escala de color."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title_a, title_b), horizontal_spacing=0.08)
    vmin = float(min(h["pp_general"].min(), h[col_b].min()))
    vmax = float(max(h["pp_general"].max(), h[col_b].max()))
    for j, col in enumerate(("pp_general", col_b), start=1):
        piv = h.pivot(index="z", columns="x", values=col)
        fig.add_trace(go.Heatmap(z=piv.to_numpy(), x=piv.columns, y=piv.index, zmin=vmin, zmax=vmax, colorscale="RdBu", showscale=(j == 2),
                                 colorbar=dict(title="Pitching+"), hovertemplate="lado %{x:+.1f} · altura %{y:.1f}<br>%{z:.1f}<extra></extra>"),
                      row=1, col=j)
        best = h.loc[h[col].idxmax()]
        fig.add_trace(go.Scatter(x=[best["x"]], y=[best["z"]], mode="markers", marker=dict(symbol="star", size=16, color="gold",
                                 line=dict(color="black", width=1)), showlegend=False, hoverinfo="skip"), row=1, col=j)
        fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="black", width=1.5), row=1, col=j)
        fig.update_xaxes(title_text="lado (pies, + = lado del brazo)", row=1, col=j)
    fig.update_yaxes(title_text="altura (pies)", row=1, col=1)
    return _layout(fig, 400)


# ---------------------------------------------------------------------------
# Recomendaciones operativas: formas de pitch, mapa de movimiento, arsenales
# ---------------------------------------------------------------------------


def shape_effects_chart(e: pd.DataFrame, metric: str, label: str, scale: float = 1.0) -> go.Figure:
    """Cambio en Extreme por tipo × forma (bajo / medio / alto movimiento característico), IC 95%."""
    colors = {"bajo": "#9fb4d6", "medio": "#4c78a8", "alto": "#13213c"}
    fig = go.Figure()
    for f in ["bajo", "medio", "alto"]:
        g = e[e["forma"] == f]
        y = scale * g[f"{metric}_E"]
        fig.add_trace(go.Bar(x=g["tipo"], y=y, name=f"{f} movimiento característico", marker_color=colors[f],
                             error_y=dict(type="data", array=1.96 * scale * g[f"{metric}_E_se"], color="#5b6577", thickness=1),
                             customdata=g["eje"], hovertemplate="%{x} · " + f + " %{customdata}<br>%{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#8892a6")
    return _layout(fig, 340, barmode="group", title=dict(text=label, font=dict(size=14)), legend=dict(orientation="h", y=-0.18, x=0))


def movement_map_chart(m: pd.DataFrame, show_shift: bool = True) -> go.Figure:
    """Mapa de movimiento HB–IVB de cada pitcher × tipo (No Altitude), color = Stuff+ del tipo; flechas = cambio en Extreme."""
    fig = go.Figure()
    rest = m[m["grupo"] == "resto"]
    fig.add_trace(go.Scatter(x=rest["hb_arm"], y=rest["ivb"], mode="markers", name="resto", marker=dict(color="#c9cfdb", size=6, opacity=0.6),
                             hovertemplate="HB %{x:.1f} · IVB %{y:.1f}<extra></extra>"))
    for grp, color, sym in (("10% más efectivo", "#2e8b57", "circle"), ("10% menos efectivo", "#d1495b", "x")):
        g = m[m["grupo"] == grp]
        fig.add_trace(go.Scatter(x=g["hb_arm"], y=g["ivb"], mode="markers", name=grp, marker=dict(color=color, size=9, symbol=sym),
                                 customdata=np.c_[g["stuff_plus_tipo"], g["velo"], g["pitcher_anon_id"]],
                                 hovertemplate="%{customdata[2]}<br>HB %{x:.1f} · IVB %{y:.1f}<br>Stuff+ del tipo %{customdata[0]:.0f} · "
                                               "%{customdata[1]:.1f} mph<extra></extra>"))
        if show_shift:
            s = g.dropna(subset=["ivb_ext", "hb_arm_ext"])
            for r in s.itertuples():
                fig.add_annotation(x=r.hb_arm_ext, y=r.ivb_ext, ax=r.hb_arm, ay=r.ivb, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color, opacity=0.6)
    fig.add_hline(y=0, line_color="#e3e7ef")
    fig.add_vline(x=0, line_color="#e3e7ef")
    return _layout(fig, 460, xaxis_title="movimiento horizontal (pulg., + = lado del brazo)", yaxis_title="movimiento vertical inducido (pulg.)",
                   legend=dict(orientation="h", y=-0.15, x=0))


def arsenal_chart(a: pd.DataFrame) -> go.Figure:
    t = a.iloc[::-1]
    fig = go.Figure(go.Scatter(x=t["stuff_plus_extreme"], y=t["arsenal"], mode="markers", marker=dict(color="#d1495b", size=10),
                               error_x=dict(type="data", symmetric=False, array=t["stuff_extreme_hi"] - t["stuff_plus_extreme"],
                                            arrayminus=t["stuff_plus_extreme"] - t["stuff_extreme_lo"], color="#8892a6"),
                               customdata=np.c_[t["pitchers_con_extreme"], t["aepp"]],
                               hovertemplate="%{y}<br>Stuff+ en Extreme %{x:.1f}<br>%{customdata[0]} pitchers · AEPP %{customdata[1]:+.2f}<extra></extra>"))
    fig.add_vline(x=100, line_dash="dot", line_color="#8892a6")
    return _layout(fig, max(320, 28 * len(t) + 80), xaxis_title="Stuff+ Physical con sus pitches en Extreme (escala de pitcher, intervalo 90%)")
