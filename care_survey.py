import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL
import json

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1h8LrASJXu1vfeE37Fu9ScLriYYlOC6S39w6haq9LHmc/export?format=csv")
df["Respondent ID"] = df["Respondent ID"].astype(str)

# Tagged open-ended answers
tagged_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1jUtETge_cxWGBLv4KVoXpS79H5SwYBvQoFnFzmfwxVg/export?format=csv")
tagged_df["Respondent ID"] = tagged_df["Respondent ID"].astype(str)
tagged_df = tagged_df.dropna(subset=["tagged", "question"])
TAGGED_QUESTIONS = sorted(tagged_df["question"].unique().tolist())

# Drop open-text / "other specify" questions and questions with too many unique answers
EXCLUDE = ["other specify", "Please list"]
MAX_UNIQUE_ANSWERS = 20

def is_valid_question(q):
    if any(x.lower() in q.lower() for x in EXCLUDE):
        return False
    n_unique = df[df["question"] == q]["answer"].nunique()
    return n_unique <= MAX_UNIQUE_ANSWERS

GROUPS = {
    "Задоволеність додатком": [
        "How would you rate the quality of information on outdoor plant care in PlantIn?",
        "How would you rate the quality of PlantIn’s weather notifications?",
        "How likely are you to recommend PlantIn to a friend?",
    ],
    "Дані про дії користувача в реальному житті": [
        "Do you check the weather for your outdoor plants?",
        "How do you prepare your outdoor plants for winter?",
    ],
    "Дані про дії користувача в додатку": [
        "Have you ever marked watering/misting in the app after it rained?",
        "How does PlantIn help you care for your outdoor plants?",
    ],
    "Дані про рослини користувача": [
        "What do you consider as an outdoor plant?",
        "How do you water your outdoor plants?",
        "How many outdoor plants do you currently have?",
        "What do you do with your outdoor plants during heatwaves?",
        "What do you do with your outdoor plants when strong wind or heavy rain is expected?",
        "What kind of care do you usually give your outdoor plants?",
        "What problems did you face with your outdoor plants after winter?",
        "Where do your outdoor plants grow?",
    ],
    "Протеговані відповіді відкритих запитань": TAGGED_QUESTIONS,
}

# Flat ordered list for indexing (regular questions only)
questions = [q for qs in list(GROUPS.values())[:-1] for q in qs if is_valid_question(q)]
# Tagged questions get indices after regular ones
tagged_q_index = {q: len(questions) + i for i, q in enumerate(TAGGED_QUESTIONS)}

ALL_RESPONDENTS = set(df["Respondent ID"].unique())

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_tagged_counts(question, respondent_ids=None):
    sub = tagged_df[tagged_df["question"] == question]
    if respondent_ids is not None:
        sub = sub[sub["Respondent ID"].isin(respondent_ids)]
    counts = (sub.groupby("tagged")["Respondent ID"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"Respondent ID": "count", "tagged": "answer"}))
    counts = counts.sort_values("count", ascending=True)
    return counts

def get_counts(question, respondent_ids=None):
    sub = df[df["question"] == question]
    if respondent_ids is not None:
        sub = sub[sub["Respondent ID"].isin(respondent_ids)]
    counts = (sub.groupby("answer")["Respondent ID"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"Respondent ID": "count"}))
    counts = counts.sort_values("count", ascending=True)
    return counts


def truncate(text, n=22):
    return text if len(text) <= n else text[:n] + "…"

def unique_labels(answers, n=22):
    """Truncate labels but ensure uniqueness by appending index if collision."""
    seen = {}
    result = []
    for ans in answers:
        short = truncate(ans, n)
        if short in seen and seen[short] != ans:
            # collision — use longer truncation
            short = truncate(ans, 35)
        seen[short] = ans
        result.append(short)
    return result

def wrap_title(text, width=35):
    """Break title into lines of max `width` chars at word boundaries."""
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)

LABEL_MARGIN = 5  # automargin handles label width

# ── Section colors (base hex) ──────────────────────────────────────────────────
SECTION_COLORS = {
    "Задоволеність додатком":                          "#4c9be8",
    "Дані про дії користувача в реальному житті":      "#27ae60",
    "Дані про дії користувача в додатку":              "#8e44ad",
    "Дані про рослини користувача":                    "#e67e22",
    "Протеговані відповіді відкритих запитань":        "#16a085",
}

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def gradient_colors(base_hex, n, selected_idx=None):
    """Generate n colors from light→dark based on base color. Selected bar is darker."""
    r, g, b = hex_to_rgb(base_hex)
    colors = []
    for i in range(n):
        t = 0.35 + 0.65 * (i / max(n - 1, 1))  # 0.35 (light) → 1.0 (full)
        colors.append(f"rgba({int(r*t)},{int(g*t)},{int(b*t)},1)")
    if selected_idx is not None:
        colors[selected_idx] = "#e74c3c"  # highlight selected in red
    return colors

def wrap_label(text, width=25):
    """Insert <br> into label at word boundaries."""
    words = text.split()
    lines, line, length = [], [], 0
    for w in words:
        if length + len(w) > width and line:
            lines.append(" ".join(line))
            line, length = [w], len(w)
        else:
            line.append(w)
            length += len(w) + 1
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)

NPS_Q = "How likely are you to recommend PlantIn to a friend?"

def calc_nps_csat(respondent_ids=None):
    sub = df[df["question"] == NPS_Q]
    if respondent_ids is not None:
        sub = sub[sub["Respondent ID"].isin(respondent_ids)]

    counts = sub.groupby("answer")["Respondent ID"].nunique()
    total = counts.sum()
    if total == 0:
        return 0, 0, 0

    promoters  = counts.get("5 - I would definitely recommend it", 0)
    passives   = counts.get("4", 0) + counts.get("3", 0)
    detractors = counts.get("2", 0) + counts.get("1", 0) + counts.get("0 - I would never recommend it", 0)

    nps  = round((promoters - detractors) / total * 100)
    csat = round((promoters + passives) / total * 100)
    return nps, csat, int(total)


def make_nps_csat_card(respondent_ids=None):
    nps, _, total = calc_nps_csat(respondent_ids)

    # NPS range is -100 to +100
    bar_color = "#27ae60" if nps >= 30 else "#e67e22" if nps >= 0 else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=nps,
        number={"font": {"size": 36, "color": "#222"}},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 0, "visible": False},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#f0f0f0",
            "borderwidth": 0,
            "steps": [
                {"range": [-100,  0], "color": "#fce8e6"},
                {"range": [0,    30], "color": "#fef3e2"},
                {"range": [30,  100], "color": "#e8f8f0"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.75,
                "value": nps,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=10),
        height=200,
        paper_bgcolor="white",
        font={"family": "sans-serif"},
        annotations=[dict(
            text="Promoters (5) − Detractors (0–2)",
            xref="paper", yref="paper",
            x=0.5, y=-0.05, xanchor="center",
            showarrow=False,
            font=dict(size=10, color="#aaa"),
        )],
    )

    return html.Div([
        html.Div([
            html.Span("NPS", style={"fontSize": 12, "fontWeight": "600", "color": "#222"}),
            html.Span(" · шкала від −100 до +100", style={"fontSize": 10, "color": "#999", "marginLeft": 6}),
        ], style={"padding": "8px 8px 0"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ], style={"background": "white", "borderRadius": 6, "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"})

def make_figure(question, respondent_ids=None, selected_answer=None,
                precomputed_counts=None, total_override=None, base_color="#4c9be8"):
    counts = precomputed_counts if precomputed_counts is not None else get_counts(question, respondent_ids)
    if total_override is not None:
        total = total_override
    else:
        total = len(respondent_ids) if respondent_ids is not None else len(ALL_RESPONDENTS)

    n = len(counts)
    sel_idx = None
    if selected_answer and selected_answer in counts["answer"].tolist():
        sel_idx = counts["answer"].tolist().index(selected_answer)
    colors = gradient_colors(base_color, n, sel_idx)

    # Wrap y-axis labels at word boundaries
    wrapped_labels = [wrap_label(str(a)) for a in counts["answer"]]

    pcts = (counts["count"] / total * 100).round(1)
    bar_text = [f"  {c} ({p}%)" for c, p in zip(counts["count"], pcts)]

    # Row height: base 38px + 16px per extra wrapped line
    max_lines = max((lbl.count("<br>") + 1) for lbl in wrapped_labels) if wrapped_labels else 1
    row_h = 38 + (max_lines - 1) * 16

    fig = go.Figure(go.Bar(
        x=counts["count"],
        y=wrapped_labels,
        orientation="h",
        marker_color=colors,
        text=bar_text,
        textposition="outside",
        textfont=dict(size=10, color="#333"),
        customdata=list(zip(counts["answer"], pcts)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{x} respondents (%{customdata[1]}%)<extra></extra>",
        cliponaxis=False,
    ))

    fig.update_layout(
        margin=dict(l=LABEL_MARGIN, r=90, t=10, b=5),
        xaxis=dict(showgrid=False, visible=False, range=[0, counts["count"].max() * 1.4]),
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(220, n * row_h + 60),
        showlegend=False,
        hoverlabel=dict(bgcolor="#333", font=dict(color="white", size=12), bordercolor="#333"),
    )
    return fig, total


# ── App layout ─────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # для gunicorn

app.index_string = '''
<!DOCTYPE html>
<html>
<head>{%metas%}<title>PlantIn Survey</title>{%favicon%}{%css%}
<style>
.info-tooltip { position: relative; display: inline-block; cursor: default; }
.info-tooltip .tooltip-text {
    visibility: hidden; opacity: 0;
    width: 280px; background: #333; color: #fff;
    font-size: 12px; line-height: 1.5; border-radius: 6px;
    padding: 8px 12px; position: absolute; top: 28px; right: 0;
    z-index: 999; transition: opacity 0.2s; pointer-events: none;
}
.info-tooltip:hover .tooltip-text { visibility: visible; opacity: 1; }

.charts-grid { display: grid; gap: 16px; grid-template-columns: repeat(4, 1fr); }
@media (max-width: 1200px) { .charts-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 900px)  { .charts-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px)  { .charts-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

TOOLTIP_STYLE = {
    "display": "inline-block",
    "marginLeft": 8,
    "fontSize": 13,
    "color": "#aaa",
    "cursor": "default",
    "position": "relative",
}

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H2("PlantIn Survey Explorer", style={"margin": "0", "fontSize": 22, "fontWeight": "700"}),
                html.Span(f"n = {len(ALL_RESPONDENTS)}", style={
                    "fontSize": 12, "color": "#aaa", "marginLeft": 10,
                    "fontWeight": "400", "alignSelf": "flex-end", "paddingBottom": 2,
                }),
            ], style={"display": "flex", "alignItems": "baseline", "gap": 0}),

            html.Div([
                html.Button("Скинути фільтр", id="reset-btn", style={
                    "padding": "6px 16px",
                    "cursor": "pointer",
                    "border": "1.5px solid #4c9be8",
                    "borderRadius": 6,
                    "background": "white",
                    "color": "#4c9be8",
                    "fontWeight": "600",
                    "fontSize": 13,
                    "transition": "all 0.15s",
                }),
                html.Div([
                    html.Span("ℹ️", style={"fontSize": 16}),
                    html.Span(
                        "Клікни на будь-який бар — всі графіки відфільтруються по цих респондентах. Клікни ще раз — скидання.",
                        className="tooltip-text"
                    ),
                ], className="info-tooltip"),
            ], style={"display": "flex", "alignItems": "center", "gap": 8}),
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
    ], style={"padding": "14px 24px", "borderBottom": "2px solid #e8e8e8", "background": "#fafafa"}),

    dcc.Store(id="filter-state", data={"respondent_ids": None, "source_q": None, "source_ans": None}),

    html.Div(id="charts-container", style={"padding": "0 24px 32px"}),
], style={"fontFamily": "sans-serif"})


# ── Build chart list on load ───────────────────────────────────────────────────
@app.callback(
    Output("charts-container", "children"),
    Input("filter-state", "data"),
)
def render_charts(filter_state):
    respondent_ids = filter_state.get("respondent_ids")
    source_q = filter_state.get("source_q")
    source_ans = filter_state.get("source_ans")

    rid_set = set(respondent_ids) if respondent_ids else None

    q_index = {q: i for i, q in enumerate(questions)}
    sections = []

    for group_name, group_qs in GROUPS.items():
        base_color = SECTION_COLORS.get(group_name, "#4c9be8")
        is_tagged_section = group_name == "Протеговані відповіді відкритих запитань"

        if is_tagged_section:
            valid_qs = [q for q in group_qs if pd.notna(q)]
        else:
            valid_qs = [q for q in group_qs if q in q_index]

        if not valid_qs:
            continue

        cards = []
        for q in valid_qs:
            if is_tagged_section:
                i = tagged_q_index[q]
                counts = get_tagged_counts(q, rid_set)
                total = counts["count"].sum()
                fig, _ = make_figure(q, rid_set, source_ans if q == source_q else None,
                                     precomputed_counts=counts, total_override=total,
                                     base_color=base_color)
            else:
                i = q_index[q]
                sel_ans = source_ans if q == source_q else None
                fig, total = make_figure(q, rid_set, sel_ans, base_color=base_color)

            cards.append(
                html.Div([
                    html.Div([
                        html.Span(q, style={"fontSize": 12, "fontWeight": "600", "color": "#222"}),
                    ], style={"padding": "8px 8px 2px", "lineHeight": "1.3"}),
                    dcc.Graph(
                        id={"type": "survey-chart", "index": i},
                        figure=fig,
                        config={"displayModeBar": False},
                    ),
                ], style={"background": "white", "borderRadius": 6,
                          "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"})
            )
            if q == NPS_Q:
                cards.append(make_nps_csat_card(rid_set))

        sections.append(html.Div([
            html.Div([
                html.Span(group_name, style={
                    "fontSize": 13,
                    "fontWeight": "700",
                    "color": "#222",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                }),
            ], style={
                "borderLeft": f"4px solid {base_color}",
                "paddingLeft": 10,
                "marginBottom": 12,
            }),
            html.Div(cards, className="charts-grid"),
        ], style={"marginTop": 36}))

    return sections


# ── Click handler ──────────────────────────────────────────────────────────────
@app.callback(
    Output("filter-state", "data"),
    Input({"type": "survey-chart", "index": ALL}, "clickData"),
    Input("reset-btn", "n_clicks"),
    State("filter-state", "data"),
    prevent_initial_call=True,
)
def handle_click(click_data_list, reset_clicks, current_state):
    ctx = callback_context
    if not ctx.triggered:
        return current_state

    trigger_id = ctx.triggered[0]["prop_id"]

    # Reset button
    if "reset-btn" in trigger_id:
        return {"respondent_ids": None, "source_q": None, "source_ans": None}

    # Find which chart was clicked
    triggered_prop = json.loads(trigger_id.replace(".clickData", ""))
    chart_index = triggered_prop["index"]

    # Determine if it's a regular or tagged question
    all_questions = questions + TAGGED_QUESTIONS
    clicked_q = all_questions[chart_index]
    is_tagged = chart_index >= len(questions)

    click_data = click_data_list[chart_index]
    if not click_data:
        return current_state

    clicked_answer = click_data["points"][0]["customdata"][0]

    # Toggle: click same answer again → reset
    if (current_state.get("source_q") == clicked_q and
            current_state.get("source_ans") == clicked_answer):
        return {"respondent_ids": None, "source_q": None, "source_ans": None}

    # Filter respondents
    if is_tagged:
        mask = (tagged_df["question"] == clicked_q) & (tagged_df["tagged"] == clicked_answer)
        matching_ids = tagged_df[mask]["Respondent ID"].unique().tolist()
    else:
        mask = (df["question"] == clicked_q) & (df["answer"] == clicked_answer)
        matching_ids = df[mask]["Respondent ID"].unique().tolist()

    return {
        "respondent_ids": matching_ids,
        "source_q": clicked_q,
        "source_ans": clicked_answer,
    }


if __name__ == "__main__":
    app.run(debug=True, port=8050)
