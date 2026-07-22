"""
Generate an advisor-facing PDF report for Gate 4 mechanism-validation results.

The script reads the Gate 4 CSV outputs, recomputes the reported ratios and
Wilcoxon signed-rank p-values, embeds the generated figures, and writes a PDF to
output/pdf/gate4_advisor_report.pdf.

Usage from project root:
    python experiments/framing_validation/make_gate4_report.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Direct script execution needs both the src layout and the experiment packages.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
for _import_path in (_SRC_DIR, _PROJECT_ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))
_root = _PROJECT_ROOT
_EXPERIMENT_DIR = Path(__file__).resolve().parent

import pandas as pd
import numpy as np
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_HERE = Path(__file__).resolve().parent
OUT_PDF = _HERE / "output_reports" / "gate4_advisor_report.pdf"
G4A_DIR = _HERE / "output_gate4a"
G4A_N20_DIR = _HERE / "output_gate4a_n20"
G4A_N15_DIR = _HERE / "output_gate4a_n15"
G4B_DIR = _HERE / "output_gate4b"
G4A_FIG = G4A_DIR / "gate4a_fix_epistemic.png"
G4A_N20_FIG = G4A_N20_DIR / "gate4a_fix_epistemic.png"
G4A_N15_FIG = G4A_N15_DIR / "gate4a_fix_epistemic.png"
G4B_FIG = G4B_DIR / "gate4b_capacity_limit.png"


def f_true(x: np.ndarray, bump_sd: float) -> np.ndarray:
    base = np.sin(2 * np.pi * x)
    bump = 1.5 * np.exp(-((x - 0.5) ** 2) / (2 * bump_sd**2))
    return base + bump


def stage1_sites(part: str) -> np.ndarray:
    if part == "a":
        left = np.linspace(0.0, 0.34, 25)
        inside = np.linspace(0.37, 0.63, 6)
        right = np.linspace(0.66, 1.0, 25)
        return np.concatenate([left, inside, right])
    return np.linspace(0.0, 1.0, 100)


def _rankdata(values: list[float]) -> list[float]:
    """Average ranks for positive values, matching scipy.stats.rankdata."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def wilcoxon_less_p(x: pd.Series, y: pd.Series) -> float:
    """Exact one-sided Wilcoxon p-value for H1: x < y, for n <= 24."""
    diffs = [float(a - b) for a, b in zip(x, y) if float(a - b) != 0.0]
    n = len(diffs)
    if n == 0:
        return 1.0
    abs_vals = [abs(v) for v in diffs]
    ranks = _rankdata(abs_vals)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0.0)

    # Enumerate the exact null distribution. Gate 4 uses n=20, so this is cheap.
    total = 1 << n
    le_count = 0
    for signs in itertools.product((0, 1), repeat=n):
        stat = sum(rank for rank, sign in zip(ranks, signs) if sign)
        if stat <= w_plus + 1e-12:
            le_count += 1
    return le_count / total


def fmt(x: float, digits: int = 3) -> str:
    if abs(x) < 1e-4 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def load_g4a_summary(label: str, dir_path: Path, n_stage2: int) -> dict:
    g4a = pd.read_csv(dir_path / "gate4a_metrics.csv")

    a_hot = g4a.pivot(index="macrorep", columns="arm", values="hot_cdf_l2")
    a_q = g4a.pivot(index="macrorep", columns="arm", values="q_hat_post")

    return {
        "label": label,
        "n_stage2": n_stage2,
        "g4a": g4a,
        "a_utail_lhs_ratio": float((a_hot["utail"] / a_hot["lhs"]).median()),
        "a_utail_lhs_p": wilcoxon_less_p(a_hot["utail"], a_hot["lhs"]),
        "a_oracle_lhs_ratio": float((a_hot["oracle"] / a_hot["lhs"]).median()),
        "a_oracle_lhs_p": wilcoxon_less_p(a_hot["oracle"], a_hot["lhs"]),
        "a_q_ratio": float((a_q["utail"] / a_q["lhs"]).median()),
        "a_q_p": wilcoxon_less_p(a_q["utail"], a_q["lhs"]),
        "a_n1": g4a.groupby("arm")["n1_in_window"].mean().to_dict(),
        "a_si": g4a.groupby("arm")["si_utail_post"].median().to_dict(),
        "a_pre_si": float(g4a.drop_duplicates("macrorep")["pre_si_utail"].median()),
        "a_cov": g4a.groupby("arm")["coverage"].mean().to_dict(),
    }


def load_results() -> dict:
    a_main = load_g4a_summary("N1=60", G4A_DIR, 60)
    a_budget = [
        a_main,
        load_g4a_summary("N1=20", G4A_N20_DIR, 20),
        load_g4a_summary("N1=15", G4A_N15_DIR, 15),
    ]
    g4b = pd.read_csv(G4B_DIR / "gate4b_metrics.csv")
    lof = pd.read_csv(G4B_DIR / "gate4b_lof.csv")

    b_hot = g4b.pivot(index="macrorep", columns="arm", values="hot_cdf_l2")

    lof_fixed = g4b.loc[g4b["arm"] == "utail_fixed", "hot_lof_max"]
    lof_retune = g4b.loc[g4b["arm"] == "utail_retune", "hot_lof_max"]
    bg99 = float(lof.loc[~lof["in_window"], "lof"].quantile(0.99))

    return {
        "g4a": a_main["g4a"],
        "a_budget": a_budget,
        "g4b": g4b,
        "a_utail_lhs_ratio": a_main["a_utail_lhs_ratio"],
        "a_utail_lhs_p": a_main["a_utail_lhs_p"],
        "a_oracle_lhs_ratio": a_main["a_oracle_lhs_ratio"],
        "a_oracle_lhs_p": a_main["a_oracle_lhs_p"],
        "a_q_ratio": a_main["a_q_ratio"],
        "a_q_p": a_main["a_q_p"],
        "a_n1": a_main["a_n1"],
        "a_si": a_main["a_si"],
        "a_pre_si": a_main["a_pre_si"],
        "a_cov": a_main["a_cov"],
        "b_fixed_lhs_ratio": float((b_hot["utail_fixed"] / b_hot["lhs_fixed"]).median()),
        "b_retune_fixed_ratio": float((b_hot["utail_retune"] / b_hot["utail_fixed"]).median()),
        "b_retune_fixed_p": wilcoxon_less_p(b_hot["utail_retune"], b_hot["utail_fixed"]),
        "b_lof_fixed_gt10": float((lof_fixed > 10.0).mean()),
        "b_lof_retune_lt10": float((lof_retune < 10.0).mean()),
        "b_lof_fixed_med": float(lof_fixed.median()),
        "b_lof_retune_med": float(lof_retune.median()),
        "b_bg99": bg99,
        "b_lof_fixed_gt_bg99": float((lof_fixed > bg99).mean()),
        "b_lof_retune_gt_bg99": float((lof_retune > bg99).mean()),
        "b_ell_counts": g4b.loc[g4b["arm"] == "utail_retune", "ell_x"].value_counts().sort_index().to_dict(),
        "b_q": g4b.groupby("arm")["q_hat_post"].mean().to_dict(),
        "b_cov": g4b.groupby("arm")["coverage"].mean().to_dict(),
    }


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=15,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#17324d"),
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#17324d"),
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontSize=9.3,
            leading=12.2,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontSize=8.2,
            leading=10.2,
            textColor=colors.HexColor("#555555"),
        ),
        "eq": ParagraphStyle(
            "EquationCustom",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=7.7,
            leading=10.0,
            textColor=colors.HexColor("#202020"),
        ),
        "cell": ParagraphStyle(
            "CellCustom",
            parent=base["BodyText"],
            fontSize=7.8,
            leading=9.3,
            alignment=TA_LEFT,
        ),
        "cell_center": ParagraphStyle(
            "CellCenterCustom",
            parent=base["BodyText"],
            fontSize=7.8,
            leading=9.3,
            alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "CellHeaderCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.3,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
    }
    return styles


def p(text: str, styles: dict, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet_list(items: list[str], styles: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles), leftIndent=8) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
        bulletFontSize=5,
    )


def styled_table(rows: list[list[str]], styles: dict, widths: list[float]) -> Table:
    styled_rows = []
    for r, row in enumerate(rows):
        styled_row = []
        for i, cell in enumerate(row):
            if r == 0:
                styled_row.append(p(cell, styles, "cell_header"))
            else:
                styled_row.append(p(cell, styles, "cell_center" if i == 0 else "cell"))
        styled_rows.append(styled_row)
    table = Table(styled_rows, colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c3d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def equation_box(rows: list[tuple[str, str]], styles: dict, widths: list[float] | None = None) -> Table:
    if widths is None:
        widths = [1.55 * inch, 5.65 * inch]
    table_rows = []
    for label, equation in rows:
        table_rows.append([p(label, styles, "cell_header"), p(equation, styles, "eq")])
    table = Table(table_rows, colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c3d0")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f7f8fa")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def figure(path: Path, width: float, caption: str, styles: dict) -> KeepTogether:
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return KeepTogether([img, Spacer(1, 0.08 * inch), p(caption, styles, "small")])


def g4a_budget_table(res: dict, styles: dict) -> Table:
    rows = [
        ["Budget", "Window sites", "utail/lhs hot CDF L2", "oracle/lhs", "q_hat", "SI(lhs -> utail)", "Reading"],
    ]
    for item in res["a_budget"]:
        if item["n_stage2"] == 60:
            reading = "Mechanism works; CDF gain is saturated/bias-limited."
        else:
            reading = "Small budget still does not make raw u_tail beat LHS on CDF error."
        rows.append(
            [
                item["label"],
                f"{fmt(item['a_n1']['lhs'], 1)} -> {fmt(item['a_n1']['utail'], 1)}",
                f"{fmt(item['a_utail_lhs_ratio'])} (p={fmt(item['a_utail_lhs_p'], 3)})",
                fmt(item["a_oracle_lhs_ratio"]),
                fmt(item["a_q_ratio"]),
                f"{fmt(item['a_si']['lhs'], 1)} -> {fmt(item['a_si']['utail'], 1)}",
                reading,
            ]
        )
    return styled_table(rows, styles, [0.58 * inch, 0.86 * inch, 1.22 * inch, 0.78 * inch, 0.66 * inch, 0.95 * inch, 2.25 * inch])


def original_function_drawing() -> Drawing:
    """ReportLab drawing of the two true DGP functions used in Gate 4."""
    width = 6.9 * inch
    height = 2.35 * inch
    drawing = Drawing(width, height)

    panel_w = (width - 0.35 * inch) / 2.0
    panel_h = 1.65 * inch
    y0 = 0.40 * inch
    panels = [
        {
            "x0": 0.12 * inch,
            "title": "G4a: wide bump, data-scarcity hotspot",
            "sd": 0.10,
            "window": (0.35, 0.65),
            "sites": stage1_sites("a"),
        },
        {
            "x0": 0.12 * inch + panel_w + 0.30 * inch,
            "title": "G4b: narrow bump, capacity-limited hotspot",
            "sd": 0.03,
            "window": (0.41, 0.59),
            "sites": stage1_sites("b"),
        },
    ]
    xs = np.linspace(0.0, 1.0, 500)
    y_all = np.concatenate([f_true(xs, 0.10), f_true(xs, 0.03), np.sin(2 * np.pi * xs)])
    y_min = float(y_all.min() - 0.15)
    y_max = float(y_all.max() + 0.15)

    def sx(x: float, x0: float) -> float:
        return x0 + 0.35 * inch + x * (panel_w - 0.55 * inch)

    def sy(y: float) -> float:
        return y0 + 0.20 * inch + (y - y_min) / (y_max - y_min) * (panel_h - 0.42 * inch)

    for panel in panels:
        x0 = panel["x0"]
        x_left = sx(0.0, x0)
        x_right = sx(1.0, x0)
        y_bottom = sy(y_min)
        y_top = sy(y_max)

        drawing.add(String(x0 + panel_w / 2.0, height - 0.18 * inch, panel["title"], fontName="Helvetica-Bold", fontSize=8.2, textAnchor="middle"))
        drawing.add(Rect(x_left, y_bottom, x_right - x_left, y_top - y_bottom, strokeColor=colors.HexColor("#b7c3d0"), fillColor=None, strokeWidth=0.4))

        win_lo, win_hi = panel["window"]
        drawing.add(Rect(sx(win_lo, x0), y_bottom, sx(win_hi, x0) - sx(win_lo, x0), y_top - y_bottom, strokeColor=None, fillColor=colors.HexColor("#eeeeee")))
        drawing.add(Line(x_left, sy(0.0), x_right, sy(0.0), strokeColor=colors.HexColor("#888888"), strokeWidth=0.35))

        base_pts = []
        f_pts = []
        for x in xs:
            base_pts.extend([sx(float(x), x0), sy(float(np.sin(2 * np.pi * x)))])
            f_pts.extend([sx(float(x), x0), sy(float(f_true(np.array([x]), panel["sd"])[0]))])
        drawing.add(PolyLine(base_pts, strokeColor=colors.HexColor("#666666"), strokeWidth=0.9, strokeDashArray=[3, 2]))
        drawing.add(PolyLine(f_pts, strokeColor=colors.HexColor("#d95f02"), strokeWidth=1.7))

        rug_y0 = y_bottom + 0.05 * inch
        rug_y1 = rug_y0 + 0.12 * inch
        for x in panel["sites"]:
            drawing.add(Line(sx(float(x), x0), rug_y0, sx(float(x), x0), rug_y1, strokeColor=colors.HexColor("#1b9e77"), strokeWidth=0.35))

        for tick in [0.0, 0.5, 1.0]:
            drawing.add(Line(sx(tick, x0), y_bottom, sx(tick, x0), y_bottom - 0.04 * inch, strokeColor=colors.black, strokeWidth=0.35))
            drawing.add(String(sx(tick, x0), y_bottom - 0.15 * inch, f"{tick:.1f}", fontSize=6.4, textAnchor="middle"))
        for tick in [-1.0, 0.0, 1.0, 1.5]:
            drawing.add(Line(x_left - 0.035 * inch, sy(tick), x_left, sy(tick), strokeColor=colors.black, strokeWidth=0.35))
            drawing.add(String(x_left - 0.055 * inch, sy(tick) - 2.0, f"{tick:g}", fontSize=6.1, textAnchor="end"))
        drawing.add(String((x_left + x_right) / 2.0, y_bottom - 0.28 * inch, "x", fontSize=7.0, textAnchor="middle"))

    return drawing


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "Gate 4 mechanism-validation report")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    res = load_results()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.55 * inch,
        title="Gate 4 Mechanism-Validation Report",
        author="Jin Zhao",
    )
    story = []

    story.append(p("Gate 4: Diagnostic-Driven Model Repair", styles, "title"))
    story.append(p("Experiment-section draft for the CKME two-stage adaptive design paper - generated from local Gate 4 CSV outputs", styles, "subtitle"))

    story.append(p("Purpose", styles, "h1"))
    story.append(
        p(
            "Gate 4 evaluates whether an epistemic diagnostic can support a closed decision loop after the first-stage CKME fit. The experiment asks three questions. First, can the bootstrap tail-quantile variance u_tail identify a local region where the fitted conditional distribution is unreliable? Second, does targeted Stage-2 sampling remove that diagnostic signal and improve the local conditional-CDF error relative to space-filling LHS? Third, when sampling alone cannot remove the error, can a lack-of-fit statistic distinguish a data-scarcity problem from a model-capacity problem and route the procedure to a model repair?",
            styles,
        )
    )
    story.append(
        p(
            "The answer is mixed in a useful way. G4b provides the clean positive result: a narrow bump below the fixed kernel length scale creates a bias floor, targeted data alone does not fix it, and retuning ell_x reduces hotspot CDF error by about 9x. G4a validates the diagnostic/action mechanism, but the CDF-efficiency claim fails in this DGP, including under the N1=20 and N1=15 small-budget follow-ups.",
            styles,
        )
    )

    story.append(p("Data-Generating Model", styles, "h1"))
    story.append(
        equation_box(
            [
                ("Stochastic output", "Y(x) = f_s(x) + sigma * epsilon,    epsilon ~ N(0, 1),    sigma = 0.10,    x in [0, 1]."),
                ("Mean function", "f_s(x) = sin(2*pi*x) + 1.5 * exp(-(x - 0.5)^2 / (2*s^2))."),
                ("G4a setting", "s = 0.10, hotspot window W_a = [0.35, 0.65]; Stage-1 design is thinned in W_a."),
                ("G4b setting", "s = 0.03, hotspot window W_b = [0.41, 0.59]; fixed ell_x = 0.10 is too smooth for the bump."),
            ],
            styles,
        )
    )
    story.append(
        p(
            "Thus G4a is intended to represent a resolvable but undersampled feature, whereas G4b deliberately places the local feature below the fixed model length scale. The same observation noise is used in both parts, so the distinction is not an aleatoric-scale effect.",
            styles,
        )
    )

    story.append(p("Notation", styles, "h1"))
    notation_rows = [
        ["Symbol", "Definition", "Role in the experiment"],
        ["D0", "Stage-1 simulation data", "Used to fit the initial CKME conditional CDF estimator."],
        ["D1", "Stage-2 simulation data", "Allocated by LHS, u_tail targeting, oracle targeting, or retuning arm."],
        ["F_hat(t|x)", "CKME estimate of P(Y <= t | X=x)", "The fitted conditional distribution used for scores, quantiles, and diagnostics."],
        ["q_tau(x)", "Conditional quantile from F_hat", "Defined by the first grid point t with F_hat(t|x) >= tau."],
        ["W", "Hotspot window", "Local region where the designed bump creates the diagnostic challenge."],
        ["qhat", "Split-conformal calibration quantile", "Controls marginal 90% prediction-interval coverage."],
        ["u_tail(x)", "Bootstrap variance of fitted lower/upper tail quantiles", "Observable epistemic diagnostic used for targeted sampling."],
        ["LOF(x)", "Lack-of-fit statistic", "Observable post-sampling diagnostic for deciding whether to retune ell_x."],
    ]
    story.append(styled_table(notation_rows, styles, [0.75 * inch, 2.05 * inch, 4.4 * inch]))

    story.append(PageBreak())
    story.append(p("Estimands and Decision Rules", styles, "h1"))
    story.append(
        equation_box(
            [
                ("CP score", "S_i = |F_hat(Y_i | X_i) - 0.5|,    qhat = S_(ceil((1-alpha)(n_cal+1))),    alpha = 0.10."),
                ("Prediction interval", "C(x) = [q_max(0,0.5-qhat)(x), q_min(1,0.5+qhat)(x)]."),
                ("Hotspot CDF error", "e_CDF(x) = int (F_hat(t|x) - F_0(t|x))^2 dt,    HotCDFL2 = mean_{x in W} e_CDF(x)."),
                ("Bootstrap diagnostic", "u_tail(x) = Var_b(q_0.05^(b)(x)) + Var_b(q_0.95^(b)(x)), with B=30 site-bootstrap refits."),
                ("Separation index", "SI_W(g) = max_{x in W} g(x) / median_{x notin W} g(x)."),
                ("Stage-2 targeting", "P(select x_j) = gamma/|C| + (1-gamma) w_j/sum_k w_k,    gamma=0.20,    w_j = u_tail(x_j) or oracle e_CDF(x_j)."),
                ("Lack-of-fit", "LOF(x_j) = r_LOF * (Ybar_j - q_0.5(x_j))^2 / s_j^2,    r_LOF=20."),
            ],
            styles,
        )
    )
    story.append(p("G4a decision rule", styles, "h2"))
    story.append(
        p(
            "The intended positive C1 pattern is that u_tail-targeted Stage-2 sampling should concentrate simulation sites in W, reduce SI_W(u_tail), and reduce HotCDFL2 and qhat relative to LHS. The oracle arm is not implementable in practice; it is included only to show whether the window is in principle fixable by allocating more simulation sites.",
            styles,
        )
    )
    story.append(p("G4b decision rule", styles, "h2"))
    story.append(
        p(
            "For the capacity-limited case, the test is different. If utail_fixed/lhs_fixed is near one, then targeted data alone did not remove the bias floor. If utail_retune/utail_fixed is small and LOF falls sharply after retuning ell_x, the workflow supports the route: diagnose a hotspot, try targeted data, observe persistent LOF, then repair model capacity.",
            styles,
        )
    )

    story.append(PageBreak())

    story.append(p("Executive Verdict", styles, "h1"))
    story.append(
        bullet_list(
            [
                "<b>G4b is the core positive result.</b> A capacity-limited hotspot is not fixed by adding targeted data alone, but site-level CV retuning of ell_x reduces hotspot CDF error by about 9x.",
                "<b>G4a validates the targeting mechanism but not the originally specified effect-size gate.</b> u_tail drives Stage-2 sites into the window and removes the u_tail hotspot, but this does not translate into a clear hotspot CDF-error gain over LHS.",
                "<b>The small-budget follow-up has now been tested.</b> N1=20 and N1=15 still give utail/lhs hot CDF L2 ratios above 1.0, while oracle remains below LHS. The limitation is not only saturation; raw u_tail is not aligned enough with the CDF-error target in this DGP.",
                "<b>For the paper, lead with G4b and state G4a as a diagnostic/action demonstration.</b> A positive C1 efficiency figure would require a revised DGP or allocation rule, not just a smaller Stage-2 budget.",
            ],
            styles,
        )
    )

    story.append(p("Experiment Design", styles, "h1"))
    design_rows = [
        ["Part", "Purpose", "DGP / design", "Arms", "Main diagnostic"],
        [
            "G4a",
            "Fixable data-scarcity hotspot",
            "Wide bump sd=0.10; Stage 1 thinned inside [0.35, 0.65]; N1=60 plus N1=20/15 follow-ups, r=10",
            "lhs, utail, oracle",
            "Does u_tail-targeted allocation beat LHS on hotspot CDF L2 and q_hat?",
        ],
        [
            "G4b",
            "Capacity-limited hotspot",
            "Narrow bump sd=0.03 with fixed ell_x=0.1; N1=60, r=10",
            "lhs_fixed, utail_fixed, utail_retune",
            "Does LOF route from data acquisition to model retuning?",
        ],
    ]
    story.append(styled_table(design_rows, styles, [0.55 * inch, 1.35 * inch, 2.3 * inch, 1.25 * inch, 2.05 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        p(
            "Common setup: BASE_SEED=20260708, alpha=0.1, sigma=0.10, gamma=0.2 exploration floor, 20 macroreps, B=30 site-bootstrap refits for u_tail, shared calibration/test/LOF data within each macrorep. The G4a budget follow-up reruns only Part A with N1=20 and N1=15 using the same paired macrorep structure.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("Figure 0. Original DGP Functions", styles, "h1"))
    story.append(
        KeepTogether(
            [
                original_function_drawing(),
                Spacer(1, 0.08 * inch),
                p(
                    "The left panel is the G4a wide-bump DGP, where the feature is resolvable by ell_x=0.1 if enough local data are available. The right panel is the G4b narrow-bump DGP, where the bump is below the fixed length scale and creates a model-capacity bottleneck. Gray bands mark the hotspot windows; green rugs show Stage-1 site locations.",
                    styles,
                    "small",
                ),
            ]
        )
    )

    story.append(PageBreak())
    story.append(p("G4a: Data-Scarcity Hotspot", styles, "h1"))
    g4a_rows = [
        ["Quantity", "Result", "Interpretation"],
        [
            "Stage-2 concentration",
            f"utail places {fmt(res['a_n1']['utail'], 1)}/60 sites in the window; LHS places {fmt(res['a_n1']['lhs'], 1)}/60",
            "The action mechanism works: u_tail directs budget into the flagged region.",
        ],
        [
            "u_tail hotspot",
            f"median SI drops from {fmt(res['a_pre_si'], 1)} pre to {fmt(res['a_si']['utail'], 1)} post under utail; LHS remains {fmt(res['a_si']['lhs'], 1)}",
            "Targeting removes the bootstrap-variance signal much more than LHS.",
        ],
        [
            "Hot CDF L2",
            f"utail/lhs median ratio {fmt(res['a_utail_lhs_ratio'])}, p={fmt(res['a_utail_lhs_p'], 4)}; oracle/lhs ratio {fmt(res['a_oracle_lhs_ratio'])}",
            "The originally specified effect-size gate is not met; even oracle improves only modestly.",
        ],
        [
            "q_hat refund",
            f"utail/lhs q_hat ratio {fmt(res['a_q_ratio'])}, p={fmt(res['a_q_p'], 4)}",
            "Statistically small reduction, not a substantive width-tax refund.",
        ],
        [
            "Coverage",
            f"mean coverage: lhs {fmt(res['a_cov']['lhs'])}, utail {fmt(res['a_cov']['utail'])}, oracle {fmt(res['a_cov']['oracle'])}",
            "Split-CP validity sanity check holds.",
        ],
    ]
    story.append(styled_table(g4a_rows, styles, [1.18 * inch, 2.35 * inch, 3.67 * inch]))
    story.append(
        p(
            "Reading: G4a is a mechanism pass but an effect-size miss. The original N1=60 result could be explained partly by saturation: LHS already contributes about 18 sites in the hotspot. However, the starved-budget checks below rule out the simple claim that lowering N1 to 15-20 is enough to produce a positive C1 result in this DGP.",
            styles,
        )
    )
    story.append(p("G4a Budget Sensitivity", styles, "h2"))
    story.append(g4a_budget_table(res, styles))
    story.append(
        p(
            "Interpretation: lower budgets preserve the allocation signal but do not improve the CDF-error comparison. u_tail places many more sites in the flagged window and lowers the u_tail separation index, yet utail/lhs hot CDF L2 is 1.03 at N1=20 and 1.01 at N1=15. Oracle remains better than LHS, so the problem is not that the window is impossible to fix; it is that raw u_tail allocation is not the right CDF-error optimizer for this G4a construction.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("G4b: Capacity-Limited Hotspot", styles, "h1"))
    ell_counts = ", ".join(f"{ell:g}: {int(count)}" for ell, count in res["b_ell_counts"].items())
    g4b_rows = [
        ["Quantity", "Result", "Interpretation"],
        [
            "Sampling alone",
            f"utail_fixed/lhs_fixed hot CDF L2 ratio {fmt(res['b_fixed_lhs_ratio'])}",
            "Adding targeted data does not remove the bias floor.",
        ],
        [
            "Model retuning",
            f"utail_retune/utail_fixed ratio {fmt(res['b_retune_fixed_ratio'])}, p={fmt(res['b_retune_fixed_p'])}",
            "Retuning ell_x gives about 9x improvement and passes the hard CDF-error gate.",
        ],
        [
            "CV selection",
            ell_counts,
            "CV usually selects a smaller length scale; the 0.05 selections explain remaining hard cases.",
        ],
        [
            "LOF separation",
            f"median hot LOF fixed {fmt(res['b_lof_fixed_med'], 1)} vs retune {fmt(res['b_lof_retune_med'], 1)}; P(fixed>10)={fmt(res['b_lof_fixed_gt10'])}; P(retune<10)={fmt(res['b_lof_retune_lt10'])}",
            "Absolute threshold 10 is too strict for the max statistic, but separation is strong.",
        ],
        [
            "BG-calibrated LOF",
            f"pooled background 99% threshold {fmt(res['b_bg99'], 1)}; P(fixed>thr)={fmt(res['b_lof_fixed_gt_bg99'])}; P(retune>thr)={fmt(res['b_lof_retune_gt_bg99'])}",
            "Use a relative or background-calibrated rule in paper text.",
        ],
        [
            "q_hat / coverage",
            f"q_hat fixed {fmt(res['b_q']['utail_fixed'])} -> retune {fmt(res['b_q']['utail_retune'])}; coverage {fmt(res['b_cov']['utail_fixed'])} -> {fmt(res['b_cov']['utail_retune'])}",
            "Retuning trades local bias for global variance; CP absorbs it through wider intervals.",
        ],
    ]
    story.append(styled_table(g4b_rows, styles, [1.18 * inch, 2.35 * inch, 3.67 * inch]))

    story.append(p("Advisor Discussion Points", styles, "h1"))
    story.append(
        bullet_list(
            [
                "The clean paper claim is not that targeted sampling always improves width at a fixed budget. The supported claim is a decision loop: u_tail flags an epistemic hotspot; targeted data tests whether it is data scarcity; LOF tells us when the bottleneck is model capacity.",
                "G4b should be the headline because it is falsifiable and strongly supported: data alone fails, retuning fixes the CDF error, and LOF explains the route without oracle quantities.",
                "G4a should be described as diagnostic/action evidence rather than CDF-error efficiency evidence. It supports targeting and diagnostic removal, but the CDF-error gain over LHS is absent at N1=60, N1=20, and N1=15.",
                "Recommended follow-up: if the advisor wants a clean positive C1 figure, revise the G4a DGP or allocation rule. A natural next design is flag the region with u_tail, then allocate more evenly within the flagged region, or construct a hotspot where the u_tail and CDF-error maxima are aligned.",
            ],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("Figure 1. G4a Diagnostics and Allocation", styles, "h1"))
    story.append(figure(G4A_FIG, 6.9 * inch, "G4a: u_tail targeting concentrates Stage-2 sites in the thinned window and reduces the u_tail separation index, but hotspot CDF L2 improves only marginally over LHS at N1=60.", styles))

    story.append(PageBreak())
    story.append(p("Figure 1b. G4a Small-Budget Check: N1=20", styles, "h1"))
    story.append(figure(G4A_N20_FIG, 6.9 * inch, "At N1=20, u_tail still allocates heavily into the window and lowers the u_tail diagnostic, but hot CDF L2 does not improve over LHS; the median utail/lhs ratio is 1.03.", styles))

    story.append(PageBreak())
    story.append(p("Figure 1c. G4a Small-Budget Check: N1=15", styles, "h1"))
    story.append(figure(G4A_N15_FIG, 6.9 * inch, "At N1=15, the same pattern remains: targeted allocation removes much of the diagnostic signal, but the hot CDF L2 median ratio is 1.01, so the effect-size gate is still not met.", styles))

    story.append(PageBreak())
    story.append(p("Figure 2. G4b Capacity-Limited Hotspot", styles, "h1"))
    story.append(figure(G4B_FIG, 7.0 * inch, "G4b: targeted sampling alone leaves the narrow-bump bias floor; site-level CV retuning to smaller ell_x sharply reduces hotspot CDF L2. LOF remains high for fixed models and flags the few retuned runs that remain misfit.", styles))

    story.append(p("Files", styles, "h1"))
    story.append(
        p(
            "Source script: experiments/framing_validation/gate4_fix_epistemic.py. Report generator: experiments/framing_validation/make_gate4_report.py. Metrics and figures: experiments/framing_validation/output_gate4a/, output_gate4a_n20/, output_gate4a_n15/, and output_gate4b/.",
            styles,
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUT_PDF


if __name__ == "__main__":
    path = build_pdf()
    print(path)
