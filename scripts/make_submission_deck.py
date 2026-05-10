from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "submission_assets"
OUT = OUT_DIR / "ROCmBench_Agent_Submission_Deck.pdf"

W, H = landscape((13.333 * inch, 7.5 * inch))
BG = colors.HexColor("#050506")
PANEL = colors.HexColor("#11151d")
PANEL_2 = colors.HexColor("#171d28")
RED = colors.HexColor("#ed1c24")
RED_2 = colors.HexColor("#ff4b3e")
AMBER = colors.HexColor("#ffb000")
CYAN = colors.HexColor("#00c7d4")
TEXT = colors.HexColor("#f7f7f8")
MUTED = colors.HexColor("#aab1c0")
GRID = colors.Color(1, 1, 1, alpha=0.12)


def draw_bg(c: canvas.Canvas, kicker: str, page: int) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setStrokeColor(colors.Color(1, 1, 1, alpha=0.06))
    c.setLineWidth(0.5)
    for x in range(0, int(W), 42):
        c.line(x, 0, x, H)
    for y in range(0, int(H), 42):
        c.line(0, y, W, y)
    c.setStrokeColor(colors.Color(237 / 255, 28 / 255, 36 / 255, alpha=0.55))
    c.setLineWidth(2)
    c.line(W * 0.55, H, W, H * 0.28)
    c.line(W * 0.67, H, W, H * 0.42)
    c.setFillColor(RED)
    c.rect(0, H - 5, W, 5, stroke=0, fill=1)
    c.setFillColor(RED_2)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(42, H - 36, kicker.upper())
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 42, 28, f"ROCmBench Agent  /  {page:02d}")


def title(c: canvas.Canvas, text: str, x: float = 42, y: float = 455, size: int = 36, width: float = 650) -> float:
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", size)
    lines = wrap(text, "Helvetica-Bold", size, width)
    for line in lines:
        c.drawString(x, y, line)
        y -= size * 1.08
    return y


def body(c: canvas.Canvas, text: str, x: float, y: float, size: int = 14, width: float = 520, leading: float | None = None) -> float:
    c.setFillColor(MUTED)
    c.setFont("Helvetica", size)
    leading = leading or size * 1.45
    for line in wrap(text, "Helvetica", size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def wrap(text: str, font: str, size: int, max_width: float) -> list[str]:
    out: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if stringWidth(candidate, font, size) <= max_width:
                line = candidate
            else:
                if line:
                    out.append(line)
                line = word
        if line:
            out.append(line)
    return out or [""]


def chip(c: canvas.Canvas, text: str, x: float, y: float, w: float | None = None) -> float:
    c.setFont("Helvetica-Bold", 9)
    w = w or stringWidth(text, "Helvetica-Bold", 9) + 22
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.08))
    c.setStrokeColor(colors.Color(1, 1, 1, alpha=0.20))
    c.roundRect(x, y - 17, w, 24, 3, stroke=1, fill=1)
    c.setFillColor(TEXT)
    c.drawString(x + 11, y - 10, text)
    return x + w + 10


def metric(c: canvas.Canvas, label: str, value: str, note: str, x: float, y: float, w: float = 205, h: float = 92) -> None:
    c.setFillColor(PANEL)
    c.setStrokeColor(colors.Color(1, 1, 1, alpha=0.18))
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    c.setFillColor(RED_2)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 16, y + h - 22, label.upper())
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(x + 16, y + 36, value)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(x + 16, y + 17, note)


def bullet(c: canvas.Canvas, text: str, x: float, y: float, width: float = 480) -> float:
    c.setFillColor(RED_2)
    c.circle(x, y + 4, 3, stroke=0, fill=1)
    return body(c, text, x + 16, y, 13, width, 18) - 8


def bar_chart(c: canvas.Canvas, x: float, y: float) -> None:
    data = [
        ("c16", 2689.79, 1.695),
        ("c24", 3392.42, 1.667),
        ("c32", 3423.27, 3.249),
    ]
    max_tok = 3600
    c.setStrokeColor(GRID)
    c.setFillColor(PANEL)
    c.roundRect(x, y, 520, 260, 5, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(TEXT)
    c.drawString(x + 24, y + 226, "Qwen2.5-7B: throughput vs tail latency")
    base = y + 52
    for i, (name, tok, p95) in enumerate(data):
        bx = x + 58 + i * 135
        bh = tok / max_tok * 145
        c.setFillColor(RED if name == "c24" else colors.HexColor("#5b1618"))
        c.rect(bx, base, 72, bh, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(bx + 36, base + bh + 12, f"{tok:,.0f}")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 10)
        c.drawCentredString(bx + 36, base - 18, name)
        c.setFillColor(AMBER)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(bx + 36, base + 165, f"p95 {p95:.3f}s")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(x + 24, y + 22, "Concurrency 32 adds only 0.9% throughput over c24, but p95 latency nearly doubles.")


def architecture(c: canvas.Canvas, x: float, y: float) -> None:
    labels = [
        ("HF model", "user selects model"),
        ("vLLM + ROCm", "auto-load on AMD"),
        ("sweep", "concurrency tests"),
        ("ROCm SMI", "GPU proof"),
        ("gate", "deploy / warn / block"),
    ]
    for i, (head, sub) in enumerate(labels):
        bx = x + i * 165
        c.setFillColor(PANEL if i != 4 else colors.HexColor("#2a0c0e"))
        c.setStrokeColor(RED_2 if i == 4 else colors.Color(1, 1, 1, alpha=0.18))
        c.roundRect(bx, y, 135, 80, 5, stroke=1, fill=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(bx + 13, y + 48, head)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(bx + 13, y + 28, sub)
        if i < len(labels) - 1:
            c.setStrokeColor(RED_2)
            c.setLineWidth(1.2)
            c.line(bx + 142, y + 40, bx + 158, y + 40)
            c.line(bx + 153, y + 46, bx + 158, y + 40)
            c.line(bx + 153, y + 34, bx + 158, y + 40)


def slide1(c: canvas.Canvas) -> None:
    draw_bg(c, "AMD Developer Hackathon", 1)
    x = 42
    for label in ["AMD Instinct", "ROCm", "vLLM", "Hugging Face", "Qwen"]:
        x = chip(c, label, x, H - 76)
    title(c, "ROCmBench Agent", 42, 420, 48, 720)
    body(c, "A deployment optimization agent for open-source LLMs on AMD GPUs. It turns raw benchmark runs into production-ready serving decisions.", 46, 310, 17, 660, 25)
    metric(c, "best 7B boundary", "c24", "3,392 tok/s @ 1.667s p95", 46, 120)
    metric(c, "GPU proof", "100%", "peak ROCm SMI utilization", 276, 120)
    metric(c, "decision", "gate", "deploy / warn / block", 506, 120)


def slide2(c: canvas.Canvas) -> None:
    draw_bg(c, "Problem", 2)
    title(c, "Running a model is easy. Trusting a config is not.", 42, 440, 34, 760)
    y = 300
    y = bullet(c, "Tokens/sec alone hides tail latency, error behavior, VRAM pressure, and cost.", 60, y, 620)
    y = bullet(c, "Manual benchmark notebooks are hard to repeat and weak as deployment evidence.", 60, y, 620)
    y = bullet(c, "Teams need a production gate: what should ship, what needs another run, and why.", 60, y, 620)
    metric(c, "wrong metric", "fastest", "is not always deployable", 760, 270, 260)
    metric(c, "right question", "trust?", "would this survive prod load", 760, 150, 260)


def slide3(c: canvas.Canvas) -> None:
    draw_bg(c, "System", 3)
    title(c, "The agent moves from model choice to deployment decision.", 42, 440, 32, 780)
    architecture(c, 52, 245)
    body(c, "The live path auto-loads the selected Hugging Face model on the AMD droplet, waits for /v1/models to confirm the exact model, then benchmarks only after the endpoint is valid.", 72, 160, 14, 800)


def slide4(c: canvas.Canvas) -> None:
    draw_bg(c, "Scoring", 4)
    title(c, "The score is built for production, not leaderboard screenshots.", 42, 440, 32, 820)
    labels = [
        ("throughput", "tokens/sec + request rate"),
        ("tail latency", "avg + p95 latency"),
        ("reliability", "error-rate penalty"),
        ("cost", "GPU-hour to $/1M tokens"),
        ("ROCm proof", "utilization + VRAM saturation"),
        ("decision", "deploy / warn / block"),
    ]
    x0, y0 = 56, 260
    for i, (head, sub) in enumerate(labels):
        x = x0 + (i % 3) * 295
        y = y0 - (i // 3) * 115
        metric(c, head, head.split()[0], sub, x, y, 250, 86)


def slide5(c: canvas.Canvas) -> None:
    draw_bg(c, "Boundary Found", 5)
    title(c, "Qwen 7B found the exact point where faster stopped meaning better.", 42, 445, 31, 800)
    bar_chart(c, 56, 120)
    c.setFillColor(PANEL_2)
    c.setStrokeColor(colors.Color(1, 1, 1, alpha=0.18))
    c.roundRect(620, 155, 340, 210, 5, stroke=1, fill=1)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(645, 315, "Agent decision")
    body(c, "Pick concurrency 24 for balanced production serving. Concurrency 32 barely improves throughput, but pushes p95 from 1.667s to 3.249s.", 645, 275, 14, 275, 21)


def slide6(c: canvas.Canvas) -> None:
    draw_bg(c, "Validation", 6)
    title(c, "The follow-up suite turned one benchmark into real deployment evidence.", 42, 445, 31, 850)
    rows = [
        ("Concurrency ceiling", "Qwen 7B", "c24 wins balanced", "3,392 tok/s"),
        ("8192 context", "Qwen 7B", "c16 remains stable", "1.523s p95"),
        ("Soak stability", "Qwen 7B", "0% errors", "2,689 tok/s"),
        ("Model tradeoff", "Qwen 3B", "higher throughput, worse p95", "4,219 tok/s"),
    ]
    x, y = 64, 320
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(MUTED)
    for i, h in enumerate(["test", "model", "finding", "proof"]):
        c.drawString(x + [0, 210, 380, 640][i], y + 34, h.upper())
    for r, row in enumerate(rows):
        yy = y - r * 52
        c.setStrokeColor(GRID)
        c.line(x, yy + 25, 930, yy + 25)
        for i, cell in enumerate(row):
            c.setFillColor(TEXT if i in {0, 3} else MUTED)
            c.setFont("Helvetica-Bold" if i in {0, 3} else "Helvetica", 12)
            c.drawString(x + [0, 210, 380, 640][i], yy, cell)


def slide7(c: canvas.Canvas) -> None:
    draw_bg(c, "Product", 7)
    title(c, "What ships in the app", 42, 445, 36, 720)
    y = 310
    for text in [
        "HF search + model selection with live vLLM auto-load.",
        "Concurrency sweeps and OpenAI-compatible endpoint benchmarks.",
        "ROCm SMI monitor parsing and GPU saturation gate.",
        "Side-by-side model/config comparison across saved and live runs.",
        "Generated deployment report and vLLM launch command.",
    ]:
        y = bullet(c, text, 64, y, 720)
    metric(c, "open source", "MIT", "GitHub-ready project", 760, 250, 230)
    metric(c, "stack", "AMD", "MI300X + ROCm + vLLM", 760, 135, 230)


def slide8(c: canvas.Canvas) -> None:
    draw_bg(c, "Close", 8)
    title(c, "Benchmarking is easy. Deployment decisions are the product.", 42, 430, 34, 820)
    body(c, "ROCmBench Agent turns AMD GPU experiments into a repeatable launch gate: load the model, test the configs, prove GPU behavior, and choose what should ship.", 52, 300, 16, 760, 24)
    x = 52
    for label in ["PagedAttention scoring", "speculative decoding tests", "multi-GPU tensor parallel", "CI regression gate"]:
        x = chip(c, label, x, 185)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(52, 90, "github.com/naksshhh/ROCmBench-Agent")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    for draw in [slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8]:
        draw(c)
        c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
