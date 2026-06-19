"""Render outputs/predictor.html in headless Chromium, screenshot, report JS errors."""
from pathlib import Path
from playwright.sync_api import sync_playwright

url = Path("outputs/predictor.html").resolve().as_uri()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 1000})
    msgs = []
    pg.on("console", lambda m: msgs.append(f"{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: msgs.append(f"PAGEERROR: {e}"))
    pg.goto(url)
    pg.wait_for_timeout(800)
    pg.screenshot(path="docs/pred_dash.png", full_page=True)
    rows = pg.eval_on_selector_all("#body tr", "els=>els.length")
    nmodels = pg.eval_on_selector_all(".mcard", "e=>e.length")
    print("rows:", rows, "model cards:", nmodels)
    print("kpis:", repr(pg.inner_text("#kpis")))
    print("banner:", repr(pg.inner_text("#banner")[:100]))
    print("console:", msgs[:6])
    b.close()
