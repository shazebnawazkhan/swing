import asyncio
from playwright.async_api import async_playwright

async def main():
    url = "file:///D:/SNK/codes/repos/swing/outputs/comparison_report.html"
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        await page.goto(url, wait_until="networkidle", timeout=15000)
        await page.select_option("#strat-sel", "2")
        await page.wait_for_timeout(800)
        await page.locator("#panel-2 button.tog").first.click()
        await page.wait_for_timeout(2500)

        info = await page.evaluate("""
        (() => {
            // eq-wrap div (chart container) — id starts with 'eq-' but NOT 'eq-stats-'
            const eqDiv = document.querySelector('.eq-wrap[id]');
            const lcDiv = document.querySelector('.lc-wrap[id]');
            return {
                eqId:     eqDiv ? eqDiv.id : null,
                eqH:      eqDiv ? eqDiv.clientHeight : -1,
                eqW:      eqDiv ? eqDiv.clientWidth  : -1,
                eqCanvas: eqDiv ? eqDiv.querySelectorAll('canvas').length : -1,
                lcId:     lcDiv ? lcDiv.id : null,
                lcH:      lcDiv ? lcDiv.clientHeight : -1,
                lcCanvas: lcDiv ? lcDiv.querySelectorAll('canvas').length : -1,
            };
        })()
        """)
        print("info:", info)
        print("page_errors:", errs)
        await page.screenshot(path="shot_debug.png", full_page=True)
        await browser.close()

asyncio.run(main())
