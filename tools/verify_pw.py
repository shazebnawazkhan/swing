import asyncio
from playwright.async_api import async_playwright

async def main():
    url = "file:///D:/SNK/codes/repos/swing/outputs/comparison_report.html"
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width":1400,"height":900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(url, wait_until="networkidle", timeout=15000)
        await page.wait_for_timeout(1200)
        opts = await page.locator("#strat-sel option").count()
        await page.screenshot(path="shot_drop1.png")
        # Switch to EMA + Bollinger Bands (index 2)
        await page.select_option("#strat-sel", "2")
        await page.wait_for_timeout(1000)
        await page.screenshot(path="shot_drop2.png")
        # Expand first row with trades in panel-2
        togs = await page.locator("#panel-2 button.tog").count()
        if togs:
            await page.locator("#panel-2 button.tog").first.click()
            await page.wait_for_timeout(1500)
        eq = await page.locator("canvas[id^='eq-']").count()
        lc = await page.locator("div[id^='lc-']").count()
        await page.screenshot(path="shot_expanded.png", full_page=True)
        print(f"opts={opts}  eq_canvases={eq}  lc_divs={lc}  js_errors={errors or 'none'}")
        await browser.close()

asyncio.run(main())
