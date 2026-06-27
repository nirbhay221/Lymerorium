"""Frontend button test using Playwright headless browser."""
import sys
import asyncio
sys.stdout.reconfigure(encoding='utf-8')
from playwright.async_api import async_playwright

URL = "http://localhost:5000"
RESULTS = []

def log(status, name, detail=""):
    icon = "[OK]  " if status == "OK" else "[FAIL]" if status == "FAIL" else "[WARN]"
    line = f"{icon} {name}: {detail}"
    print(line)
    RESULTS.append((status, line))


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # ── Page load ─────────────────────────────────────────────────────────
        try:
            await page.goto(URL, timeout=10000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            log("OK", "Page load", URL)
        except Exception as e:
            log("FAIL", "Page load", str(e))
            await browser.close()
            return

        # ── Models dropdown populated ─────────────────────────────────────────
        await page.wait_for_timeout(3000)
        opts = await page.locator("#model-select option").all()
        model_val = opts[0].inner_text() if opts else "none"
        log("OK" if len(opts) > 0 else "FAIL", "Models dropdown",
            f"{len(opts)} option(s): {await opts[0].inner_text() if opts else 'none'}")

        # ── Mode toggle buttons ───────────────────────────────────────────────
        await page.click("#btn-swarm")
        swarm_cls = await page.locator("#btn-swarm").get_attribute("class")
        log("OK" if "active-swarm" in (swarm_cls or "") else "FAIL",
            "Swarm mode btn", f"class={swarm_cls}")

        await page.click("#btn-fast")
        fast_cls = await page.locator("#btn-fast").get_attribute("class")
        log("OK" if "active-fast" in (fast_cls or "") else "FAIL",
            "Fast mode btn", f"class={fast_cls}")

        # ── Gear / settings panel ─────────────────────────────────────────────
        await page.click("#gear-btn")
        await page.wait_for_timeout(300)
        bar_open = await page.locator("#settings-bar").is_visible()
        log("OK" if bar_open else "FAIL", "Settings panel", "opened" if bar_open else "did not open")
        if bar_open:
            await page.click("#gear-btn")  # close it

        # ── Detect button (fast mode, expects camera or error) ────────────────
        await page.click("#btn-fast")
        await page.click("button[onclick=\"analyze('detect')\"]")
        try:
            await page.wait_for_function(
                "document.getElementById('result').className.includes('done') || "
                "document.getElementById('result').className.includes('error')",
                timeout=90000
            )
            text = await page.locator("#result").inner_text()
            cls  = await page.locator("#result").get_attribute("class")
            if "No frame" in text:
                log("WARN", "Detect btn", "Camera offline - No frame available")
            elif "error" in (cls or ""):
                log("FAIL", "Detect btn", text[:100])
            else:
                log("OK", "Detect btn", text[:100])
        except Exception as e:
            log("FAIL", "Detect btn", f"Timeout - Gemma not responding: {str(e)[:80]}")

        # ── Send button (fast chat) ───────────────────────────────────────────
        await page.fill("#chat-input", "reply with just the word: PONG")
        await page.click("#send-btn")
        try:
            # wait for typing indicator to appear then disappear
            await page.wait_for_selector("#typing", timeout=5000)
            await page.wait_for_function(
                "!document.getElementById('typing')", timeout=90000
            )
            bubbles = page.locator(".msg.assistant .msg-bubble")
            count = await bubbles.count()
            last = await bubbles.nth(count - 1).inner_text()
            log("OK", "Send btn (chat)", last.strip()[:100])
        except Exception as e:
            log("FAIL", "Send btn (chat)", f"Timeout/error: {str(e)[:100]}")

        # ── Send button (swarm chat) ──────────────────────────────────────────
        await page.click("#btn-swarm")
        await page.fill("#chat-input", "what is 1 plus 1")
        await page.click("#send-btn")
        try:
            await page.wait_for_selector("#typing", timeout=5000)
            await page.wait_for_function(
                "!document.getElementById('typing')", timeout=90000
            )
            bubbles = page.locator(".msg.assistant .msg-bubble")
            count = await bubbles.count()
            last = await bubbles.nth(count - 1).inner_text()
            log("OK", "Send btn (swarm chat)", last.strip()[:100])
        except Exception as e:
            log("FAIL", "Send btn (swarm chat)", f"Timeout/error: {str(e)[:100]}")

        # ── Clear chat ────────────────────────────────────────────────────────
        await page.click("button.clear-btn")
        await page.wait_for_timeout(500)
        msg_count = await page.locator(".msg").count()
        log("OK" if msg_count == 1 else "WARN", "Clear chat btn", f"{msg_count} msg(s) after clear")

        await browser.close()

        print("\n" + "=" * 52)
        print("FRONTEND TEST SUMMARY")
        print("=" * 52)
        ok   = sum(1 for s, _ in RESULTS if s == "OK")
        warn = sum(1 for s, _ in RESULTS if s == "WARN")
        fail = sum(1 for s, _ in RESULTS if s == "FAIL")
        for _, line in RESULTS:
            print(line)
        print(f"\n{ok} OK  |  {warn} WARN  |  {fail} FAIL")


asyncio.run(run())
