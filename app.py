import os
# Suppress OpenCV/FFmpeg verbose RTSP error messages before importing cv2
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
import cv2
import base64
import json
import requests
import threading
import time
import uuid
import sys
from flask import Flask, jsonify, Response, render_template_string
from flask_cors import CORS

# SwarmCore integration
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm_core"))
try:
    from swarm_core.api import run_simulation as swarm_run
    from swarm_core.background import BACKGROUND_SWARM
    from swarm_core.oracle_chat import answer as oracle_answer
    from swarm_core.config import get_llm_config as _swarm_get_cfg, set_llm_config as _swarm_set_cfg
    from swarm_core.config import VISION_LLM_URL, REASONING_LLM_URL
    from swarm_core.agents import circuit_status as _circuit_status
    SWARM_AVAILABLE = True
except Exception as _e:
    _swarm_get_cfg = _swarm_set_cfg = _circuit_status = None
    VISION_LLM_URL = REASONING_LLM_URL = None
    SWARM_AVAILABLE = False
    print(f"[SwarmCore] Not available: {_e}")

app = Flask(__name__)
CORS(app)

RTSP_URL = os.getenv("RTSP_URL", "")
LLAMA_URL = os.getenv("LLAMA_URL", "")
LLAMA_MODELS_URL = os.getenv("LLAMA_MODELS_URL", "")
NEO4J_HTTP_URL = os.getenv("NEO4J_HTTP_URL", "")
MODEL = "ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M"

latest_frame = None
frame_lock = threading.Lock()
chat_history = []  # [{role, content}]
_swarm_jobs: dict = {}  # job_id -> {status, result, started_at}


def capture_frames():
    global latest_frame
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            time.sleep(5)
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            with frame_lock:
                latest_frame = frame.copy()
        cap.release()
        time.sleep(1)


def get_frame_b64():
    with frame_lock:
        if latest_frame is None:
            return None
        _, buf = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode()


def _get_active_cfg() -> dict:
    if _swarm_get_cfg:
        return _swarm_get_cfg()
    return {"provider": "local", "base_url": LLAMA_URL, "api_key": "", "model": MODEL}


def _query_openai_compat(cfg: dict, prompt: str, image_b64, use_model: str) -> str:
    content = []
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_tokens": 768,
    }
    if cfg["provider"] == "local":
        payload["enable_thinking"] = False
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    resp = requests.post(cfg["base_url"], json=payload, headers=headers, timeout=120)
    msg = resp.json()["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def query_ollama(prompt, image_b64=None, model=None):
    if SWARM_AVAILABLE:
        BACKGROUND_SWARM.pause()
    try:
        cfg = _get_active_cfg()
        use_model = model or cfg["model"]
        return _query_openai_compat(cfg, prompt, image_b64, use_model)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if SWARM_AVAILABLE:
            BACKGROUND_SWARM.resume()


def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/video')
def video():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


_PROVIDER_MODELS = {
    "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    "groq":      ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
}


@app.route('/models')
def models():
    cfg = _get_active_cfg()
    if cfg["provider"] != "local":
        return jsonify({"models": _PROVIDER_MODELS.get(cfg["provider"], [cfg["model"]])})
    try:
        resp = requests.get(LLAMA_MODELS_URL, timeout=5)
        data = resp.json().get("data", [])
        names = [m["id"] for m in data]
        return jsonify({"models": names or [MODEL]})
    except Exception as e:
        return jsonify({"models": [MODEL], "error": str(e)})


@app.route('/detect')
def detect():
    from flask import request as req
    img = get_frame_b64()
    if not img:
        return jsonify({"error": "No frame available"})
    model = req.args.get("model")
    prompt = (
        "List every distinct object you can see in this image. "
        "For each object, state: its name, location (top-left, center, bottom-right, etc), "
        "and confidence (high/medium/low). Format as a numbered list. Be precise and thorough."
    )
    result = query_ollama(prompt, img, model)
    return jsonify({"result": result})


@app.route('/extract')
def extract():
    from flask import request as req
    img = get_frame_b64()
    if not img:
        return jsonify({"error": "No frame available"})
    model = req.args.get("model")
    prompt = (
        "Extract ALL text visible in this image exactly as it appears. "
        "Preserve formatting, line breaks, and spacing. "
        "If there is no text, say 'No text found'. Do not describe the image, only return the text."
    )
    result = query_ollama(prompt, img, model)
    return jsonify({"result": result})


@app.route('/describe')
def describe():
    from flask import request as req
    img = get_frame_b64()
    if not img:
        return jsonify({"error": "No frame available"})
    model = req.args.get("model")
    prompt = (
        "Describe this scene in detail. What is happening? "
        "What objects are present? What is the context or environment?"
    )
    result = query_ollama(prompt, img, model)
    return jsonify({"result": result})


@app.route('/chat', methods=['POST'])
def chat():
    from flask import request as req
    data = req.json or {}
    user_msg = data.get("message", "").strip()
    use_vision = data.get("vision", False)
    if not user_msg:
        return jsonify({"error": "Empty message"})

    img = get_frame_b64() if use_vision else None

    history_text = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}"
        for m in chat_history[-4:]
    ])

    prompt = ""
    if history_text:
        prompt += f"{history_text}\n"
    prompt += f"User: {user_msg}\nAssistant:"

    answer = query_ollama(prompt, img)
    if not answer.startswith("Error:"):
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": answer})
    return jsonify({"response": answer})


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    chat_history.clear()
    return jsonify({"ok": True})


@app.route('/swarm_analyze', methods=['POST'])
def swarm_analyze():
    """Start a swarm job in the background - returns job_id immediately.
    Poll /swarm_poll/<job_id> for status and final result."""
    from flask import request as req
    if not SWARM_AVAILABLE:
        return jsonify({"error": "SwarmCore not available - run: pip install langgraph langchain-core networkx"})

    data = req.get_json(force=True, silent=True) or {}
    topic = data.get("topic", "").strip()
    max_rounds = int(data.get("max_rounds", 1))

    if not topic:
        img = get_frame_b64()
        if not img:
            return jsonify({"error": "No frame available and no topic provided"})
        topic = query_ollama(
            "Describe exactly what you see in one sentence. Be specific about objects, actions, and context.",
            img
        )
        if topic.startswith("Error:") or topic.startswith("["):
            return jsonify({"error": f"LLM unavailable - cannot generate topic: {topic[:120]}"})

    img_b64 = get_frame_b64() or ""
    job_id = uuid.uuid4().hex[:8]
    started_at = time.time()
    cancel_event = threading.Event()
    stream_msgs: list = []
    _swarm_jobs[job_id] = {"status": "running", "result": None, "started_at": started_at,
                           "_cancel": cancel_event, "stream_messages": stream_msgs}

    def _run():
        if SWARM_AVAILABLE:
            BACKGROUND_SWARM.pause(wait_for_current=False)
        try:
            if cancel_event.is_set():
                _swarm_jobs[job_id]["status"] = "cancelled"
                _swarm_jobs[job_id]["result"] = {"error": "Cancelled before start"}
                return
            result = swarm_run(topic, max_rounds, image_b64=img_b64,
                               cancel_event=cancel_event, stream_sink=stream_msgs)
            if cancel_event.is_set():
                _swarm_jobs[job_id]["status"] = "cancelled"
                _swarm_jobs[job_id]["result"] = {"error": "Cancelled"}
                return
            result["response_time_s"] = round(time.time() - started_at, 1)
            _swarm_jobs[job_id]["status"] = "done"
            _swarm_jobs[job_id]["result"] = result
        except Exception as e:
            if cancel_event.is_set():
                _swarm_jobs[job_id]["status"] = "cancelled"
                _swarm_jobs[job_id]["result"] = {"error": "Cancelled"}
            else:
                _swarm_jobs[job_id]["status"] = "error"
                _swarm_jobs[job_id]["result"] = {"error": f"Swarm failed: {e}"}
        finally:
            if SWARM_AVAILABLE:
                BACKGROUND_SWARM.resume()
        if len(_swarm_jobs) > 12:
            try:
                oldest = min(_swarm_jobs, key=lambda k: _swarm_jobs[k]["started_at"])
                del _swarm_jobs[oldest]
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "started": True, "topic": topic})


@app.route('/swarm_poll/<job_id>')
def swarm_poll(job_id):
    """Poll for the status of a swarm job started by /swarm_analyze."""
    job = _swarm_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job not found"})
    elapsed = round(time.time() - job["started_at"], 1)
    # Safety timeout: mark as failed if thread died silently (MemoryError etc.)
    if job["status"] == "running" and elapsed > 3600:
        job["status"] = "error"
        job["result"] = {"error": "Swarm timed out after 60 minutes"}
    if job["status"] == "running":
        return jsonify({"status": "running", "elapsed_s": elapsed})
    return jsonify({"status": job["status"], "result": job.get("result"), "elapsed_s": elapsed})


@app.route('/swarm_cancel/<job_id>', methods=['POST'])
def swarm_cancel(job_id):
    job = _swarm_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found"})
    if job["status"] != "running":
        return jsonify({"ok": False, "error": f"Job already {job['status']}"})
    job["_cancel"].set()
    job["status"] = "cancelled"
    job["result"] = {"error": "Cancelled by user"}
    return jsonify({"ok": True})


@app.route('/swarm_stream/<job_id>')
def swarm_stream(job_id):
    """
    SSE endpoint - streams each agent turn as it finishes, then the final verdict.
    Connect with: new EventSource('/swarm_stream/<job_id>')
    Events: {type:'agent_turn', agent, content, round} | {type:'verdict', content} | {type:'done'}
    """
    def generate():
        sent = 0
        while True:
            job = _swarm_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'type': 'error', 'content': 'job not found'})}\n\n"
                return
            msgs = job.get("stream_messages", [])
            while sent < len(msgs):
                m = msgs[sent]
                if not m.get("error") and m.get("agent") not in ("SYSTEM",):
                    payload = {
                        "type": "agent_turn",
                        "agent": m.get("agent", ""),
                        "content": m.get("content", ""),
                        "round": m.get("round", 0),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                sent += 1
            status = job.get("status", "running")
            if status == "done":
                verdict = (job.get("result") or {}).get("verdict", "")
                if verdict:
                    yield f"data: {json.dumps({'type': 'verdict', 'content': verdict})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            if status in ("error", "cancelled"):
                err = (job.get("result") or {}).get("error", status)
                yield f"data: {json.dumps({'type': 'error', 'content': err})}\n\n"
                return
            time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/swarm_queue', methods=['POST'])
def swarm_queue_endpoint():
    """
    Inject a topic directly into the background swarm's debate queue.
    Body: {"topic": "..."}
    The background swarm will debate this topic on its next cycle (ahead of camera/RSS).
    """
    from flask import request as req
    if not SWARM_AVAILABLE:
        return jsonify({"error": "SwarmCore not available"})
    data = req.get_json(force=True, silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic required"})
    BACKGROUND_SWARM.push_topic(topic)
    return jsonify({"ok": True, "topic": topic,
                    "queue_size": BACKGROUND_SWARM._topic_queue.qsize()})


@app.route('/swarm_chat', methods=['POST'])
def swarm_chat():
    """Smart Synthesizer chat - answers from background memory or picks the right tool."""
    from flask import request as req
    if not SWARM_AVAILABLE:
        return jsonify({"error": "SwarmCore not available"})

    data = req.get_json(force=True, silent=True) or {}
    # Accept either "query" or "message" - frontend uses "query"
    query = (data.get("query") or data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "Empty message"})

    past = BACKGROUND_SWARM.search(query)
    img_b64 = get_frame_b64() or ""
    t0 = time.time()
    try:
        result = oracle_answer(query, past, image_b64=img_b64, background_swarm=BACKGROUND_SWARM)
    except Exception as e:
        return jsonify({"error": f"Swarm chat failed: {e}"})
    result["response_time_s"] = round(time.time() - t0, 1)
    return jsonify(result)


@app.route('/system_status')
def system_status():
    """Health check for both LLM endpoints, Neo4j, and circuit breakers."""
    import time as _t

    def _check(url: str, path: str = "/health", timeout: float = 3.0) -> bool:
        try:
            base = url.replace("/v1/chat/completions", "")
            r = requests.get(base + path, timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    vision_ok    = _check(VISION_LLM_URL or "")
    reasoning_ok = _check(REASONING_LLM_URL or "http://localhost:11434", path="/api/tags")

    neo4j_ok = False
    try:
        neo4j_ok = bool(requests.get("http://localhost:7474", timeout=2).status_code == 200)
    except Exception:
        try:
            neo4j_ok = bool(requests.get(NEO4J_HTTP_URL, timeout=2).status_code == 200 if NEO4J_HTTP_URL else False)
        except Exception:
            pass

    circuits = _circuit_status() if _circuit_status else {}

    return jsonify({
        "vision_llm": {
            "url":       VISION_LLM_URL,
            "reachable": vision_ok,
            "circuit":   circuits.get("vision", {}).get("circuit", "unknown"),
            "model":     circuits.get("vision", {}).get("model", ""),
        },
        "reasoning_llm": {
            "url":       REASONING_LLM_URL,
            "reachable": reasoning_ok,
            "circuit":   circuits.get("reasoning", {}).get("circuit", "unknown"),
            "model":     circuits.get("reasoning", {}).get("model", ""),
        },
        "neo4j":           {"reachable": neo4j_ok},
        "swarm_available": SWARM_AVAILABLE,
    })


@app.route('/swarm_status')
def swarm_status():
    """Returns background swarm health - verdicts, reputation, constitution, consolidation."""
    if not SWARM_AVAILABLE:
        return jsonify({"available": False})
    rep_stats = {}
    constitution_stats = {}
    consolidation_stats = {}
    try:
        from swarm_core.reputation import get_stats as _rep_stats
        rep_stats = _rep_stats()
    except Exception:
        pass
    try:
        from swarm_core.constitution import get_all as _con_all
        constitution_stats = {"rules": _con_all()}
    except Exception:
        pass
    try:
        from swarm_core.consolidation import CONSOLIDATED_MEMORY
        consolidation_stats = {
            "insight_count": CONSOLIDATED_MEMORY.count,
            "insights": CONSOLIDATED_MEMORY.get_all(),
        }
    except Exception:
        pass
    return jsonify({
        "available": True,
        **BACKGROUND_SWARM.status,
        "reputation": rep_stats,
        "constitution": constitution_stats,
        "consolidation": consolidation_stats,
    })


@app.route('/set_llm_config', methods=['POST'])
def set_llm_config_route():
    from flask import request as req
    data = req.get_json(force=True, silent=True) or {}
    provider = data.get("provider", "local")
    api_key = data.get("api_key", "")
    model = data.get("model", "")
    if _swarm_set_cfg:
        _swarm_set_cfg(provider, api_key, model)
    return jsonify({"ok": True, "provider": provider})


@app.route('/get_llm_config')
def get_llm_config_route():
    cfg = _get_active_cfg()
    return jsonify({
        "provider": cfg.get("provider", "local"),
        "model":    cfg.get("model", MODEL),
        "has_key":  bool(cfg.get("api_key")),
    })


@app.route('/test_llm_config')
def test_llm_config_route():
    result = query_ollama("Say 'connection ok' and nothing else.")
    if result.startswith("Error:"):
        return jsonify({"ok": False, "error": result})
    return jsonify({"ok": True, "response": result[:120]})


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TypeHand Vision</title>
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg:        #000000;
    --bg2:       #1c1c1e;
    --bg3:       #2c2c2e;
    --bg4:       #3a3a3c;
    --label:     #ffffff;
    --label2:    rgba(235,235,245,0.6);
    --label3:    rgba(235,235,245,0.3);
    --sep:       rgba(84,84,88,0.55);
    --blue:      #0a84ff;
    --green:     #30d158;
    --red:       #ff453a;
    --orange:    #ff9f0a;
    --radius:    14px;
    --radius-sm: 10px;
  }

  body {
    background: var(--bg);
    color: var(--label);
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Titlebar ── */
  .titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: rgba(28,28,30,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--sep);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .titlebar-left { display: flex; align-items: center; gap: 12px; }
  .app-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #1c6ef5 0%, #0a84ff 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
  }
  .app-title { font-size: 1rem; font-weight: 600; letter-spacing: -0.2px; }
  .app-sub { font-size: 0.75rem; color: var(--label2); margin-top: 1px; }

  .titlebar-right { display: flex; align-items: center; gap: 8px; }
  .live-badge {
    display: flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 20px;
    background: rgba(48,209,88,0.15);
    border: 1px solid rgba(48,209,88,0.3);
    font-size: 0.72rem; font-weight: 600; color: var(--green);
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    animation: pulse 1.8s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }

  /* Model picker */
  .model-wrap { position: relative; }
  .model-select {
    appearance: none;
    background: var(--bg3);
    border: 1px solid var(--sep);
    color: var(--label);
    font-size: 0.8rem;
    font-family: inherit;
    padding: 6px 28px 6px 10px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    outline: none;
    transition: border-color .15s;
  }
  .model-select:focus { border-color: var(--blue); }
  .model-arrow {
    position: absolute; right: 8px; top: 50%;
    transform: translateY(-50%);
    pointer-events: none; color: var(--label2); font-size: 0.65rem;
  }

  /* ── Layout ── */
  .layout {
    display: grid;
    grid-template-columns: 1fr 380px;
    grid-template-rows: auto 1fr;
    gap: 16px;
    padding: 20px 24px;
    max-width: 1400px;
    margin: 0 auto;
    min-height: calc(100vh - 65px);
  }

  /* ── Cards ── */
  .card {
    background: var(--bg2);
    border-radius: var(--radius);
    border: 1px solid var(--sep);
    overflow: hidden;
  }
  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--sep);
    display: flex; align-items: center; justify-content: space-between;
  }
  .card-title {
    font-size: 0.78rem; font-weight: 600;
    color: var(--label2); text-transform: uppercase; letter-spacing: 0.6px;
  }

  /* ── Video ── */
  .video-card {
    grid-column: 1 / 2;
    grid-row: 1 / 3;
    display: flex; flex-direction: column;
  }
  .video-wrap {
    flex: 1; background: #000; position: relative; overflow: hidden;
    min-height: 360px;
  }
  .video-wrap img {
    width: 100%; height: 100%; object-fit: cover; display: block;
  }
  .video-overlay {
    position: absolute; bottom: 12px; left: 12px;
    display: flex; gap: 8px;
  }
  .chip {
    padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 500;
    background: rgba(0,0,0,0.6); color: var(--label2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.1);
  }

  /* ── Actions card ── */
  .actions-card { grid-column: 2 / 3; grid-row: 1 / 2; }
  .actions-body { padding: 14px; display: flex; flex-direction: column; gap: 10px; }

  /* ── Mode toggle ── */
  .mode-toggle {
    display: flex; border-radius: var(--radius-sm);
    border: 1px solid var(--sep); overflow: hidden;
    margin-bottom: 4px;
  }
  .mode-btn {
    flex: 1; padding: 7px 0; font-size: 0.75rem; font-weight: 600;
    border: none; cursor: pointer; font-family: inherit;
    transition: background .15s, color .15s;
    background: transparent; color: var(--label2);
  }
  .mode-btn:hover:not(.active-fast):not(.active-swarm) { color: var(--label); background: rgba(255,255,255,0.05); }
  .mode-btn.active-fast { background: var(--blue); color: #fff; }
  .mode-btn.active-swarm { background: rgba(191,90,242,0.85); color: #fff; }
  .mode-label {
    font-size: 0.72rem; color: var(--label2); text-align: center;
    margin-top: -4px; margin-bottom: 6px;
  }
  .swarm-badge {
    display: inline-block; font-size: 0.6rem; font-weight: 700;
    background: rgba(191,90,242,0.25); color: rgba(191,90,242,0.9);
    border: 1px solid rgba(191,90,242,0.4);
    border-radius: 4px; padding: 1px 5px; margin-left: 6px;
    vertical-align: middle; letter-spacing: 0.3px;
  }

  /* Swarm result styles */
  .swarm-verdict {
    background: rgba(191,90,242,0.12); border: 1px solid rgba(191,90,242,0.3);
    border-radius: var(--radius-sm); padding: 12px; margin-bottom: 10px;
    font-size: 0.85rem; line-height: 1.65; white-space: pre-wrap;
  }
  .swarm-agent-line { font-size: 0.78rem; margin-bottom: 6px; line-height: 1.5; }
  .swarm-agent-name { font-weight: 700; }

  .action-btn {
    width: 100%;
    display: flex; align-items: center; gap: 12px;
    padding: 13px 16px;
    border: none; border-radius: var(--radius-sm);
    font-family: inherit; font-size: 0.9rem; font-weight: 500;
    cursor: pointer;
    transition: opacity .15s, transform .1s;
    text-align: left;
  }
  .action-btn:active { transform: scale(0.98); }
  .action-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  .action-btn .btn-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
  }
  .action-btn .btn-text { display: flex; flex-direction: column; }
  .action-btn .btn-label { font-weight: 600; font-size: 0.88rem; }
  .action-btn .btn-desc { font-size: 0.72rem; opacity: 0.65; margin-top: 1px; }

  .btn-detect { background: rgba(10,132,255,0.18); color: #5ac8fa; }
  .btn-detect .btn-icon { background: rgba(10,132,255,0.25); }
  .btn-detect:hover:not(:disabled) { background: rgba(10,132,255,0.26); }

  .btn-extract { background: rgba(48,209,88,0.15); color: #32d74b; }
  .btn-extract .btn-icon { background: rgba(48,209,88,0.22); }
  .btn-extract:hover:not(:disabled) { background: rgba(48,209,88,0.23); }

  .btn-describe { background: rgba(255,159,10,0.15); color: #ffd60a; }
  .btn-describe .btn-icon { background: rgba(255,159,10,0.22); }
  .btn-describe:hover:not(:disabled) { background: rgba(255,159,10,0.23); }

  /* ── Result card ── */
  .result-card { grid-column: 2 / 3; grid-row: 2 / 3; display: flex; flex-direction: column; }
  .result-body {
    flex: 1; padding: 16px;
    font-size: 0.875rem; line-height: 1.75;
    white-space: pre-wrap; color: var(--label2);
    min-height: 160px;
    overflow-y: auto;
  }
  .result-body.loading { color: var(--blue); font-style: italic; }
  .result-body.error { color: var(--red); }
  .result-body.done { color: var(--label); }
  .result-footer {
    padding: 8px 16px;
    border-top: 1px solid var(--sep);
    display: flex; justify-content: space-between; align-items: center;
  }
  .result-badge {
    display: inline-block; padding: 2px 9px; border-radius: 5px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .badge-detect { background: rgba(10,132,255,0.2); color: #5ac8fa; }
  .badge-extract { background: rgba(48,209,88,0.18); color: #32d74b; }
  .badge-describe { background: rgba(255,159,10,0.18); color: #ffd60a; }
  .latency-text { font-size: 0.72rem; color: var(--label3); }

  /* ── Chat card ── */
  .chat-card {
    grid-column: 1 / 3;
    display: flex; flex-direction: column;
    max-height: 420px;
  }
  .chat-header-row {
    display: flex; align-items: center; justify-content: space-between;
  }
  .chat-messages {
    flex: 1; overflow-y: auto; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg { display: flex; flex-direction: column; gap: 3px; max-width: 75%; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.assistant { align-self: flex-start; align-items: flex-start; }
  .msg-bubble {
    padding: 10px 14px; border-radius: 18px;
    font-size: 0.875rem; line-height: 1.6; white-space: pre-wrap;
  }
  .msg.user .msg-bubble {
    background: var(--blue); color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant .msg-bubble {
    background: var(--bg3); color: var(--label);
    border-bottom-left-radius: 4px;
  }
  .msg-meta { font-size: 0.68rem; color: var(--label3); padding: 0 4px; }
  .vision-ctx {
    font-size: 0.72rem; color: var(--orange); opacity: 0.8;
    padding: 0 4px; font-style: italic;
  }
  .chat-input-row {
    display: flex; gap: 8px; padding: 12px 16px;
    border-top: 1px solid var(--sep); align-items: flex-end;
  }
  .chat-input {
    flex: 1; background: var(--bg3); border: 1px solid var(--sep);
    color: var(--label); font-family: inherit; font-size: 0.9rem;
    padding: 10px 14px; border-radius: var(--radius-sm);
    outline: none; resize: none; min-height: 42px; max-height: 120px;
    transition: border-color .15s;
    line-height: 1.4;
  }
  .chat-input:focus { border-color: var(--blue); }
  .chat-input::placeholder { color: var(--label3); }
  .vision-toggle {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.75rem; color: var(--label2);
    cursor: pointer; user-select: none; flex-shrink: 0;
    padding: 10px 0;
  }
  .vision-toggle input { accent-color: var(--orange); width: 14px; height: 14px; cursor: pointer; }
  .send-btn {
    padding: 10px 18px; border: none; border-radius: var(--radius-sm);
    background: var(--blue); color: #fff; font-family: inherit;
    font-size: 0.875rem; font-weight: 600; cursor: pointer;
    transition: opacity .15s, transform .1s; flex-shrink: 0;
  }
  .send-btn:hover { opacity: 0.85; }
  .send-btn:active { transform: scale(0.97); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .clear-btn {
    padding: 4px 10px; border: 1px solid var(--sep);
    border-radius: 6px; background: transparent;
    color: var(--label3); font-size: 0.72rem; cursor: pointer;
    font-family: inherit; transition: color .15s;
  }
  .clear-btn:hover { color: var(--red); border-color: var(--red); }
  .typing-indicator { display: flex; gap: 4px; padding: 10px 14px; }
  .typing-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--label3);
    animation: bounce 1.2s ease-in-out infinite;
  }
  .typing-dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

  /* ── Settings panel ── */
  .gear-btn {
    padding: 6px 9px; border: 1px solid var(--sep);
    border-radius: var(--radius-sm); background: transparent;
    color: var(--label2); font-size: 0.85rem; cursor: pointer;
    font-family: inherit; line-height: 1; transition: color .15s, border-color .15s;
  }
  .gear-btn:hover { color: var(--label); border-color: var(--label2); }
  .gear-btn.active { color: var(--blue); border-color: var(--blue); }
  .settings-bar {
    display: none; padding: 10px 24px; gap: 10px;
    align-items: center; flex-wrap: wrap;
    background: rgba(28,28,30,0.97);
    border-bottom: 1px solid var(--sep);
    backdrop-filter: blur(20px);
  }
  .settings-bar.open { display: flex; }
  .cfg-label { font-size: 0.73rem; color: var(--label2); white-space: nowrap; }
  .cfg-select, .cfg-input {
    background: var(--bg3); border: 1px solid var(--sep);
    color: var(--label); font-family: inherit; font-size: 0.8rem;
    padding: 5px 9px; border-radius: var(--radius-sm); outline: none;
  }
  .cfg-select:focus, .cfg-input:focus { border-color: var(--blue); }
  .cfg-input { width: 200px; }
  .cfg-input.wide { width: 220px; }
  .cfg-btn {
    padding: 5px 13px; border: none; border-radius: var(--radius-sm);
    font-family: inherit; font-size: 0.8rem; font-weight: 600; cursor: pointer;
    transition: opacity .15s;
  }
  .cfg-btn:hover { opacity: 0.85; }
  .cfg-btn.save { background: var(--blue); color: #fff; }
  .cfg-btn.test { background: var(--bg4); color: var(--label); border: 1px solid var(--sep); }
  .cfg-status { font-size: 0.73rem; }
  .cfg-status.ok  { color: var(--green); }
  .cfg-status.err { color: var(--red); }
</style>
</head>
<body>

<div class="titlebar">
  <div class="titlebar-left">
    <div class="app-icon"></div>
    <div>
      <div class="app-title">TypeHand Vision</div>
      <div class="app-sub">Jetson Orin Nano Super</div>
    </div>
  </div>
  <div class="titlebar-right">
    <div class="live-badge"><span class="live-dot"></span>Live</div>
    <div class="model-wrap">
      <select class="model-select" id="model-select">
        <option value="moondream">moondream</option>
      </select>
      <span class="model-arrow">&#9660;</span>
    </div>
    <button class="gear-btn" id="gear-btn" onclick="toggleSettings()" title="LLM Provider Settings">&#9881;</button>
  </div>
</div>

<div class="settings-bar" id="settings-bar">
  <span class="cfg-label">Provider</span>
  <select class="cfg-select" id="cfg-provider" onchange="onProviderChange()">
    <option value="local">Local Jetson</option>
    <option value="openai">OpenAI</option>
    <option value="groq">&#9889; Groq</option>
  </select>
  <span class="cfg-label" id="cfg-key-label">API Key</span>
  <input class="cfg-input" id="cfg-key" type="password" placeholder="sk-… or API key">
  <span class="cfg-label">Model</span>
  <input class="cfg-input wide" id="cfg-model" type="text" placeholder="gpt-4o-mini">
  <button class="cfg-btn save" onclick="saveLLMConfig()">Save</button>
  <button class="cfg-btn test" onclick="testLLMConfig()">Test</button>
  <span class="cfg-status" id="cfg-status"></span>
</div>

<div class="layout">
  <div class="card video-card">
    <div class="card-header">
      <span class="card-title">Live Feed</span>
      <span class="card-title" style="color:var(--label3)">Logitech Brio 101</span>
    </div>
    <div class="video-wrap">
      <img src="/video" alt="Live stream">
      <div class="video-overlay">
        <span class="chip">RTSP &rarr; :8554</span>
        <span class="chip">640x480</span>
      </div>
    </div>
  </div>

  <div class="card actions-card">
    <div class="card-header">
      <span class="card-title">Actions</span>
      <span class="card-title" id="mode-indicator" style="color:var(--blue)">&#9889; Fast</span>
    </div>
    <div class="actions-body">
      <div class="mode-toggle">
        <button class="mode-btn active-fast" id="btn-fast" onclick="setMode('fast')">&#9889; Fast</button>
        <button class="mode-btn" id="btn-swarm" onclick="setMode('swarm')">Swarm</button>
      </div>
      <div class="mode-label" id="mode-label">Single Gemma call &mdash; instant result</div>
      <button class="action-btn btn-detect" onclick="analyze('detect')">
        <div class="btn-icon"></div>
        <div class="btn-text">
          <span class="btn-label">Detect Objects</span>
          <span class="btn-desc">Find and locate all objects</span>
        </div>
      </button>
      <button class="action-btn btn-extract" onclick="analyze('extract')">
        <div class="btn-icon"></div>
        <div class="btn-text">
          <span class="btn-label">Extract Text</span>
          <span class="btn-desc">OCR - read all visible text</span>
        </div>
      </button>
      <button class="action-btn btn-describe" onclick="analyze('describe')">
        <div class="btn-icon"></div>
        <div class="btn-text">
          <span class="btn-label">Describe Scene</span>
          <span class="btn-desc">Full scene understanding</span>
        </div>
      </button>
    </div>
  </div>

  <div class="card result-card">
    <div class="card-header">
      <span class="card-title">Output</span>
      <span id="result-badge"></span>
    </div>
    <div class="result-body" id="result">Select an action to analyze the current frame.</div>
    <div class="result-footer">
      <span class="latency-text" id="latency"></span>
    </div>
  </div>

  <div class="card chat-card">
    <div class="card-header">
      <div class="chat-header-row" style="width:100%">
        <span class="card-title">Chat &mdash; gemma4:e2b</span>
        <button class="clear-btn" onclick="clearChat()">Clear</button>
      </div>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="msg assistant">
        <div class="msg-bubble">Hi! I'm gemma4. Ask me anything about the scene, or toggle "See camera" to have me look at what the camera sees right now.</div>
        <div class="msg-meta">gemma4:e2b</div>
      </div>
    </div>
    <div class="chat-input-row">
      <textarea class="chat-input" id="chat-input" placeholder="Ask about the scene…" rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"
        oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px'"></textarea>
      <label class="vision-toggle">
        <input type="checkbox" id="vision-toggle"> See camera
      </label>
      <button class="send-btn" id="send-btn" onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<script>
let currentMode = 'fast';

const BTN_DESCS = {
  detect:  ['Find and locate all objects',          'Swarm debates what objects reveal'],
  extract: ['OCR - read all visible text',          'Swarm debates what the text means'],
  describe:['Full scene understanding',             'Swarm debates the scene in depth'],
};

function setMode(mode) {
  currentMode = mode;
  const fastBtn   = document.getElementById('btn-fast');
  const swarmBtn  = document.getElementById('btn-swarm');
  const label     = document.getElementById('mode-label');
  const indicator = document.getElementById('mode-indicator');

  fastBtn.className  = 'mode-btn' + (mode === 'fast'  ? ' active-fast'  : '');
  swarmBtn.className = 'mode-btn' + (mode === 'swarm' ? ' active-swarm' : '');

  const swarm = mode === 'swarm';
  if (swarm) {
    label.innerHTML = 'Swarm mode - deep debate (~10-20 min on Jetson)';
    indicator.innerHTML = 'Swarm';
    indicator.style.color = 'rgba(191,90,242,0.9)';
  } else {
    label.innerHTML = '&#9889; Fast mode - single Gemma call, instant result';
    indicator.innerHTML = '&#9889; Fast';
    indicator.style.color = 'var(--blue)';
  }

  // Update action button descriptions + add SWARM badge
  for (const [type, [fastDesc, swarmDesc]] of Object.entries(BTN_DESCS)) {
    const descEl = document.querySelector(`[onclick="analyze('${type}')"] .btn-desc`);
    const labelEl = document.querySelector(`[onclick="analyze('${type}')"] .btn-label`);
    if (!descEl || !labelEl) continue;
    descEl.textContent = swarm ? swarmDesc : fastDesc;
    // toggle SWARM badge on button label
    const existing = labelEl.querySelector('.swarm-badge');
    if (swarm && !existing) {
      const badge = document.createElement('span');
      badge.className = 'swarm-badge';
      badge.textContent = 'SWARM';
      labelEl.appendChild(badge);
    } else if (!swarm && existing) {
      existing.remove();
    }
  }
}

async function loadModels() {
  try {
    const res = await fetch('/models');
    const data = await res.json();
    const sel = document.getElementById('model-select');
    if (data.models && data.models.length) {
      sel.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
    }
  } catch(e) {}
}

function getModel() {
  return document.getElementById('model-select').value;
}

async function analyze(type) {
  if (currentMode === 'swarm') {
    await analyzeSwarm(type);
    return;
  }

  const box    = document.getElementById('result');
  const badge  = document.getElementById('result-badge');
  const latency = document.getElementById('latency');

  const meta = {
    detect:   ['Detecting objects…', 'Detect',   'badge-detect'],
    extract:  ['Extracting text…',   'Text OCR', 'badge-extract'],
    describe: ['Describing scene…',  'Describe', 'badge-describe'],
  };
  const [msg, label, cls] = meta[type];

  document.querySelectorAll('.action-btn').forEach(b => b.disabled = true);
  box.className = 'result-body loading';
  box.textContent = msg;
  badge.innerHTML = '';
  latency.textContent = '';

  const start = Date.now();
  try {
    const res  = await fetch('/' + type + '?model=' + encodeURIComponent(getModel()));
    const data = await res.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    box.className = 'result-body' + (data.error ? ' error' : ' done');
    box.textContent = data.result || data.error;
    badge.innerHTML = '<span class="result-badge ' + cls + '">' + label + '</span>';
    latency.textContent = elapsed + 's';
  } catch(e) {
    box.className = 'result-body error';
    box.textContent = 'Connection error: ' + e.message;
  }
  document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
}

async function analyzeSwarm(type) {
  const box    = document.getElementById('result');
  const badge  = document.getElementById('result-badge');
  const latency = document.getElementById('latency');

  document.querySelectorAll('.action-btn').forEach(b => b.disabled = true);
  box.className = 'result-body loading';
  badge.innerHTML = '';
  latency.textContent = '';

  // Step 1: get scene description to use as swarm topic
  box.textContent = 'Step 1/2 - Gemma describing scene…';
  let topic = '';
  try {
    const r = await fetch('/' + type + '?model=' + encodeURIComponent(getModel()));
    const d = await r.json();
    topic = d.result || d.error || '';
  } catch(e) {
    box.className = 'result-body error';
    box.textContent = 'Scene capture failed: ' + e.message;
    document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
    return;
  }

  // Step 2: start swarm job in background - get a job_id immediately
  let jobId = null;
  try {
    const res = await fetch('/swarm_analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topic: topic, max_rounds: 1})
    });
    const data = await res.json();
    if (data.error) {
      box.className = 'result-body error';
      box.textContent = data.error;
      document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
      return;
    }
    jobId = data.job_id;
  } catch(e) {
    box.className = 'result-body error';
    box.textContent = 'Swarm start failed: ' + e.message;
    document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
    return;
  }

  // Step 3: stream agent turns live via SSE - show Stop button
  const jobStart = Date.now();
  badge.innerHTML = '<button id="swarm-stop-btn" onclick="stopSwarm(\'' + jobId + '\')" style="'
    + 'padding:3px 12px;border:1px solid rgba(255,69,58,0.6);border-radius:6px;background:rgba(255,69,58,0.15);'
    + 'color:#ff453a;font-size:0.72rem;font-weight:600;cursor:pointer;font-family:inherit;">&#9632; Stop</button>';

  const agentColors = {
    Skeptic:'#ff6b6b', Visionary:'#5ac8fa', Realist:'#32d74b',
    Ethicist:'#ffd60a', Technologist:'#0a84ff', Economist:'#ff9f0a',
    Contrarian:'#ff453a', Synthesizer:'#64d2ff'
  };

  box.className = 'result-body done';
  box.innerHTML = '<div style="font-size:0.72rem;color:var(--label3);margin-bottom:8px">Step 2/2 - agents debating live…</div>';

  const sse = new EventSource('/swarm_stream/' + jobId);
  let turnCount = 0;

  sse.onmessage = function(e) {
    const data = JSON.parse(e.data);
    if (data.type === 'agent_turn') {
      const color = agentColors[data.agent] || '#fff';
      const esc = s => (s||'').replace(/</g,'&lt;').replace(/\n/g,'<br>');
      const div = document.createElement('div');
      div.className = 'swarm-agent-line';
      div.innerHTML = '<span class="swarm-agent-name" style="color:' + color + '">[' + data.agent + ']</span> ' + esc(data.content);
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
      turnCount++;
      const elapsed = ((Date.now() - jobStart) / 1000).toFixed(0);
      latency.textContent = elapsed + 's - ' + turnCount + ' turns so far';
    } else if (data.type === 'verdict') {
      const vdiv = document.createElement('div');
      vdiv.className = 'swarm-verdict';
      vdiv.style.marginTop = '12px';
      vdiv.innerHTML = (data.content||'').replace(/</g,'&lt;').replace(/\n/g,'<br>');
      box.appendChild(vdiv);
      box.scrollTop = box.scrollHeight;
    } else if (data.type === 'done') {
      sse.close();
      document.getElementById('swarm-stop-btn')?.remove();
      const elapsed = ((Date.now() - jobStart) / 1000).toFixed(1);
      badge.innerHTML = '<span class="result-badge" style="background:rgba(191,90,242,0.2);color:rgba(191,90,242,0.9)">Swarm</span>';
      latency.textContent = elapsed + 's - ' + turnCount + ' agent turns';
      document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
    } else if (data.type === 'error') {
      sse.close();
      document.getElementById('swarm-stop-btn')?.remove();
      box.className = 'result-body error';
      box.textContent = data.content || 'Swarm error';
      badge.innerHTML = '';
      document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
    }
  };

  sse.onerror = function() {
    sse.close();
    document.getElementById('swarm-stop-btn')?.remove();
    document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
  };
}

async function stopSwarm(jobId) {
  const btn = document.getElementById('swarm-stop-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Stopping…'; }
  try {
    await fetch('/swarm_cancel/' + jobId, {method: 'POST'});
  } catch(e) {}
}

function renderSwarmResult(data) {
  const esc = s => s.split('<').join('&lt;').split('\\n').join('<br>');
  let html = '<div class="swarm-verdict">' + esc(data.verdict) + '</div>';

  if (data.cache_hit) {
    const validators = (data.cache_validators || []).join(', ');
    const votes = data.cache_change_votes != null ? data.cache_change_votes : '?';
    html += '<div style="font-size:0.72rem;color:var(--green);margin-top:8px">';
    html += 'OK Served from memory - ' + validators + ' confirmed (' + votes + '/3 detected change)';
    html += '</div>';
    return html;
  }

  const agentColors = {
    Skeptic:'#ff6b6b', Visionary:'#5ac8fa', Realist:'#32d74b',
    Ethicist:'#ffd60a', Technologist:'#0a84ff', Economist:'#ff9f0a',
    Contrarian:'#ff453a', Synthesizer:'#64d2ff'
  };
  const msgs = data.messages || [];
  if (msgs.length) {
    html += '<div style="font-size:0.72rem;color:var(--label3);margin-bottom:8px">Debate transcript (' + msgs.length + ' turns):</div>';
    for (const m of msgs) {
      const color = agentColors[m.agent] || '#fff';
      html += '<div class="swarm-agent-line">';
      html += '<span class="swarm-agent-name" style="color:' + color + '">[' + m.agent + ']</span> ';
      html += esc(m.content);
      html += '</div>';
    }
  }
  return html;
}

loadModels();

function scrollChat() {
  const el = document.getElementById('chat-messages');
  el.scrollTop = el.scrollHeight;
}

function appendMsg(role, content, meta) {
  const box = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const safe = content.split('<').join('&lt;').split('\\n').join('<br>');
  div.innerHTML =
    '<div class="msg-bubble">' + safe + '</div>' +
    '<div class="msg-meta">' + (meta || role) + '</div>';
  box.appendChild(div);
  scrollChat();
  return div;
}

function appendTyping() {
  const box = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.id = 'typing';
  div.innerHTML = '<div class="msg-bubble typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
  box.appendChild(div);
  scrollChat();
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('send-btn');
  const useVision = document.getElementById('vision-toggle').checked;
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  input.style.height = 'auto';
  btn.disabled = true;

  const now = new Date();
  const ts = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  appendMsg('user', msg, 'You &middot; ' + ts);
  appendTyping();

  const start = Date.now();
  const useSwarm = (currentMode === 'swarm');
  const endpoint = useSwarm ? '/swarm_chat' : '/chat';
  const body = useSwarm
    ? {query: msg, vision: useVision}
    : {message: msg, vision: useVision};

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    document.getElementById('typing')?.remove();

    if (data.error) {
      appendMsg('assistant', 'Error: ' + data.error,
                (useSwarm ? 'swarm' : 'gemma4:e2b') + ' &middot; ' + elapsed + 's');
    } else if (useSwarm) {
      // Tier 3 preliminary: prepend a banner so users know a fuller answer is coming
      let body = data.answer || '';
      if (data.note) {
        body = 'Working ' + data.note + '\\n\\nNote Preliminary answer:\\n\\n' + body;
      }
      const meta = [
        data.tier || 'swarm',
        (data.agents_used || []).join('+') || 'oracle',
        elapsed + 's'
      ].filter(Boolean).join(' &middot; ');
      appendMsg('assistant', body, meta);
    } else {
      appendMsg('assistant', data.response, 'gemma4:e2b &middot; ' + elapsed + 's');
    }
  } catch(e) {
    document.getElementById('typing')?.remove();
    appendMsg('assistant', 'Connection error: ' + e.message,
              useSwarm ? 'swarm' : 'gemma4:e2b');
  }
  btn.disabled = false;
  input.focus();
}

async function clearChat() {
  await fetch('/clear_chat', {method:'POST'});
  const box = document.getElementById('chat-messages');
  box.innerHTML = '<div class="msg assistant"><div class="msg-bubble">Chat cleared. Ask me anything!</div><div class="msg-meta">gemma4:e2b</div></div>';
}

// ── LLM Provider Settings ────────────────────────────────────────────────────
const PROVIDER_DEFAULTS = {
  local:     {model: '',                       keyVisible: false},
  openai:    {model: 'gpt-4o-mini',            keyVisible: true},
  groq:      {model: 'llama-3.3-70b-versatile',keyVisible: true},
};

function toggleSettings() {
  const bar  = document.getElementById('settings-bar');
  const gear = document.getElementById('gear-btn');
  const open = bar.classList.toggle('open');
  gear.classList.toggle('active', open);
}

function onProviderChange() {
  const provider = document.getElementById('cfg-provider').value;
  const def = PROVIDER_DEFAULTS[provider];
  const keyLabel = document.getElementById('cfg-key-label');
  const keyInput = document.getElementById('cfg-key');
  keyLabel.style.display = def.keyVisible ? '' : 'none';
  keyInput.style.display = def.keyVisible ? '' : 'none';
  document.getElementById('cfg-model').value = def.model;
}

async function saveLLMConfig() {
  const provider = document.getElementById('cfg-provider').value;
  const apiKey   = document.getElementById('cfg-key').value.trim();
  const model    = document.getElementById('cfg-model').value.trim();
  const status   = document.getElementById('cfg-status');
  status.textContent = 'Saving…';
  status.className = 'cfg-status';
  try {
    const res  = await fetch('/set_llm_config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({provider, api_key: apiKey, model}),
    });
    const data = await res.json();
    if (data.ok) {
      status.textContent = 'OK Saved - ' + provider;
      status.className = 'cfg-status ok';
      loadModels();  // refresh model list for new provider
    } else {
      status.textContent = 'Error ' + (data.error || 'Failed');
      status.className = 'cfg-status err';
    }
  } catch(e) {
    status.textContent = 'Error ' + e.message;
    status.className = 'cfg-status err';
  }
}

async function testLLMConfig() {
  const status = document.getElementById('cfg-status');
  status.textContent = 'Testing connection…';
  status.className = 'cfg-status';
  try {
    const res  = await fetch('/test_llm_config');
    const data = await res.json();
    if (data.ok) {
      status.textContent = 'OK ' + (data.response || 'Connected').slice(0, 60);
      status.className = 'cfg-status ok';
    } else {
      status.textContent = 'Error ' + (data.error || 'No response');
      status.className = 'cfg-status err';
    }
  } catch(e) {
    status.textContent = 'Error ' + e.message;
    status.className = 'cfg-status err';
  }
}

async function loadLLMConfig() {
  try {
    const res  = await fetch('/get_llm_config');
    const data = await res.json();
    if (data.provider) {
      document.getElementById('cfg-provider').value = data.provider;
      onProviderChange();
      if (data.model) document.getElementById('cfg-model').value = data.model;
    }
  } catch(e) {}
}
loadLLMConfig();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


# ── Background swarm fallback topic generator ────────────────────────────────
# Used when the RTSP camera is offline so the background swarm still debates.

_FALLBACK_TOPICS = [
    "How should edge AI devices balance privacy with real-time awareness?",
    "Can small language models outperform large ones for specialized embedded tasks?",
    "What are the real risks of autonomous vision systems in public spaces?",
    "Is AI augmenting or replacing human situational awareness?",
    "What ethical frameworks should govern always-on computer vision?",
    "How does real-time object detection change human interaction with physical spaces?",
    "What does genuine AI alignment look like for resource-constrained hardware?",
    "Should embedded AI systems be allowed to make autonomous safety decisions?",
    "How will multi-agent AI systems reshape human decision-making at the edge?",
    "What are the limits of quantized language models for real-world reasoning?",
]
_fallback_topic_idx = 0


def _get_fallback_topic() -> str:
    """Fetch a live AI/tech headline, fall back to a rotating curated topic list."""
    global _fallback_topic_idx
    try:
        from swarm_core.tools import web_search_tool
        import re
        results = web_search_tool("AI robotics edge computing news today", max_results=3)
        if results and not results.startswith("["):
            for line in results.split("\n"):
                m = re.search(r'\[(.+?)\]', line)
                if m and len(m.group(1)) > 20:
                    return m.group(1)
    except Exception:
        pass
    topic = _FALLBACK_TOPICS[_fallback_topic_idx % len(_FALLBACK_TOPICS)]
    _fallback_topic_idx += 1
    return topic


if __name__ == '__main__':
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()

    if SWARM_AVAILABLE:
        def _describe_frame(img_b64: str) -> str:
            return query_ollama(
                "Describe exactly what you see in one sentence. Be specific about objects, actions, and context.",
                img_b64
            )
        # topic_source fires when camera is offline (RTSP unavailable)
        BACKGROUND_SWARM.start(get_frame_b64, _describe_frame, topic_source=_get_fallback_topic)

    print("Starting Vision AI server at http://localhost:5000")
    print("Connecting to RTSP:", RTSP_URL)
    print("Using llama-server at:", LLAMA_URL)
    app.run(host='0.0.0.0', port=5000, debug=False)
