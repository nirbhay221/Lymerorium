
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
    label.innerHTML = 'Switch to Fast for quick results';
    indicator.innerHTML = 'Swarm';
    indicator.style.color = 'rgba(191,90,242,0.9)';
  } else {
    label.innerHTML = '&#9889; Switch to Swarm for an 8-agent deep debate';
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

  // Step 2: run swarm on that topic
  box.textContent = 'Step 2/2 - 8 agents debating… (this takes 1-3 min)';
  const start = Date.now();
  try {
    const res  = await fetch('/swarm_analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topic: topic, max_rounds: 2})
    });
    const data = await res.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (data.error) {
      box.className = 'result-body error';
      box.textContent = data.error;
    } else {
      box.className = 'result-body done';
      box.innerHTML = renderSwarmResult(data);
      badge.innerHTML = '<span class="result-badge" style="background:rgba(191,90,242,0.2);color:rgba(191,90,242,0.9)">Swarm</span>';
      latency.textContent = elapsed + 's - ' + data.messages.length + ' agent turns';
    }
  } catch(e) {
    box.className = 'result-body error';
    box.textContent = 'Swarm error: ' + e.message;
  }
  document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
}

function renderSwarmResult(data) {
  const esc = s => s.split('<').join('&lt;').split('\n').join('<br>');
  let html = '<div class="swarm-verdict">' + esc(data.verdict) + '</div>';
  html += '<div style="font-size:0.72rem;color:var(--label3);margin-bottom:8px">Debate transcript (' + data.messages.length + ' turns):</div>';
  const agentColors = {
    Skeptic:'#ff6b6b', Visionary:'#5ac8fa', Realist:'#32d74b',
    Ethicist:'#ffd60a', Technologist:'#0a84ff', Economist:'#ff9f0a',
    Contrarian:'#ff453a', Synthesizer:'#64d2ff'
  };
  for (const m of data.messages) {
    const color = agentColors[m.agent] || '#fff';
    html += '<div class="swarm-agent-line">';
    html += '<span class="swarm-agent-name" style="color:' + color + '">[' + m.agent + ']</span> ';
    html += esc(m.content);
    html += '</div>';
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
  const safe = content.split('<').join('&lt;').split('\n').join('<br>');
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
        body = 'Working ' + data.note + '\n\nNote Preliminary answer:\n\n' + body;
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
