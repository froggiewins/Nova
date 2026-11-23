// Nova's brain, running in the browser.
// A plain-JavaScript copy of the GPT model in mymodel/model.py. It loads the
// exported weights (model.bin + model.json) and generates replies one character at a time.

const USER = "\x01", BOT = "\x02", END = "\x03";
let cfg, chars, stoi, E, H, HD, L, T, V, END_ID;
let P = {};               // weights by name (Float32Array)
let kCache, vCache;       // per layer: Float32Array(T * E)
let pos = 0;              // next position in the cache
let ctx = [];             // tokens currently in the cache
let meta_threshold = 0;   // surprise level above which Nova says it doesn't understand

// ---------- loading ----------
function halfTable() {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
         : e === 31 ? (f ? NaN : s * Infinity)
         : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return t;
}

// expectedBytes comes from model.json, because servers that compress the file
// (like GitHub Pages) report the compressed size, not the real one.
async function fetchWithProgress(url, expectedBytes, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const chunks = [];
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(1, got / expectedBytes));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function load() {
  const v = new URL(self.location.href).search; // same ?v= as the page, so new models aren't cached
  const meta = await (await fetch("model.json" + v)).json();
  let expected = 0;
  for (const { shape } of Object.values(meta.tensors)) expected += 2 * shape.reduce((a, b) => a * b, 1);
  const bytes = await fetchWithProgress("model.bin" + v, expected, (f) => postMessage({ type: "progress", value: f }));
  if (bytes.byteLength !== expected) throw new Error(`model.bin is ${bytes.byteLength} bytes, expected ${expected}`);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const table = halfTable();
  for (const [name, { offset, shape }] of Object.entries(meta.tensors)) {
    const n = shape.reduce((a, b) => a * b, 1);
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = table[half[offset + i]];
    P[name] = a;
  }
  cfg = meta.config; chars = meta.chars;
  meta_threshold = meta.surprise_threshold || 0;
  stoi = new Map(chars.map((c, i) => [c, i]));
  E = cfg.n_embd; H = cfg.n_head; HD = E / H; L = cfg.n_layer; T = cfg.block_size; V = cfg.vocab_size;
  END_ID = stoi.get(END);
  kCache = []; vCache = [];
  for (let l = 0; l < L; l++) { kCache.push(new Float32Array(T * E)); vCache.push(new Float32Array(T * E)); }
}

// ---------- math ----------
function layerNorm(x, w, b) {
  const n = x.length, out = new Float32Array(n);
  let mean = 0; for (let i = 0; i < n; i++) mean += x[i]; mean /= n;
  let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; } v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// out = W x + b, where W is [rows, cols] (PyTorch Linear layout)
function linear(W, b, x, rows, cols) {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = b ? b[r] : 0;
    const o = r * cols;
    for (let c = 0; c < cols; c++) s += W[o + c] * x[c];
    out[r] = s;
  }
  return out;
}

function erf(x) { // Abramowitz & Stegun 7.1.26
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

// Run one token through the model at position p. Returns logits for the next token.
function step(id, p) {
  const tok = P["tok_emb.weight"], pe = P["pos_emb.weight"];
  let x = new Float32Array(E);
  for (let i = 0; i < E; i++) x[i] = tok[id * E + i] + pe[p * E + i];
  const scale = 1 / Math.sqrt(HD);
  const att = new Float32Array(p + 1);
  for (let l = 0; l < L; l++) {
    const pre = `blocks.${l}.`;
    let h = layerNorm(x, P[pre + "ln1.weight"], P[pre + "ln1.bias"]);
    const qkv = linear(P[pre + "attn.qkv.weight"], P[pre + "attn.qkv.bias"], h, 3 * E, E);
    const K = kCache[l], Vc = vCache[l];
    K.set(qkv.subarray(E, 2 * E), p * E);
    Vc.set(qkv.subarray(2 * E, 3 * E), p * E);
    const y = new Float32Array(E);
    for (let hh = 0; hh < H; hh++) {
      const o = hh * HD;
      let max = -Infinity;
      for (let t = 0; t <= p; t++) {
        let s = 0; const ko = t * E + o;
        for (let d = 0; d < HD; d++) s += qkv[o + d] * K[ko + d];
        s *= scale; att[t] = s; if (s > max) max = s;
      }
      let sum = 0;
      for (let t = 0; t <= p; t++) { att[t] = Math.exp(att[t] - max); sum += att[t]; }
      for (let t = 0; t <= p; t++) {
        const w = att[t] / sum, vo = t * E + o;
        for (let d = 0; d < HD; d++) y[o + d] += w * Vc[vo + d];
      }
    }
    const a = linear(P[pre + "attn.proj.weight"], P[pre + "attn.proj.bias"], y, E, E);
    for (let i = 0; i < E; i++) x[i] += a[i];
    h = layerNorm(x, P[pre + "ln2.weight"], P[pre + "ln2.bias"]);
    const m = linear(P[pre + "mlp.0.weight"], P[pre + "mlp.0.bias"], h, 4 * E, E);
    for (let i = 0; i < m.length; i++) m[i] = 0.5 * m[i] * (1 + erf(m[i] / Math.SQRT2));
    const m2 = linear(P[pre + "mlp.2.weight"], P[pre + "mlp.2.bias"], m, E, 4 * E);
    for (let i = 0; i < E; i++) x[i] += m2[i];
  }
  x = layerNorm(x, P["ln_f.weight"], P["ln_f.bias"]);
  return linear(tok, null, x, V, E); // output layer shares weights with the token embedding
}

// Feed one token into the running conversation, sliding the window when it's full.
function feed(id) {
  if (pos >= T) {
    const keep = ctx.slice(-Math.floor(T * 0.6));
    pos = 0; ctx = [];
    for (const t of keep) { step(t, pos); ctx.push(t); pos++; }
  }
  const logits = step(id, pos);
  ctx.push(id); pos++;
  return logits;
}

function sample(logits, temperature, topK) {
  if (temperature <= 0) { let b = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i; return b; }
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
  const mx = logits[idx[0]];
  const ps = idx.map((i) => Math.exp((logits[i] - mx) / temperature));
  let r = Math.random() * ps.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; j++) { r -= ps[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}

// ---------- dictionary lookup (exact definitions from WordNet, loaded on first use) ----------
let DICT = null;
async function loadDictionary() {
  if (!DICT) {
    const v = new URL(self.location.href).search;
    DICT = await (await fetch("dictionary.json" + v)).json();
  }
  return DICT;
}

// "what does X mean?", "define X", "meaning of X" ... -> X
function definitionRequest(message) {
  const m = message.toLowerCase().trim().replace(/[?.!]+$/, "").replace(/\s+(pls|please)$/, "").trim();
  const pats = [
    /^(?:hey |so |um )?(?:what|wat|wht)\s*(?:does|do|'s|s|is)\s+(?:the word\s+|the phrase\s+)?(.+?)\s+mean$/,
    /^(?:can you |could you |pls |please )?define(?: the word| the phrase)?\s+(.+)$/,
    /^(?:what(?:'s|s| is)?\s+)?(?:the\s+)?(?:meaning|definition|def)\s+of\s+(?:the word\s+)?(.+)$/,
    /^(?:what(?:'s|s| is| are)?\s+)(?:a |an |the )?meaning of\s+(.+)$/,
  ];
  for (const p of pats) { const r = m.match(p); if (r) return r[1].replace(/^["']|["']$/g, "").trim(); }
  return null;
}

// Find the dictionary form of a word: cats -> cat, running -> run, mice -> mouse
function baseForm(word, d) {
  const w = word.toLowerCase().trim();
  if (d.words[w]) return w;
  if (d.irregular[w]) return d.irregular[w];
  const rules = [["ies", "y"], ["ves", "f"], ["ves", "fe"], ["ses", "s"], ["xes", "x"], ["zes", "z"], ["ches", "ch"],
    ["shes", "sh"], ["men", "man"], ["es", "e"], ["es", ""], ["s", ""], ["ied", "y"], ["ed", "e"], ["ed", ""],
    ["ing", "e"], ["ing", ""], ["ier", "y"], ["iest", "y"], ["er", "e"], ["er", ""], ["est", "e"], ["est", ""], ["ly", ""]];
  for (const [suf, rep] of rules) {
    if (!w.endsWith(suf) || w.length <= suf.length + 1) continue;
    const stem = w.slice(0, -suf.length);
    for (const cand of [stem + rep, /([b-df-hj-np-tv-z])\1$/.test(stem) ? stem.slice(0, -1) : null]) {
      if (cand && d.words[cand]) return cand;
    }
  }
  return null;
}

function describe(word, base, d) {
  const e = d.words[base];
  let s = `${base.charAt(0).toUpperCase() + base.slice(1)} (${e[0][0]}): ${e[0][1]}.`;
  for (const [pos, def] of e.slice(1, 3)) s += ` As ${/^[aeiou]/.test(pos) ? "an" : "a"} ${pos}: ${def}.`;
  if (base !== word.toLowerCase()) s = `"${word}" is a form of "${base}". ` + s;
  return s;
}

async function lookup(word) {
  const d = await loadDictionary();
  const base = baseForm(word, d);
  return base ? describe(word, base, d)
    : `I couldn't find "${word}" in my dictionary. Could you check the spelling?`;
}

// For a message Nova didn't understand: is it just a word, or "what is a X"?
async function lookupFallback(message) {
  const m = message.toLowerCase().trim().replace(/[?.!]+$/, "");
  const r = m.match(/^(?:what(?:'s| is| are)\s+)?(?:a |an |the )?([a-z][a-z' -]{1,30})$/);
  if (!r || r[1].split(" ").length > 3) return null;
  const d = await loadDictionary();
  const base = baseForm(r[1], d);
  return base ? describe(r[1], base, d) : null;
}

const CONFUSED ="I'm not sure I understand that. I'm a small model, so I only know about everyday things, " +
  "animals, countries, science, word meanings and basic HTML. Could you ask it a different way?";

// How unlikely the model found the token `id`, given the logits that predicted it.
function surpriseOf(logits, id) {
  let mx = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let s = 0; for (let i = 0; i < logits.length; i++) s += Math.exp(logits[i] - mx);
  return Math.log(s) + mx - logits[id];
}

async function reply(message, temperature) {
  // Word definitions come straight from the dictionary, not from the model's memory.
  const word = definitionRequest(message);
  if (word) return await lookup(word);

  const startPos = pos, startLen = ctx.length;
  const msgIds = [...message.slice(0, 300)].map((c) => stoi.get(c)).filter((id) => id !== undefined);
  let logits = feed(stoi.get(USER)), total = 0, n = 0;
  for (const id of msgIds) { total += surpriseOf(logits, id); n++; logits = feed(id); }
  const threshold = meta_threshold;
  if (threshold && n >= 3 && total / n > threshold) {
    // Too unfamiliar to answer: forget this message (when the window didn't slide), then
    // try the dictionary in case it's just a word, otherwise say so.
    if (ctx.length === startLen + 1 + msgIds.length) { pos = startPos; ctx.length = startLen; }
    return (await lookupFallback(message)) || CONFUSED;
  }
  logits = feed(stoi.get(BOT));
  let text = "", last = performance.now();
  for (let n = 0; n < 450; n++) {
    const id = sample(logits, temperature, 20);
    if (id === END_ID) { feed(id); break; }
    const ch = chars[id];
    if (ch === USER || ch === BOT) { feed(END_ID); break; }
    text += ch;
    logits = feed(id);
    if (performance.now() - last > 60) { postMessage({ type: "token", text }); last = performance.now(); }
    if (n === 449) feed(END_ID);
  }
  return text.trim() || "...";
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") { await load(); postMessage({ type: "ready" }); }
    else if (msg.type === "reset") { pos = 0; ctx = []; }
    else if (msg.type === "chat") {
      const text = await reply(msg.text, msg.temperature ?? 0.5);
      postMessage({ type: "done", text });
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.message || err) });
  }
};
