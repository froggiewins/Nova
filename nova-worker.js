// Nova's brain, running in the browser.
// A plain-JavaScript copy of the GPT model in mymodel/model.py. It loads the
// exported weights (model.bin + model.json) and generates replies one character at a time.

const USER = "\x01", BOT = "\x02", END = "\x03";
let cfg, chars, stoi, E, H, HD, L, T, V, END_ID;
let P = {};               // weights by name (Float32Array)
let kCache, vCache;       // per layer: Float32Array(T * E)
let pos = 0;              // next position in the cache
let ctx = [];             // tokens currently in the cache
let meta_threshold = 0;   // average surprise above which Nova says it doesn't understand
let word_threshold = 0;   // same, for the single most unfamiliar word

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
  word_threshold = meta.word_surprise_threshold || 0;
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

// ---------- web search (Wikipedia; same rules as mymodel/websearch.py) ----------
// Nova is too small to read a page and rewrite it, so it quotes the sentences that best match the question.
const WIKIS = ["https://simple.wikipedia.org", "https://en.wikipedia.org"];
const SEARCH_CMD = /^(?:hey |so |please |pls |can you |could you |nova,? )*(?:search|look up|lookup|google|find)(?: the web| online| the internet| wikipedia| on wikipedia| on the web)?(?: for| about| up)?\s+(.+)$/i;
const QUESTION = /^(?:hey |so |um |please |pls |do you know |can you tell me |tell me )?(?:who|what|whats|what's|when|where|which|why|how|tell me about|explain|is|are|was|were|did|does)\b/i;
const MAKE = /\b(?:make|build|create|write|code|draw|design|program)\b/i;
const WANTS = [ // what kind of number an answer needs, so the sentence that has one wins
  [/^(?:when|what year|in what year)\b/i, /\b(?:1\d{3}|20\d{2})\b/],
  [/^how (?:many|much)\b/i, /\d[\d,.]*\d|\b\d+\b|\b(?:hundred|thousand|million|billion)\b/i],
  [/^how (?:big|tall|long|far|deep|high|heavy|wide|fast)\b/i,
   /\d[\d,.]*\s*(?:m|km|cm|metres?|meters?|kilometres?|kilometers?|feet|ft|miles?|kg|tonnes?|tons|km\/h|mph)\b/i],
  [/^how old\b/i, /\b\d[\d,.]*\s*(?:years?|million|billion)\b|\b(?:1\d{3}|20\d{2})\b/i],
];
const STOP = new Set(`a an the of in on at to for from by with and or is are was were be been being do does did
how many much what whats what's who whom whose when where which why there their they them it its this that
these those me my i you your u ur we our can could would should will shall may might please pls tell about
know explain define search look up lookup google find web online internet wikipedia hey so um any some
big old tall long far fast heavy deep high year much like have has had`.split(/\s+/));
const IRREGULAR = { sank: "sink", sunk: "sink", began: "begin", begun: "begin", built: "build", made: "make",
  won: "win", wrote: "write", written: "write", ran: "run", flew: "fly", flown: "fly", grew: "grow", grown: "grow",
  ate: "eat", eaten: "eat", found: "find", fought: "fight", taught: "teach", bought: "buy", thought: "think",
  born: "bear", died: "die", dies: "die", lived: "live", lives: "live", people: "person", children: "child",
  mice: "mouse", feet: "foot", teeth: "tooth", geese: "goose", men: "man", women: "woman" };

function searchCommand(message) {
  const r = message.trim().replace(/[?.! ]+$/, "").match(SEARCH_CMD);
  return r ? r[1] : null;
}
function isFactQuestion(message) { const m = message.trim(); return QUESTION.test(m) && !MAKE.test(m); }
const words = (t) => t.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
const keywords = (t) => words(t).filter((w) => !STOP.has(w) && w.length > 1);
function stem(w) {
  if (IRREGULAR[w]) return IRREGULAR[w];
  for (const suf of ["ies", "ing", "es", "ed", "s"]) if (w.endsWith(suf) && w.length > suf.length + 2) return w.slice(0, -suf.length);
  return w;
}
const stemsOf = (t) => new Set(words(t).map(stem));

async function wikiApi(wiki, params) {
  const r = await fetch(`${wiki}/w/api.php?` + new URLSearchParams({ ...params, format: "json", origin: "*" }));
  if (!r.ok) throw new Error(`Wikipedia said ${r.status}`);
  return r.json();
}

// Search, then take the first result whose whole title is in the question (Moon, not "Earth phase").
// If none fits, try the article named after the question's first keyword (Cat, not "Schrödinger's cat").
async function findPage(wiki, kw) {
  const hits = (await wikiApi(wiki, { action: "query", list: "search", srsearch: kw.join(" "), srlimit: 10 })).query.search;
  if (!hits.length) return null;
  const want = new Set(kw.map(stem));
  const fit = hits.find((h) => [...stemsOf(h.title)].every((s) => want.has(s) || STOP.has(s)));
  const titles = fit ? [fit.title] : [kw[0].charAt(0).toUpperCase() + kw[0].slice(1), hits[0].title];
  for (const title of titles) {
    const pages = (await wikiApi(wiki, { action: "query", prop: "extracts|info", titles: title, explaintext: 1,
      exsectionformat: "plain", inprop: "url", redirects: 1 })).query.pages;
    const page = Object.values(pages)[0];
    if (page && !("missing" in page) && page.extract) return page;
  }
  return null;
}

function bestSentences(question, page, kw) {
  const text = (page.extract || "").replace(/'''/g, "")
    .split(/\n(?:References|See also|Notes|Sources|Other websites|External links)\n/)[0];
  const sents = text.split(/(?<=[.!?])\s+(?=[A-Z0-9"(])|(?<=[a-z][.!?])(?=[A-Z])|\n+/).map((s) => s.trim())
    .filter((s) => s.length > 25 && s.length < 400 && /[.!?]$/.test(s)).slice(0, 150);
  if (!sents.length) return [];
  const stems = new Set(kw.map(stem));
  const q = question.trim().replace(/^(?:hey |so |um |please |pls |do you know |can you tell me |tell me )+/i, "");
  const want = (WANTS.find(([ask]) => ask.test(q)) || [])[1];
  const score = (i) => {
    let n = 0; for (const s of stemsOf(sents[i])) if (stems.has(s)) n++;
    return n + (want && want.test(sents[i]) ? 3 : 0) - i * 0.01; // earlier wins ties
  };
  const about = new Set([...stemsOf(page.title), ...stemsOf(sents[0])]);
  if (![...stems].some((s) => about.has(s))) return []; // the page isn't about what was asked
  let best = 0; for (let i = 1; i < sents.length; i++) if (score(i) > score(best)) best = i;
  if (best === 0 || (!want && score(best) < score(0) + 1))
    return sents[0].length < 120 && sents.length > 1 ? sents.slice(0, 2) : sents.slice(0, 1);
  return [sents[0], sents[best]];
}

// Nova's answer with its source, or null if nothing useful was found (or the visitor is offline).
async function webSearch(question) {
  const kw = keywords(question).slice(0, 8);
  if (!kw.length) return null;
  for (const wiki of WIKIS) {
    let page;
    try { page = await findPage(wiki, kw); } catch { continue; }
    if (!page) continue;
    const sents = bestSentences(question, page, kw);
    if (!sents.length) continue;
    let answer = sents.join(" ");
    if (answer.length > 600) answer = answer.slice(0, 600).replace(/\s+\S*$/, "") + "...";
    return `I searched Wikipedia and found this:\n\n${answer}\n\nSource: ${page.title}, ${page.fullurl}`;
  }
  return null;
}

const NOT_FOUND = "I searched Wikipedia but couldn't find a good answer to that. Could you try different words?";

const CONFUSED ="I'm not sure I understand that. I'm a small model, so I only know about everyday things, " +
  "animals, countries, science, word meanings and basic HTML. Could you ask it a different way?";

// How unlikely the model found the token `id`, given the logits that predicted it.
function surpriseOf(logits, id) {
  let mx = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let s = 0; for (let i = 0; i < logits.length; i++) s += Math.exp(logits[i] - mx);
  return Math.log(s) + mx - logits[id];
}

// After these phrases comes the person's own content (a name, a button label, a link...).
// Nova can't have seen it before, so it doesn't count as "not understanding". Same rule as surprise.py.
const FREE_TEXT = /\b(?:my name is|name's|i'm|im|i am|call me|says|saying|called|named|that reads|with the text|link to|links to|list of|title)\b/i;

// Average surprise of each word (3+ letters) before any free-text part of the message.
function worstWord(message, charScores) {
  const m = FREE_TEXT.exec(message);
  const cut = m ? m.index + m[0].length : message.length;
  let worst = 0;
  for (const w of message.slice(0, cut).matchAll(/\S+/g)) {
    const word = w[0];
    if (word.replace(/[^A-Za-z]/g, "").length < 3 || /[0-9./@:]/.test(word)) continue;
    const seg = charScores.slice(w.index, w.index + word.length).filter((s) => s !== null);
    if (seg.length) worst = Math.max(worst, seg.reduce((a, b) => a + b, 0) / seg.length);
  }
  return worst;
}

async function reply(message, temperature) {
  // Word definitions come straight from the dictionary, not from the model's memory.
  const word = definitionRequest(message);
  if (word) return await lookup(word);
  // "search for X", "look up X": go straight to the web.
  const topic = searchCommand(message);
  if (topic) { postMessage({ type: "token", text: "Searching Wikipedia…" }); return (await webSearch(topic)) || NOT_FOUND; }

  const startPos = pos, startLen = ctx.length;
  const text300 = message.slice(0, 300);
  const charScores = new Array(text300.length).fill(null);
  let logits = feed(stoi.get(USER)), total = 0, n = 0;
  for (let i = 0; i < text300.length; i++) {
    const id = stoi.get(text300[i]);
    if (id === undefined) continue;
    const s = surpriseOf(logits, id);
    charScores[i] = s; total += s; n++;
    logits = feed(id);
  }
  const unfamiliar = message.trim().length >= 3 && n > 0 && (
    (meta_threshold && total / n > meta_threshold) ||
    (word_threshold && worstWord(text300, charScores) > word_threshold));
  if (unfamiliar) {
    // Too unfamiliar to answer: forget this message (when the window didn't slide), then
    // search the web if it's a question, then try the dictionary in case it's just a word, otherwise say so.
    if (ctx.length === startLen + 1 + n) { pos = startPos; ctx.length = startLen; }
    if (isFactQuestion(message)) {
      postMessage({ type: "token", text: "Searching Wikipedia…" });
      const found = await webSearch(message);
      if (found) return found;
    }
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
    } else if (msg.type === "search") { // the "Search the web" button under an answer
      const text = (await webSearch(searchCommand(msg.text) || msg.text)) || NOT_FOUND;
      postMessage({ type: "done", text, searched: true });
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.message || err) });
  }
};
