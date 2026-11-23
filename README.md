# Nova

Nova is a tiny AI chatbot trained from scratch: a 10-million-parameter GPT-style transformer that reads one character at a time.

The whole model runs **in your browser** in plain JavaScript. There's no server and no API, and nothing you type leaves your device.

## What Nova knows

- Everyday common sense: what things are for, where they're found, sizes, and safety.
- Animals, countries and capitals, and basic science.
- HTML and CSS: what tags do, plus example code for buttons, links, lists, tables, forms and pages.
- Jokes, small talk, and remembering your name during a chat.

Nova is small, so it only knows what it was trained on. Questions outside those topics may get a confident-sounding guess.

## Files

| File | What it is |
| --- | --- |
| `index.html` | The chat page |
| `nova-worker.js` | The model, written in JavaScript (runs in a background thread) |
| `model.bin` | The trained weights (float16, 20 MB) |
| `model.json` | Model settings, vocabulary, and where each weight is in `model.bin` |

## Run it locally

The browser can't load the model from a `file://` page, so start a small local server:

```
python -m http.server 8080
```

Then open http://localhost:8080.
