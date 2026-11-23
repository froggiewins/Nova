# Nova

Nova is a tiny AI chatbot trained from scratch: a 10-million-parameter GPT-style transformer that reads one character at a time.

The whole model runs **in your browser** in plain JavaScript. There's no AI server or API: your chat stays on your device.

The one exception is **Teach Nova the right answer**. If you choose to send a lesson, your question and suggested answer go to Nova's trainer. The trainer reviews each lesson by hand and may use it to train the next version of Nova.

## How Nova learns

Every so often, Nova goes through a learning round:

1. It reads short summaries from [Simple English Wikipedia](https://simple.wikipedia.org) (text licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
2. It collects lessons sent from this site, which the trainer approves or rejects one by one.
3. It continues training on the new knowledge, mixed with its original practice conversations so it doesn't forget them.

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
| `config.js` | Where Teach lessons are sent (a Supabase URL and its public key). If it's empty, the Teach button is hidden |

## Run it locally

The browser can't load the model from a `file://` page, so start a small local server:

```
python -m http.server 8080
```

Then open http://localhost:8080.
