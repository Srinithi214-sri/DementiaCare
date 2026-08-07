# DementiaCare OS — Patient Companion Screen

The actual patient-facing frontend. Plain React + plain CSS (no Tailwind,
no CSS framework, no UI library).

## Run it

```
npm install
npm run dev
```

Then open the local URL it prints (usually http://localhost:5173).

For your demo tomorrow, open it fullscreen (F11 in most browsers) on a
laptop or tablet browser to show what the patient would actually see.

## How it behaves right now

There's no backend yet, so the screen runs through a fixed script of
moments (quiet → visitor greeting → quiet → memory/reminiscence → quiet →
reassurance) on a timer, the same way it eventually will once the Fusion
Agent is feeding it real events instead of `SCRIPT` in `App.jsx`.

- **Click/tap anywhere on the screen** to skip ahead early — useful so you
  aren't stuck waiting on the timer while presenting live.
- **The time-of-day background is real**, not simulated — it reads your
  computer's actual clock (morning / afternoon / evening), the same way
  the tablet would in someone's home.

## Where to plug in real data later

Everything the patient could eventually see lives in two places in
`src/App.jsx`:

- `SCRIPT` — the sequence and timing of moments. Replace this with
  whatever the Fusion Agent's `RESPOND` node sends down (e.g. over a
  WebSocket or polling an endpoint) instead of a fixed array.
- `COPY` — the actual text/captions shown for each state. Replace the
  hardcoded strings with whatever the LLM or answer bank generates.

Everything else (the breathing presence, the memory corner, the visitor
marker, the color themes) is pure CSS in `src/App.css` and doesn't need to
change when you wire up real data.

## What's intentionally not here yet

- No caregiver alerts, no login, no settings — this screen is only ever
  the patient's ambient companion view. The caregiver dashboard is a
  separate app.
- No real camera/mic/AI — `SCRIPT` is a stand-in for what those signals
  will eventually trigger.
