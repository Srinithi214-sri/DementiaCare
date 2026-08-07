import { useEffect, useMemo, useState, useCallback } from "react";

/**
 * DementiaCare OS — Patient Companion Screen
 *
 * This is the actual patient-facing screen. There is no menu, no login,
 * no settings — the patient never operates anything. In production, every
 * state change below would be triggered by the Fusion Agent reacting to
 * camera/mic signals. Here, since there is no backend yet, the screen
 * quietly cycles through the same states on a timer so you can see and
 * demo the real behavior end to end.
 *
 * Click/tap anywhere on the screen to skip to the next moment early —
 * useful when presenting live so you aren't stuck waiting on the timer.
 */

// What the fusion agent would eventually feed in. Hardcoded here for now.
const SCRIPT = [
  { state: "quiet", holdMs: 7000 },
  { state: "greeting", holdMs: 6500 },
  { state: "quiet", holdMs: 5000 },
  { state: "memory", holdMs: 7500 },
  { state: "quiet", holdMs: 5000 },
  { state: "reassurance", holdMs: 6500 },
];

const COPY = {
  quiet: {
    caption: null,
    sub: "A quiet moment. Nothing is needed right now.",
  },
  greeting: {
    caption: "This is Priya — your daughter. She visits every Sunday.",
    sub: null,
    marker: true,
  },
  memory: {
    caption: "Your wedding day, Chennai, 1987. You wore your mother's necklace.",
    sub: null,
    memory: true,
  },
  reassurance: {
    caption: "You're safe. Just rest a moment — help is close by.",
    sub: null,
  },
};

function getTimeOfDay(date) {
  const h = date.getHours();
  if (h >= 5 && h < 12) return "morning";
  if (h >= 12 && h < 17) return "afternoon";
  return "evening";
}

export default function App() {
  const [now, setNow] = useState(new Date());
  const [scriptIndex, setScriptIndex] = useState(0);
  const [showCaption, setShowCaption] = useState(false);

  // Real clock — the actual product reads the real time, it doesn't fake it.
  useEffect(() => {
    const clockId = setInterval(() => setNow(new Date()), 60 * 1000);
    return () => clearInterval(clockId);
  }, []);

  const timeOfDay = useMemo(() => getTimeOfDay(now), [now]);
  const step = SCRIPT[scriptIndex];
  const copy = COPY[step.state];

  const advance = useCallback(() => {
    setScriptIndex((i) => (i + 1) % SCRIPT.length);
  }, []);

  // Auto-advance on the script's own timing.
  useEffect(() => {
    const id = setTimeout(advance, step.holdMs);
    return () => clearTimeout(id);
  }, [scriptIndex, step.holdMs, advance]);

  // Caption fades in slightly after the state changes, never instantly.
  useEffect(() => {
    setShowCaption(false);
    const id = setTimeout(() => setShowCaption(true), 450);
    return () => clearTimeout(id);
  }, [scriptIndex]);

  return (
    <div
      className={`screen theme-${timeOfDay} state-${step.state}`}
      onClick={advance}
      role="presentation"
    >
      <div className="ambient-glow ambient-glow--a" />
      <div className="ambient-glow ambient-glow--b" />

      <div className="time-label">a gentle {timeOfDay}</div>

      <div className="presence-wrap">
        <div className="presence">
          <div className="presence-core" />
        </div>

        <div className="caption-slot">
          {copy.caption && showCaption && (
            <div className="caption-bubble">{copy.caption}</div>
          )}
          {!copy.caption && copy.sub && showCaption && (
            <div className="caption-sub">{copy.sub}</div>
          )}
        </div>
      </div>

      <div className={`memory-corner ${copy.memory ? "is-active" : ""}`} />

      {copy.marker && <div className="visitor-marker" />}

      {step.state === "reassurance" && <div className="reassurance-wash" />}
    </div>
  );
}
