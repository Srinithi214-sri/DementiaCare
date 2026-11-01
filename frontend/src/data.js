// What the fusion agent would eventually feed in. Hardcoded here for now.
export const SCRIPT = [
  { state: "quiet", holdMs: 7000 },
  { state: "greeting", holdMs: 6500 },
  { state: "quiet", holdMs: 5000 },
  { state: "memory", holdMs: 7500 },
  { state: "quiet", holdMs: 5000 },
  { state: "reassurance", holdMs: 6500 },
];

export const COPY = {
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
