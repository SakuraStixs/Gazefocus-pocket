# GazeFocus Pocket

A phone-sized taste of GazeFocus, the webcam eye-tracking desktop app: pick one of four "windows" just by looking at it, using your phone's front camera.

**Try it:** https://sakurastixs.github.io/gazefocus-pocket/

1. Tap **Start camera** and allow the camera.
2. Hold the phone at eye level, about 30 cm away, with light on your face. Follow the 13 dots with your eyes.
3. Look at a window for half a second and it takes focus. If it drifts, tap where you're looking: each tap teaches it.

It runs the same pipeline as the desktop app, in the browser: MediaPipe Face Landmarker, then eye and head features, an eye-shape profile with a blink guard, a per-user ridge regression, a fixation filter with a spring glide, dwell to focus, undo, and learning from taps. Everything stays on the phone; no video is uploaded anywhere.

On a phone screen expect window-level accuracy: it can tell which quarter you're looking at, not exact spots.

`vendor/` holds MediaPipe Tasks Vision 0.10.14 and `models/` its face landmarker model (both Apache 2.0, from Google), so the page needs nothing from other servers.
