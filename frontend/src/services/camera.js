/**
 * Provides a safe abstraction for requesting and managing local camera streams.
 */

export async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Browser does not support getUserMedia");
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false // STRICT INVARIANT: Microphone access is prohibited in this phase
    });
    return stream;
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      throw new Error("Camera permission denied");
    } else if (err.name === 'NotFoundError') {
      throw new Error("Camera unavailable");
    }
    throw new Error("Failed to start camera");
  }
}

export function stopCamera(stream) {
  if (!stream) return;
  stream.getTracks().forEach(track => {
    track.stop();
  });
}
