/**
 * Lightweight, ephemeral visual processor.
 * 
 * STRICT PRIVACY INVARIANT: 
 * This module MUST NEVER return raw image data, base64 strings, or blobs.
 * It is solely responsible for generating safe, conservative metadata from the local video frame.
 */

// We create a reusable offscreen canvas so we don't spam the DOM.
const offscreenCanvas = document.createElement("canvas");
const ctx = offscreenCanvas.getContext("2d");

export function processFrame(videoElement) {
  if (!videoElement || videoElement.readyState !== 4) {
    return {
      frame_available: false,
      frame_quality: "unusable"
    };
  }

  // 1. Configure dimensions
  const width = videoElement.videoWidth;
  const height = videoElement.videoHeight;
  
  if (width === 0 || height === 0) {
    return { frame_available: false, frame_quality: "unusable" };
  }

  offscreenCanvas.width = width;
  offscreenCanvas.height = height;

  // 2. Draw frame locally
  try {
    ctx.drawImage(videoElement, 0, 0, width, height);
    
    // We could extract pixels here via ctx.getImageData, but to remain 
    // strictly conservative for Phase 10 without external dependencies, 
    // we simply confirm the frame was drawn successfully.
    
    return {
      frame_available: true,
      frame_width: width,
      frame_height: height,
      frame_quality: "usable"
      // Note: We deliberately DO NOT return "face_detected": true because 
      // we are not running a real face detector. We only return derived facts.
    };
  } catch (err) {
    return {
      frame_available: false,
      frame_quality: "error"
    };
  }
}
