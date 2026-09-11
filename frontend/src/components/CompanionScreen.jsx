import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { api } from "../services/api";
import { startCamera, stopCamera } from "../services/camera";
import { processFrame } from "../services/visualProcessor";
import { createSpeechRecognizer } from "../services/speechRecognition";
import { processTranscript } from "../services/speechProcessor";
import { textToSpeech } from "../services/textToSpeech";
function getTimeOfDay(date) {
  const h = date.getHours();
  if (h >= 5 && h < 12) return "morning";
  if (h >= 12 && h < 17) return "afternoon";
  return "evening";
}

export default function CompanionScreen({ patient, onExit }) {
  const [now, setNow] = useState(new Date());
  
  // State for FusionDecision
  const [decision, setDecision] = useState({
    intent: "quiet",
    response_type: "silent",
    response_text: null,
    action: "none"
  });
  
  const [showCaption, setShowCaption] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // --- CAMERA STATE ---
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // --- SPEECH STATE ---
  const [speechActive, setSpeechActive] = useState(false);
  const [speechError, setSpeechError] = useState(null);
  const speechRef = useRef(null);

  // --- TTS STATE ---
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [ttsSpeaking, setTtsSpeaking] = useState(false);
  const currentRequestId = useRef(0);

  // Real clock
  useEffect(() => {
    const clockId = setInterval(() => setNow(new Date()), 60 * 1000);
    return () => clearInterval(clockId);
  }, []);

  const timeOfDay = useMemo(() => getTimeOfDay(now), [now]);

  // Handle manual UI interaction event -> API call -> Update UI
  const handleInteraction = useCallback(async () => {
    if (isProcessing) return;
    setIsProcessing(true);
    
    currentRequestId.current += 1;
    const reqId = currentRequestId.current;

    try {
      const eventResp = await api.createEvent(patient.id, { interaction: "touch" });
      const fusionDecision = await api.processAgent(patient.id, eventResp.id);
      
      if (reqId === currentRequestId.current) {
        setDecision(fusionDecision);
        if (voiceEnabled && fusionDecision.response_text && typeof fusionDecision.response_text === 'string') {
          textToSpeech.speak(
            fusionDecision.response_text,
            { language: patient.language },
            () => {
              setTtsSpeaking(true);
              // Prevent feedback loop: stop microphone if active
              if (speechRef.current) {
                speechRef.current.stop();
                speechRef.current = null;
              }
              setSpeechActive(false);
            },
            () => setTtsSpeaking(false),
            () => setTtsSpeaking(false)
          );
        }
      }
    } catch (err) {
      console.error("Fusion Agent error:", err);
      if (reqId === currentRequestId.current) {
        setDecision({
          intent: "quiet",
          response_type: "silent",
          response_text: "Connection temporarily lost. Please rest.",
          action: "none"
        });
      }
    } finally {
      if (reqId === currentRequestId.current) setIsProcessing(false);
    }
  }, [patient.id, patient.language, isProcessing, voiceEnabled]);

  // --- CAMERA LIFECYCLE ---
  const handleStartCamera = async (e) => {
    e.stopPropagation();
    setCameraError(null);
    try {
      const stream = await startCamera();
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
    } catch (err) {
      setCameraError(err.message);
    }
  };

  const handleStopCamera = useCallback((e) => {
    if (e) e.stopPropagation();
    if (streamRef.current) {
      stopCamera(streamRef.current);
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  // Ensure camera stops when component unmounts or patient changes
  useEffect(() => {
    return () => {
      handleStopCamera();
    };
  }, [patient.id, handleStopCamera]);

  // --- SPEECH LIFECYCLE ---
  const handleStartSpeech = useCallback((e) => {
    if (e) e.stopPropagation();
    setSpeechError(null);
    
    const recognizer = createSpeechRecognizer({ continuous: false, interimResults: false });
    if (!recognizer.supported) {
      setSpeechError("Speech Recognition not supported in this browser");
      return;
    }

    recognizer.setOnResult(async (rawTranscript) => {
      // Auto-stop UI state when result arrives (single-shot behavior)
      setSpeechActive(false);
      
      const payload = processTranscript(rawTranscript);
      if (!payload.transcript_available) return;

      if (isProcessing) return;
      setIsProcessing(true);

      currentRequestId.current += 1;
      const reqId = currentRequestId.current;

      try {
        const eventResp = await api.createEvent(patient.id, payload, "speech", "speech_transcript");
        const fusionDecision = await api.processAgent(patient.id, eventResp.id);
        
        if (reqId === currentRequestId.current) {
          setDecision(fusionDecision);
          if (voiceEnabled && fusionDecision.response_text && typeof fusionDecision.response_text === 'string') {
            textToSpeech.speak(
              fusionDecision.response_text,
              { language: patient.language },
              () => setTtsSpeaking(true),
              () => setTtsSpeaking(false),
              () => setTtsSpeaking(false)
            );
          }
        }
      } catch (err) {
        console.error("Speech flow error:", err);
      } finally {
        if (reqId === currentRequestId.current) setIsProcessing(false);
      }
    });

    recognizer.setOnError((err) => {
      setSpeechError(err.error || err.message || "Speech error");
      setSpeechActive(false);
    });

    recognizer.setOnEnd(() => {
      setSpeechActive(false);
    });

    try {
      recognizer.start();
      speechRef.current = recognizer;
      setSpeechActive(true);
    } catch (err) {
      setSpeechError(err.message);
    }
  }, [patient.id, patient.language, isProcessing, voiceEnabled]);

  const handleStopSpeech = useCallback((e) => {
    if (e) e.stopPropagation();
    if (speechRef.current) {
      speechRef.current.stop();
      speechRef.current = null;
    }
    setSpeechActive(false);
  }, []);

  // Ensure speech stops when component unmounts or patient changes
  useEffect(() => {
    // When patient changes, invalidate current request ID
    currentRequestId.current += 1;
    textToSpeech.stop();
    setTtsSpeaking(false);
    return () => {
      handleStopSpeech();
      textToSpeech.stop();
    };
  }, [patient.id, handleStopSpeech]);

  // --- OBSERVATION LOOP ---
  useEffect(() => {
    if (cameraActive) {
      intervalRef.current = setInterval(async () => {
        // Prevent overlapping processing
        if (isProcessing) return;
        
        const metadata = processFrame(videoRef.current);
        if (!metadata.frame_available) return;

        setIsProcessing(true);
        currentRequestId.current += 1;
        const reqId = currentRequestId.current;

        try {
          // 1. Create visual event
          const eventResp = await api.createEvent(patient.id, metadata, "camera", "visual_observation");
          // 2. Process Fusion Agent
          const fusionDecision = await api.processAgent(patient.id, eventResp.id);
          
          if (reqId === currentRequestId.current) {
            setDecision(fusionDecision);
            if (voiceEnabled && fusionDecision.response_text && typeof fusionDecision.response_text === 'string') {
              textToSpeech.speak(
                fusionDecision.response_text,
                { language: patient.language },
                () => {
                  setTtsSpeaking(true);
                  if (speechRef.current) {
                    speechRef.current.stop();
                    speechRef.current = null;
                  }
                  setSpeechActive(false);
                },
                () => setTtsSpeaking(false),
                () => setTtsSpeaking(false)
              );
            }
          }
        } catch (err) {
          console.error("Visual processing flow error:", err);
        } finally {
          if (reqId === currentRequestId.current) setIsProcessing(false);
        }
      }, 3000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [cameraActive, patient.id, patient.language, isProcessing, voiceEnabled]);


  // Auto-clear transient states
  useEffect(() => {
    if (decision.intent !== "quiet" && decision.intent !== "idle") {
      const id = setTimeout(() => {
        setDecision({
          intent: "quiet",
          response_type: "silent",
          response_text: null,
          action: "none"
        });
      }, 8000);
      return () => clearTimeout(id);
    }
  }, [decision]);

  // Caption animation
  useEffect(() => {
    setShowCaption(false);
    const id = setTimeout(() => setShowCaption(true), 450);
    return () => clearTimeout(id);
  }, [decision]);

  // Map backend intents to CSS states
  let cssState = "quiet";
  if (decision.intent === "greeting") cssState = "greeting";
  if (decision.intent === "reminiscence" || decision.action === "display_memory") cssState = "memory";
  if (decision.intent === "reassurance") cssState = "reassurance";
  if (decision.response_type === "clarification") cssState = "quiet";

  return (
    <div
      className={`screen theme-${timeOfDay} state-${cssState}`}
      onClick={handleInteraction}
      role="presentation"
    >
      {/* UI Controls */}
      <div style={{ position: "absolute", top: "1rem", left: "1rem", zIndex: 1000, display: "flex", gap: "0.5rem" }}>
        <button 
          onClick={(e) => { e.stopPropagation(); handleStopCamera(); handleStopSpeech(); textToSpeech.stop(); onExit(); }} 
          style={{ padding: "0.5rem", background: "rgba(0,0,0,0.5)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
        >
          Exit Companion
        </button>

        {!voiceEnabled ? (
          <button 
            onClick={(e) => { e.stopPropagation(); setVoiceEnabled(true); }}
            style={{ padding: "0.5rem", background: "rgba(0,123,255,0.8)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Enable Voice
          </button>
        ) : (
          <button 
            onClick={(e) => { e.stopPropagation(); setVoiceEnabled(false); textToSpeech.stop(); setTtsSpeaking(false); }}
            style={{ padding: "0.5rem", background: "rgba(220,53,69,0.8)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Disable Voice
          </button>
        )}

        {ttsSpeaking && (
          <button 
            onClick={(e) => { e.stopPropagation(); textToSpeech.stop(); setTtsSpeaking(false); }}
            style={{ padding: "0.5rem", background: "rgba(255,193,7,0.9)", color: "black", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Stop Speaking
          </button>
        )}

        {!cameraActive ? (
          <button 
            onClick={handleStartCamera}
            style={{ padding: "0.5rem", background: "rgba(40,167,69,0.8)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Start Camera
          </button>
        ) : (
          <button 
            onClick={handleStopCamera}
            style={{ padding: "0.5rem", background: "rgba(220,53,69,0.8)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Stop Camera
          </button>
        )}
        
        {!speechActive ? (
          <button 
            onClick={handleStartSpeech}
            style={{ padding: "0.5rem", background: "rgba(0,123,255,0.8)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Start Speech
          </button>
        ) : (
          <button 
            onClick={handleStopSpeech}
            style={{ padding: "0.5rem", background: "rgba(220,53,69,0.8)", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            Stop Speech
          </button>
        )}

        {cameraError && <span style={{ color: "red", alignSelf: "center", background: "rgba(255,255,255,0.8)", padding: "2px 4px", borderRadius: "2px" }}>{cameraError}</span>}
        {speechError && <span style={{ color: "red", alignSelf: "center", background: "rgba(255,255,255,0.8)", padding: "2px 4px", borderRadius: "2px" }}>{speechError}</span>}
      </div>

      {/* Local Video Preview */}
      <video 
        ref={videoRef}
        autoPlay 
        playsInline 
        muted 
        style={{ 
          position: "absolute", 
          bottom: "1rem", 
          right: "1rem", 
          width: "160px", 
          height: "120px", 
          zIndex: 1000,
          backgroundColor: "#000",
          borderRadius: "8px",
          display: cameraActive ? "block" : "none"
        }} 
      />

      <div className="ambient-glow ambient-glow--a" />
      <div className="ambient-glow ambient-glow--b" />

      <div className="time-label">a gentle {timeOfDay}</div>

      <div className="presence-wrap">
        <div className="presence">
          <div className="presence-core" />
        </div>

        <div className="caption-slot">
          {decision.response_text && showCaption && (
            <div className="caption-bubble">{decision.response_text}</div>
          )}
          {!decision.response_text && cssState === "quiet" && showCaption && (
            <div className="caption-sub">A quiet moment. Nothing is needed right now.</div>
          )}
          {isProcessing && <div className="caption-sub" style={{ opacity: 0.5 }}>...</div>}
        </div>
      </div>

      <div className={`memory-corner ${cssState === "memory" ? "is-active" : ""}`} />

      {cssState === "greeting" && <div className="visitor-marker" />}
      {cssState === "reassurance" && <div className="reassurance-wash" />}
    </div>
  );
}
