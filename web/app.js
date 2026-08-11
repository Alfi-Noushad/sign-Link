const form = document.getElementById("translate-form");
const input = document.getElementById("translator-input");
const replayButton = document.getElementById("replay-button");
const resultsStrip = document.getElementById("results-strip");
const statusPill = document.getElementById("status-pill");
const fingerSpellNote = document.getElementById("finger-spell-note");
const clipCounter = document.getElementById("clip-counter");
const activeLabel = document.getElementById("active-label");
const sequenceTitle = document.getElementById("sequence-title");
const clipVideo = document.getElementById("clip-video");
const fallbackStage = document.getElementById("fallback-stage");
const fallbackLetter = document.getElementById("fallback-letter");
const fallbackLabel = document.getElementById("fallback-label");

const cameraVideo = document.getElementById("camera-video");
const cameraCanvas = document.getElementById("camera-canvas");
const cameraPlaceholder = document.getElementById("camera-placeholder");
const cameraState = document.getElementById("camera-state");
const liveSignLabel = document.getElementById("live-sign-label");
const liveSignMeta = document.getElementById("live-sign-meta");
const currentSignBadge = document.getElementById("current-sign-badge");
const confidenceBadge = document.getElementById("confidence-badge");
const detectorStatus = document.getElementById("detector-status");
const signOutput = document.getElementById("sign-output");
const predictionSpotlight = document.getElementById("prediction-spotlight");
const predictionSubtext = document.getElementById("prediction-subtext");
const topPredictions = document.getElementById("top-predictions");
const startCameraButton = document.getElementById("start-camera-button");
const stopCameraButton = document.getElementById("stop-camera-button");
const addCurrentButton = document.getElementById("add-current-button");
const spaceSignButton = document.getElementById("space-sign-button");
const backspaceSignButton = document.getElementById("backspace-sign-button");
const clearSignButton = document.getElementById("clear-sign-button");
const speakSignOutputButton = document.getElementById("speak-sign-output-button");
const sendToAudioButton = document.getElementById("send-to-audio-button");

const audioText = document.getElementById("audio-text");
const voiceSelect = document.getElementById("voice-select");
const voiceVolume = document.getElementById("voice-volume");
const speakAudioButton = document.getElementById("speak-audio-button");
const stopAudioButton = document.getElementById("stop-audio-button");
const fillAudioDemoButton = document.getElementById("fill-audio-demo-button");
const audioState = document.getElementById("audio-state");
const audioNote = document.getElementById("audio-note");
const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const trackedSections = ["overview", "translator-card", "detector-card", "audio-card"]
  .map((id) => document.getElementById(id))
  .filter(Boolean);
const ambientGrid = document.querySelector(".ambient-grid");
const parallaxShapes = Array.from(document.querySelectorAll("[data-parallax]"));
const revealNodes = Array.from(
  document.querySelectorAll("[data-reveal], .mini-stat"),
);

const playbackState = {
  clips: [],
  currentIndex: -1,
  timerId: null,
};

const detectorState = {
  stream: null,
  loopId: null,
  requestInFlight: false,
  running: false,
  available: true,
  unavailableReason: "",
  candidateLabel: null,
  candidateFrames: 0,
  releaseFrames: 0,
  readyToCommit: true,
  currentRawLabel: null,
  currentRawConfidence: 0,
  currentPredictions: [],
};

const DETECTOR_INTERVAL_MS = 420;
const DETECTOR_COMMIT_FRAMES = 4;
const DETECTOR_RELEASE_FRAMES = 2;
const NAV_HIGHLIGHT_MS = 2600;

let availableVoices = [];
let scrollTicking = false;
let volumeRestartTimer = null;
let navHighlightTimer = null;
let navHighlightFrame = null;

const speechState = {
  text: "",
  charIndex: 0,
  token: 0,
  speaking: false,
  utterance: null,
};

const navState = {
  lockedId: null,
};

function setStatus(message, tone = "default") {
  statusPill.textContent = message;
  statusPill.classList.remove("is-error", "is-active");

  if (tone === "error") {
    statusPill.classList.add("is-error");
  } else if (tone === "active") {
    statusPill.classList.add("is-active");
  }
}

function formatStat(key, value) {
  if (typeof value !== "number") {
    return value;
  }

  if (key === "model_accuracy") {
    return `${value.toFixed(1)}%`;
  }

  return value.toLocaleString();
}

async function loadStats() {
  try {
    const response = await fetch("/api/project-stats");
    const payload = await response.json();

    if (!payload.ok) {
      return;
    }

    Object.entries(payload.stats).forEach(([key, value]) => {
      document.querySelectorAll(`[data-stat="${key}"]`).forEach((node) => {
        node.textContent = formatStat(key, value);
      });
    });

    if (payload.stats.text_to_sign_notice) {
      fingerSpellNote.textContent = `${payload.stats.text_to_sign_notice} The translator will use generated letter cards until local clips are added.`;
    }

    if (payload.stats.sign_to_text_error) {
      detectorState.available = false;
      detectorState.unavailableReason = "Add sign_model.pkl to enable live detection.";
      startCameraButton.disabled = true;
      updateDetectorDisplay({
        liveLabel: "Model unavailable",
        liveMeta: "Live sign-to-text is disabled until a local model file is added",
        current: "-",
        confidence: "0.00",
        status: "Model unavailable",
        camera: "Model unavailable",
        spotlight: "-",
        spotlightNote: detectorState.unavailableReason,
        predictions: [],
      });
    }
  } catch (error) {
    console.error("Could not load stats", error);
  }
}

function clearScheduledFallback() {
  if (playbackState.timerId) {
    window.clearTimeout(playbackState.timerId);
    playbackState.timerId = null;
  }
}

function stopPlayback() {
  clearScheduledFallback();
  clipVideo.pause();
}

function markActiveChip(index) {
  document.querySelectorAll(".result-chip").forEach((chip, chipIndex) => {
    chip.classList.toggle("active", chipIndex === index);
  });
}

function renderChips(clips) {
  if (!clips.length) {
    resultsStrip.innerHTML = "";
    return;
  }

  resultsStrip.innerHTML = clips
    .map((clip) => `<span class="result-chip">${clip.label}</span>`)
    .join("");
}

function showFallbackClip(clip) {
  clipVideo.pause();
  clipVideo.removeAttribute("src");
  clipVideo.load();
  clipVideo.classList.remove("is-visible");
  fallbackStage.classList.remove("is-hidden");

  fallbackLetter.textContent = clip.label.charAt(0);
  fallbackLabel.textContent = `${clip.label} rendered as a generated fallback`;
  activeLabel.textContent = "Generated clip";
  sequenceTitle.textContent = clip.label;

  clearScheduledFallback();
  playbackState.timerId = window.setTimeout(() => {
    playClip(playbackState.currentIndex + 1);
  }, clip.duration_ms || 850);
}

async function showVideoClip(clip) {
  fallbackStage.classList.add("is-hidden");
  clipVideo.classList.add("is-visible");
  clipVideo.src = clip.url;
  clipVideo.load();

  activeLabel.textContent = "Dataset clip";
  sequenceTitle.textContent = clip.label;

  try {
    await clipVideo.play();
  } catch (error) {
    console.error("Could not autoplay clip", error);
    setStatus("A clip could not autoplay, skipping ahead.", "error");
    playClip(playbackState.currentIndex + 1);
  }
}

function playClip(index) {
  clearScheduledFallback();

  if (!playbackState.clips.length) {
    setStatus("Generate a sequence to begin.", "default");
    return;
  }

  if (index >= playbackState.clips.length) {
    playbackState.currentIndex = -1;
    markActiveChip(-1);
    activeLabel.textContent = "Finished";
    sequenceTitle.textContent = "Sequence complete";
    clipCounter.textContent = `${playbackState.clips.length} clips played`;
    setStatus("Playback finished.", "default");
    return;
  }

  playbackState.currentIndex = index;
  const clip = playbackState.clips[index];

  markActiveChip(index);
  clipCounter.textContent = `Clip ${index + 1} of ${playbackState.clips.length}`;
  setStatus("Playing your translation", "active");

  if (clip.type === "generated") {
    showFallbackClip(clip);
    return;
  }

  showVideoClip(clip);
}

async function submitTranslation(text) {
  stopPlayback();
  setStatus("Building the sequence...", "active");

  try {
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Translation failed.");
    }

    playbackState.clips = payload.clips;
    renderChips(payload.clips);

    if (payload.finger_spelled_words.length) {
      fingerSpellNote.textContent = `Finger-spelled: ${payload.finger_spelled_words.join(", ")}`;
    } else {
      fingerSpellNote.textContent = "Every word matched a sign clip in the dataset.";
    }

    playClip(0);
  } catch (error) {
    playbackState.clips = [];
    renderChips([]);
    clipCounter.textContent = "No sequence yet";
    activeLabel.textContent = "Idle";
    sequenceTitle.textContent = "Your translation will play here";
    fingerSpellNote.textContent = error.message;
    setStatus(error.message, "error");
  }
}

function renderTopPredictions(predictions = []) {
  if (!predictions.length) {
    topPredictions.innerHTML = '<span class="prediction-chip">Waiting for a live prediction</span>';
    return;
  }

  topPredictions.innerHTML = predictions
    .map(
      (prediction, index) =>
        `<span class="prediction-chip${index === 0 ? " active" : ""}">${prediction.label} <small>${Math.round(prediction.confidence * 100)}%</small></span>`,
    )
    .join("");
}

function updateDetectorDisplay({
  liveLabel = "No hand detected",
  liveMeta = "Waiting for camera access",
  current = "-",
  confidence = "0.00",
  status = "Ready",
  camera = null,
  spotlight = "-",
  spotlightNote = "Waiting for camera access",
  predictions = null,
} = {}) {
  liveSignLabel.textContent = liveLabel;
  liveSignMeta.textContent = liveMeta;
  currentSignBadge.textContent = current;
  confidenceBadge.textContent = confidence;
  detectorStatus.textContent = status;
  predictionSpotlight.textContent = spotlight;
  predictionSubtext.textContent = spotlightNote;

  if (camera !== null) {
    cameraState.textContent = camera;
  }

  if (predictions !== null) {
    renderTopPredictions(predictions);
  }
}

function resetDetectorState() {
  detectorState.candidateLabel = null;
  detectorState.candidateFrames = 0;
  detectorState.releaseFrames = 0;
  detectorState.readyToCommit = true;
  detectorState.currentRawLabel = null;
  detectorState.currentRawConfidence = 0;
  detectorState.currentPredictions = [];
}

function appendToSignOutput(fragment) {
  signOutput.value += fragment;
  signOutput.dispatchEvent(new Event("input"));
}

function addSpaceToSignOutput() {
  if (signOutput.value && !signOutput.value.endsWith(" ")) {
    appendToSignOutput(" ");
  }
}

function useCurrentPrediction() {
  if (!detectorState.currentRawLabel) {
    detectorStatus.textContent = "No live prediction yet";
    return;
  }

  appendToSignOutput(detectorState.currentRawLabel);
  detectorStatus.textContent = `Added ${detectorState.currentRawLabel} manually`;
  detectorState.readyToCommit = false;
  detectorState.releaseFrames = 0;
}

function handleNoHandState() {
  detectorState.currentRawLabel = null;
  detectorState.currentRawConfidence = 0;
  detectorState.currentPredictions = [];
  detectorState.candidateLabel = null;
  detectorState.candidateFrames = 0;
  detectorState.releaseFrames += 1;

  if (detectorState.releaseFrames >= DETECTOR_RELEASE_FRAMES) {
    detectorState.readyToCommit = true;
  }

  updateDetectorDisplay({
    liveLabel: "No hand detected",
    liveMeta: detectorState.readyToCommit
      ? "Show one hand clearly to the camera"
      : "Hand released, ready for the next sign",
    current: "-",
    confidence: "0.00",
    status: detectorState.readyToCommit ? "Waiting for the next sign" : "Release complete",
    camera: "Camera live",
    spotlight: "-",
    spotlightNote: detectorState.readyToCommit
      ? "Waiting for a confident live guess"
      : "Release your hand before the next sign",
    predictions: [],
  });
}

function handlePrediction(payload) {
  const handDetected = Boolean(payload.hand_detected);
  const committedLabel = payload.label ? String(payload.label).toUpperCase() : null;
  const rawLabel = payload.raw_label ? String(payload.raw_label).toUpperCase() : null;
  const rawConfidence = Number(payload.raw_confidence || 0);
  const predictions = Array.isArray(payload.top_predictions)
    ? payload.top_predictions.map((prediction) => ({
        label: String(prediction.label || "").toUpperCase(),
        confidence: Number(prediction.confidence || 0),
      }))
    : [];

  detectorState.currentRawLabel = rawLabel;
  detectorState.currentRawConfidence = rawConfidence;
  detectorState.currentPredictions = predictions;

  if (!handDetected) {
    handleNoHandState();
    return;
  }

  if (!rawLabel) {
    updateDetectorDisplay({
      liveLabel: "Hand detected",
      liveMeta: "The model cannot identify a clear sign yet",
      current: "-",
      confidence: "0.00",
      status: "Try a clearer hand pose",
      camera: "Camera live",
      spotlight: "?",
      spotlightNote: "Hand found, but prediction is still unclear",
      predictions,
    });
    return;
  }

  detectorState.releaseFrames = 0;

  const spotlightNote = `${Math.round(rawConfidence * 100)}% confidence`;

  if (!detectorState.readyToCommit) {
    updateDetectorDisplay({
      liveLabel: `Model sees ${rawLabel}`,
      liveMeta: "Prediction is visible, but auto-add is waiting for hand release",
      current: rawLabel,
      confidence: rawConfidence.toFixed(2),
      status: "Move your hand away before the next letter",
      camera: "Camera live",
      spotlight: rawLabel,
      spotlightNote,
      predictions,
    });
    return;
  }

  if (!committedLabel) {
    detectorState.candidateLabel = null;
    detectorState.candidateFrames = 0;
    updateDetectorDisplay({
      liveLabel: `Model sees ${rawLabel}`,
      liveMeta: "Visible prediction only. Hold steadier or use the current guess manually.",
      current: rawLabel,
      confidence: rawConfidence.toFixed(2),
      status: "Prediction not confident enough to auto-add",
      camera: "Camera live",
      spotlight: rawLabel,
      spotlightNote,
      predictions,
    });
    return;
  }

  if (detectorState.candidateLabel === committedLabel) {
    detectorState.candidateFrames += 1;
  } else {
    detectorState.candidateLabel = committedLabel;
    detectorState.candidateFrames = 1;
  }

  if (detectorState.candidateFrames >= DETECTOR_COMMIT_FRAMES) {
    appendToSignOutput(committedLabel);
    detectorState.readyToCommit = false;
    detectorState.candidateLabel = null;
    detectorState.candidateFrames = 0;

    updateDetectorDisplay({
      liveLabel: `Added ${committedLabel}`,
      liveMeta: "Auto-added from a stable prediction",
      current: committedLabel,
      confidence: rawConfidence.toFixed(2),
      status: `Committed ${committedLabel}`,
      camera: "Camera live",
      spotlight: committedLabel,
      spotlightNote: `Auto-added at ${Math.round(rawConfidence * 100)}% confidence`,
      predictions,
    });
    return;
  }

  updateDetectorDisplay({
    liveLabel: `Model sees ${committedLabel}`,
    liveMeta: "Keep the same sign steady for auto-add",
    current: committedLabel,
    confidence: rawConfidence.toFixed(2),
    status: `Hold ${committedLabel} steady`,
    camera: "Camera live",
    spotlight: committedLabel,
    spotlightNote: `${DETECTOR_COMMIT_FRAMES - detectorState.candidateFrames} more stable frame(s) needed`,
    predictions,
  });
}

async function captureAndPredict() {
  if (!detectorState.running || detectorState.requestInFlight) {
    return;
  }

  if (cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  detectorState.requestInFlight = true;

  try {
    cameraCanvas.width = cameraVideo.videoWidth || 640;
    cameraCanvas.height = cameraVideo.videoHeight || 480;

    const context = cameraCanvas.getContext("2d");
    context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

    const image = cameraCanvas.toDataURL("image/jpeg", 0.84);
    const response = await fetch("/api/predict-sign", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image }),
    });

    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    handlePrediction(payload);
  } catch (error) {
    updateDetectorDisplay({
      liveLabel: "Prediction error",
      liveMeta: error.message,
      current: "-",
      confidence: "0.00",
      status: "Check the model or camera feed",
      camera: detectorState.running ? "Camera live" : "Camera idle",
      spotlight: "!",
      spotlightNote: "Prediction request failed",
      predictions: detectorState.currentPredictions,
    });
  } finally {
    detectorState.requestInFlight = false;
  }
}

async function startCamera() {
  if (!detectorState.available) {
    updateDetectorDisplay({
      liveLabel: "Model unavailable",
      liveMeta: "This project needs sign_model.pkl for live sign detection",
      current: "-",
      confidence: "0.00",
      status: "Model unavailable",
      camera: "Model unavailable",
      spotlight: "-",
      spotlightNote: detectorState.unavailableReason || "Add sign_model.pkl to enable live detection.",
      predictions: [],
    });
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    updateDetectorDisplay({
      liveLabel: "Camera unsupported",
      liveMeta: "This browser does not support webcam capture",
      status: "Camera unavailable",
      camera: "Camera unavailable",
      spotlight: "-",
      spotlightNote: "Webcam access is not supported here",
      predictions: [],
    });
    return;
  }

  if (detectorState.running) {
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    detectorState.stream = stream;
    cameraVideo.srcObject = stream;
    await cameraVideo.play();

    cameraPlaceholder.classList.add("is-hidden");
    detectorState.running = true;
    resetDetectorState();
    updateDetectorDisplay({
      liveLabel: "Camera connected",
      liveMeta: "Show one sign at a time",
      status: "Starting detector",
      camera: "Camera live",
      spotlight: "-",
      spotlightNote: "Live prediction will appear here",
      predictions: [],
    });

    detectorState.loopId = window.setInterval(captureAndPredict, DETECTOR_INTERVAL_MS);
  } catch (error) {
    updateDetectorDisplay({
      liveLabel: "Camera blocked",
      liveMeta: "Allow camera access in the browser to use sign-to-text",
      status: "Permission needed",
      camera: "Camera blocked",
      spotlight: "-",
      spotlightNote: "Camera permission is required",
      predictions: [],
    });
  }
}

function stopCamera() {
  if (detectorState.loopId) {
    window.clearInterval(detectorState.loopId);
    detectorState.loopId = null;
  }

  if (detectorState.stream) {
    detectorState.stream.getTracks().forEach((track) => track.stop());
    detectorState.stream = null;
  }

  detectorState.running = false;
  detectorState.requestInFlight = false;
  cameraVideo.srcObject = null;
  cameraPlaceholder.classList.remove("is-hidden");
  resetDetectorState();

  updateDetectorDisplay({
    liveLabel: "No hand detected",
    liveMeta: "Waiting for camera access",
    current: "-",
    confidence: "0.00",
    status: "Ready",
    camera: "Camera idle",
    spotlight: "-",
    spotlightNote: "Waiting for camera access",
    predictions: [],
  });
}

function populateVoices() {
  if (!("speechSynthesis" in window)) {
    audioState.textContent = "Speech unsupported";
    audioNote.textContent = "Your browser does not support built-in speech synthesis.";
    voiceSelect.innerHTML = "<option>Unavailable</option>";
    voiceSelect.disabled = true;
    voiceVolume.disabled = true;
    speakAudioButton.disabled = true;
    stopAudioButton.disabled = true;
    return;
  }

  availableVoices = window.speechSynthesis.getVoices().filter((voice) => voice.lang.startsWith("en"));
  if (!availableVoices.length) {
    availableVoices = window.speechSynthesis.getVoices();
  }

  voiceSelect.innerHTML = "";

  availableVoices.forEach((voice, index) => {
    const option = document.createElement("option");
    option.value = voice.voiceURI;
    option.textContent = `${voice.name} (${voice.lang})`;
    if (index === 0) {
      option.selected = true;
    }
    voiceSelect.appendChild(option);
  });

  audioState.textContent = availableVoices.length ? "Speech ready" : "Speech loading";
}

function setActiveNavLink(activeId) {
  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${activeId}`);
  });
}

function navHighlightTargets(targetId) {
  if (targetId === "overview") {
    return Array.from(
      document.querySelectorAll("#overview .intro-card, #overview .info-card"),
    );
  }

  const target = document.getElementById(targetId);
  return target ? [target] : [];
}

function triggerNavTargetHighlight(targetId) {
  const targets = navHighlightTargets(targetId);
  if (!targets.length) {
    return;
  }

  document.querySelectorAll(".is-nav-targeted").forEach((node) => {
    node.classList.remove("is-nav-targeted");
  });

  clearTimeout(navHighlightTimer);
  if (navHighlightFrame) {
    window.cancelAnimationFrame(navHighlightFrame);
    navHighlightFrame = null;
  }

  navHighlightFrame = window.requestAnimationFrame(() => {
    targets.forEach((node) => {
      node.classList.add("is-nav-targeted");
    });
    navHighlightFrame = null;
  });

  navHighlightTimer = window.setTimeout(() => {
    targets.forEach((node) => node.classList.remove("is-nav-targeted"));
  }, NAV_HIGHLIGHT_MS);
}

function sectionViewportMetrics(section, viewportHeight) {
  const rect = section.getBoundingClientRect();
  const visibleTop = Math.max(rect.top, 0);
  const visibleBottom = Math.min(rect.bottom, viewportHeight);
  const visibleHeight = Math.max(0, visibleBottom - visibleTop);
  const effectiveHeight = Math.max(1, Math.min(rect.height, viewportHeight));

  return {
    rect,
    visibleHeight,
    visibleRatio: visibleHeight / effectiveHeight,
  };
}

function deriveActiveSectionId() {
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
  const focusY = Math.min(viewportHeight * 0.28, 220);
  let activeId = "overview";
  let bestScore = Number.NEGATIVE_INFINITY;

  trackedSections.forEach((section) => {
    const { rect, visibleHeight } = sectionViewportMetrics(section, viewportHeight);

    if (visibleHeight <= 0) {
      return;
    }

    const topDistance = Math.abs(rect.top - focusY);
    const visibilityScore = visibleHeight * 1.35;
    const anchorScore = Math.max(0, 320 - topDistance);
    const score = visibilityScore + anchorScore;

    if (score > bestScore) {
      bestScore = score;
      activeId = section.id;
    }
  });

  return activeId;
}

function setActiveNav() {
  let activeId = deriveActiveSectionId();

  if (navState.lockedId) {
    const lockedSection = document.getElementById(navState.lockedId);
    if (lockedSection) {
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
      const { rect, visibleRatio } = sectionViewportMetrics(lockedSection, viewportHeight);
      const focusBand = viewportHeight * 0.34;
      const intersectsFocusBand = rect.top < focusBand && rect.bottom > focusBand;

      if (visibleRatio >= 0.22 || intersectsFocusBand) {
        activeId = navState.lockedId;
      } else if (activeId !== navState.lockedId) {
        navState.lockedId = null;
      }
    } else {
      navState.lockedId = null;
    }
  }

  setActiveNavLink(activeId);
}

function baseRotationForShape(element) {
  if (element.classList.contains("shape-diamond")) {
    return 45;
  }
  if (element.classList.contains("shape-bar")) {
    return -12;
  }
  if (element.classList.contains("shape-orbit")) {
    return 8;
  }
  return 0;
}

function rotationFactorForShape(element) {
  if (element.classList.contains("shape-diamond")) {
    return -0.58;
  }
  if (element.classList.contains("shape-bar")) {
    return -0.52;
  }
  if (element.classList.contains("shape-orbit")) {
    return 0.68;
  }
  return 0.42;
}

function revealAnchorForNode(node) {
  return node.closest(".hero-stats, .intro-shell, .info-grid") || node;
}

function updateRevealProgress() {
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
  const progressCache = new Map();

  revealNodes.forEach((node) => {
    const anchor = revealAnchorForNode(node);
    let progress = progressCache.get(anchor);

    if (progress === undefined) {
      const rect = anchor.getBoundingClientRect();
      const visibleTop = Math.max(rect.top, 0);
      const visibleBottom = Math.min(rect.bottom, viewportHeight);
      const visibleHeight = Math.max(0, visibleBottom - visibleTop);
      const targetVisible = Math.max(1, Math.min(rect.height, viewportHeight * 0.68));
      progress = Math.max(0, Math.min(1, visibleHeight / targetVisible));
      progressCache.set(anchor, progress);
    }

    node.style.setProperty("--reveal-progress", progress.toFixed(3));
  });
}

function updateScrollEffects() {
  const scrollY = window.scrollY;
  document.body.classList.toggle("nav-scrolled", scrollY > 36);

  if (ambientGrid) {
    ambientGrid.style.transform = `translate3d(${scrollY * -0.03}px, ${scrollY * -0.06}px, 0) rotate(${scrollY * 0.012}deg)`;
  }

  parallaxShapes.forEach((shape) => {
    const depth = Number(shape.dataset.parallax || 0.15);
    const driftX = shape.classList.contains("shape-diamond")
      ? scrollY * depth * 0.08
      : shape.classList.contains("shape-orbit")
        ? scrollY * depth * -0.05
        : scrollY * depth * 0.03;
    const translateY = scrollY * depth * 0.42;
    const rotate = baseRotationForShape(shape) + scrollY * depth * rotationFactorForShape(shape);
    shape.style.transform = `translate3d(${driftX}px, ${translateY}px, 0) rotate(${rotate}deg)`;
  });

  updateRevealProgress();
  setActiveNav();
  scrollTicking = false;
}

function requestScrollEffects() {
  if (scrollTicking) {
    return;
  }
  scrollTicking = true;
  window.requestAnimationFrame(updateScrollEffects);
}

function currentVolumeValue() {
  return Number(voiceVolume.value || 1);
}

function currentVolumePercent() {
  return Math.round(currentVolumeValue() * 100);
}

function clearVolumeRestartTimer() {
  if (volumeRestartTimer) {
    window.clearTimeout(volumeRestartTimer);
    volumeRestartTimer = null;
  }
}

function resetSpeechState() {
  clearVolumeRestartTimer();
  speechState.token += 1;
  speechState.text = "";
  speechState.charIndex = 0;
  speechState.speaking = false;
  speechState.utterance = null;
}

function normalizedResumeIndex(text, charIndex = 0) {
  const boundedIndex = Math.max(0, Math.min(charIndex, text.length));
  const tail = text.slice(boundedIndex);
  const leadingWhitespace = tail.match(/^\s*/)?.[0].length || 0;
  return Math.min(text.length, boundedIndex + leadingWhitespace);
}

function speakText(text, { resumeFromCharIndex = 0, reason = "default" } = {}) {
  const fullMessage = String(text || "").trim();
  if (!fullMessage) {
    audioState.textContent = "No text to speak";
    audioNote.textContent = "Type some text or generate text from signs first.";
    resetSpeechState();
    return;
  }

  if (!("speechSynthesis" in window)) {
    audioState.textContent = "Speech unsupported";
    audioNote.textContent = "Your browser does not support built-in speech synthesis.";
    resetSpeechState();
    return;
  }

  const absoluteStart = normalizedResumeIndex(fullMessage, resumeFromCharIndex);
  const message = fullMessage.slice(absoluteStart);

  if (!message.trim()) {
    audioState.textContent = "Speech ready";
    audioNote.textContent = "Playback finished.";
    resetSpeechState();
    return;
  }

  const token = speechState.token + 1;
  speechState.token = token;
  speechState.text = fullMessage;
  speechState.charIndex = absoluteStart;
  speechState.speaking = false;
  speechState.utterance = null;
  clearVolumeRestartTimer();
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(message);
  speechState.utterance = utterance;
  const selectedVoice = availableVoices.find((voice) => voice.voiceURI === voiceSelect.value);
  if (selectedVoice) {
    utterance.voice = selectedVoice;
    utterance.lang = selectedVoice.lang;
  }

  utterance.rate = 1;
  utterance.volume = currentVolumeValue();
  utterance.onstart = () => {
    if (speechState.token !== token) {
      return;
    }
    speechState.speaking = true;
    audioState.textContent = "Speaking";
    audioNote.textContent = reason === "volume-change"
      ? `Continuing at ${currentVolumePercent()}% volume.`
      : `Playing "${fullMessage.slice(0, 60)}${fullMessage.length > 60 ? "..." : ""}"`;
  };
  utterance.onboundary = (event) => {
    if (speechState.token !== token || typeof event.charIndex !== "number") {
      return;
    }
    speechState.charIndex = Math.max(speechState.charIndex, absoluteStart + event.charIndex);
  };
  utterance.onend = () => {
    if (speechState.token !== token) {
      return;
    }
    speechState.speaking = false;
    speechState.charIndex = fullMessage.length;
    speechState.utterance = null;
    audioState.textContent = "Speech ready";
    audioNote.textContent = "Playback finished.";
  };
  utterance.onerror = () => {
    if (speechState.token !== token) {
      return;
    }
    speechState.speaking = false;
    speechState.utterance = null;
    audioState.textContent = "Speech error";
    audioNote.textContent = "The browser could not play the selected voice.";
  };

  window.speechSynthesis.speak(utterance);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();

  if (!text) {
    setStatus("Enter some text first.", "error");
    input.focus();
    return;
  }

  submitTranslation(text);
});

replayButton.addEventListener("click", () => {
  if (!playbackState.clips.length) {
    setStatus("Generate a sequence before replaying it.", "error");
    return;
  }

  playClip(0);
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.sample || "";
    submitTranslation(input.value);
  });
});

clipVideo.addEventListener("ended", () => {
  playClip(playbackState.currentIndex + 1);
});

clipVideo.addEventListener("error", () => {
  setStatus("One clip could not be played, moving to the next.", "error");
  playClip(playbackState.currentIndex + 1);
});

startCameraButton.addEventListener("click", startCamera);
stopCameraButton.addEventListener("click", stopCamera);
addCurrentButton.addEventListener("click", useCurrentPrediction);

spaceSignButton.addEventListener("click", () => {
  addSpaceToSignOutput();
  detectorStatus.textContent = "Space added";
});

backspaceSignButton.addEventListener("click", () => {
  signOutput.value = signOutput.value.slice(0, -1);
  detectorStatus.textContent = "Removed last character";
});

clearSignButton.addEventListener("click", () => {
  signOutput.value = "";
  detectorStatus.textContent = "Cleared detected text";
});

speakSignOutputButton.addEventListener("click", () => {
  speakText(signOutput.value);
});

sendToAudioButton.addEventListener("click", () => {
  audioText.value = signOutput.value;
  audioText.focus();
  audioState.textContent = "Text imported";
  audioNote.textContent = "Detected sign text was copied into the audio studio.";
});

speakAudioButton.addEventListener("click", () => {
  speakText(audioText.value);
});

voiceVolume.addEventListener("input", () => {
  const volumePercent = currentVolumePercent();

  if ("speechSynthesis" in window && speechState.speaking && speechState.text) {
    if (speechState.utterance) {
      speechState.utterance.volume = currentVolumeValue();
    }
    audioState.textContent = "Volume updating";
    audioNote.textContent = `Applying ${volumePercent}% volume...`;
    clearVolumeRestartTimer();
    volumeRestartTimer = window.setTimeout(() => {
      if (!speechState.speaking || !speechState.text) {
        return;
      }
      speakText(speechState.text, {
        resumeFromCharIndex: speechState.charIndex,
        reason: "volume-change",
      });
    }, 280);
    return;
  }

  audioState.textContent = "Volume ready";
  audioNote.textContent = `Volume set to ${volumePercent}%.`;
});

stopAudioButton.addEventListener("click", () => {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  resetSpeechState();
  audioState.textContent = "Speech stopped";
  audioNote.textContent = "Playback was stopped.";
});

fillAudioDemoButton.addEventListener("click", () => {
  audioText.value = "This project can read simple text with the browser voice.";
  audioState.textContent = "Demo loaded";
  audioNote.textContent = "Demo text is ready to play.";
});

revealNodes.forEach((node) => {
  node.setAttribute("data-reveal", "");
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    const targetId = link.getAttribute("href")?.slice(1);
    if (!targetId) {
      return;
    }
    navState.lockedId = targetId;
    setActiveNavLink(targetId);
    triggerNavTargetHighlight(targetId);
    requestScrollEffects();
  });
});

window.addEventListener("scroll", requestScrollEffects, { passive: true });
window.addEventListener("resize", requestScrollEffects);

window.addEventListener("beforeunload", () => {
  stopCamera();
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  resetSpeechState();
});

if ("speechSynthesis" in window) {
  populateVoices();
  window.speechSynthesis.onvoiceschanged = populateVoices;
}

loadStats();
resetDetectorState();
renderTopPredictions([]);
updateScrollEffects();
