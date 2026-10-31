import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.mjs";

// ---------- CONFIGURACIÓN INICIAL ----------
const VIDEO_SOURCES = [
  "https://media.w3.org/2010/05/sintel/trailer.mp4",
  "https://media.w3.org/2010/05/bunny/trailer.mp4",
  "https://media.w3.org/2010/05/video/movie_300.mp4",
  "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
  "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/friday.mp4"
];

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4], // Pulgar
  [0,5],[5,6],[6,7],[7,8], // Índice
  [5,9],[9,10],[10,11],[11,12], // Medio
  [9,13],[13,14],[14,15],[15,16], // Anular
  [13,17],[17,18],[18,19],[19,20], // Meñique
  [0,17] // Conexión palma
];

// Elementos del DOM
const mainVideo = document.getElementById("mainVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");
const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");
const gestureStatus = document.getElementById("gestureStatus");
const gestureHint = document.getElementById("gestureHint");
const configMenu = document.getElementById("configMenu");
const openConfigBtn = document.getElementById("openConfigBtn");
const closeConfigBtn = document.getElementById("closeConfig");
const camResSelect = document.getElementById("camRes");
const detectFpsSelect = document.getElementById("detectFps");
const sensitivityInput = document.getElementById("sensitivity");
const showSkeletonCheck = document.getElementById("showSkeleton");
const showCursorCheck = document.getElementById("showCursor");

// Estado global
let handLandmarker = null;
let videoIndex = 0;
let lastVideoTime = -1;
let lastDetectTime = 0;
let detectInterval = 1000 / 30; // 30 FPS por defecto
let sensitivity = 1.0;
let showSkeleton = true;
let showCursor = true;
let cursorX = window.innerWidth / 2;
let cursorY = window.innerHeight / 2;
let isPinching = false;
let pinchStartTime = 0;
let lastGesture = "";
let gestureCooldown = 0;
let heartPhotoTaken = false;

// ---------- INICIALIZACIÓN ----------
async function init() {
  // Ajustar tamaño de los canvas
  resizeCanvases();
  window.addEventListener("resize", resizeCanvases);

  // Cargar el primer video
  loadVideo(0);

  // Iniciar cámara
  await initCamera();

  // Iniciar MediaPipe
  await initHandLandmarker();

  // Iniciar bucle de detección
  requestAnimationFrame(detectLoop);

  // Eventos de UI
  openConfigBtn.addEventListener("click", () => configMenu.classList.remove("hidden"));
  closeConfigBtn.addEventListener("click", () => configMenu.classList.add("hidden"));

  camResSelect.addEventListener("change", async () => {
    await initCamera(); // Reiniciar cámara con nueva resolución
  });

  detectFpsSelect.addEventListener("change", (e) => {
    const fps = parseInt(e.target.value, 10);
    detectInterval = 1000 / fps;
  });

  sensitivityInput.addEventListener("input", (e) => {
    sensitivity = parseFloat(e.target.value);
  });

  showSkeletonCheck.addEventListener("change", (e) => {
    showSkeleton = e.target.checked;
  });

  showCursorCheck.addEventListener("change", (e) => {
    showCursor = e.target.checked;
  });
}

function resizeCanvases() {
  overlayCanvas.width = window.innerWidth;
  overlayCanvas.height = window.innerHeight;
  cameraCanvas.width = cameraVideo.videoWidth || 220;
  cameraCanvas.height = cameraVideo.videoHeight || 165;
}

// ---------- CARGA DE VIDEOS ----------
function loadVideo(index) {
  videoIndex = index;
  mainVideo.src = VIDEO_SOURCES[index];
  mainVideo.load();
  mainVideo.play().catch(() => {});
  mainVideo.onended = () => {
    // Avanzar al siguiente video automáticamente
    const next = (videoIndex + 1) % VIDEO_SOURCES.length;
    loadVideo(next);
  };
}

// ---------- CÁMARA ----------
async function initCamera() {
  const res = camResSelect.value.split("x");
  const width = parseInt(res[0], 10);
  const height = parseInt(res[1], 10);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width, height, facingMode: "user" },
    audio: false
  });

  cameraVideo.srcObject = stream;
  await cameraVideo.play();

  // Ajustar canvas de la cámara
  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
}

// ---------- MEDIAPIPE ----------
async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
}

// ---------- BUCLE DE DETECCIÓN ----------
function detectLoop() {
  const now = performance.now();

  // Control de FPS de detección
  if (now - lastDetectTime >= detectInterval && cameraVideo.readyState >= 2) {
    lastDetectTime = now;

    const results = handLandmarker.detectForVideo(cameraVideo, now);
    processResults(results);
  }

  requestAnimationFrame(detectLoop);
}

function processResults(results) {
  // Limpiar canvas
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  cameraCtx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  const landmarksList = results.landmarks || [];

  // Dibujar esqueleto en el canvas de la cámara (mini ventana)
  if (showSkeleton) {
    landmarksList.forEach((landmarks, handIndex) => {
      drawHand(cameraCtx, landmarks, cameraCanvas.width, cameraCanvas.height, handIndex);
    });
  }

  // Dibujar esqueleto en el canvas principal (superpuesto)
  if (showSkeleton) {
    landmarksList.forEach((landmarks, handIndex) => {
      drawHand(overlayCtx, landmarks, overlayCanvas.width, overlayCanvas.height, handIndex);
    });
  }

  // Detectar gestos y actualizar cursor
  if (landmarksList.length > 0) {
    gestureStatus.textContent = `Manos detectadas: ${landmarksList.length}`;
    handleGestures(landmarksList);
  } else {
    gestureStatus.textContent = "Esperando manos...";
    gestureHint.textContent = "";
  }

  // Dibujar cursor virtual
  if (showCursor && landmarksList.length > 0) {
    drawCursor();
  }
}

// ---------- DIBUJO DE MANOS ----------
function drawHand(ctx, landmarks, width, height, handIndex) {
  const color = handIndex === 0 ? "#00e5ff" : "#ff4081"; // Azul para una mano, rosa para la otra

  // Puntos
  landmarks.forEach((lm) => {
    const x = lm.x * width;
    const y = lm.y * height;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  });

  // Conexiones
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  CONNECTIONS.forEach(([start, end]) => {
    const p1 = landmarks[start];
    const p2 = landmarks[end];
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  });
}

// ---------- CURSOR VIRTUAL ----------
function drawCursor() {
  overlayCtx.beginPath();
  overlayCtx.arc(cursorX, cursorY, 12, 0, 2 * Math.PI);
  overlayCtx.fillStyle = "rgba(255, 255, 255, 0.8)";
  overlayCtx.fill();
  overlayCtx.strokeStyle = "#6c5ce7";
  overlayCtx.lineWidth = 3;
  overlayCtx.stroke();
}

// ---------- LÓGICA DE GESTOS ----------
function handleGestures(landmarksList) {
  // Actualizar cursor con la punta del dedo índice de la primera mano
  const primaryHand = landmarksList[0];
  const indexTip = primaryHand[8]; // Punta del índice
  cursorX = indexTip.x * window.innerWidth;
  cursorY = indexTip.y * window.innerHeight;

  // Detectar gesto de cada mano
  const gestures = landmarksList.map((landmarks) => recognizeGesture(landmarks));

  // Mostrar gesto principal
  const mainGesture = gestures[0] || "none";
  gestureHint.textContent = getGestureName(mainGesture);

  // Aplicar acción según el gesto (con cooldown para evitar repeticiones)
  const now = Date.now();
  if (now - gestureCooldown < 800) return; // 800 ms de cooldown

  // Gesto de corazón con dos manos
  if (landmarksList.length === 2 && isHeartGesture(landmarksList[0], landmarksList[1])) {
    if (!heartPhotoTaken) {
      takeHeartPhoto();
      heartPhotoTaken = true;
      setTimeout(() => (heartPhotoTaken = false), 5000); // Evitar múltiples fotos seguidas
    }
    gestureCooldown = now;
    return;
  }

  // Acciones según el gesto de la mano principal
  switch (mainGesture) {
    case "open_palm":
      togglePlayPause();
      break;
    case "fist":
      stopVideo();
      break;
    case "thumb_up":
      increaseVolume();
      break;
    case "thumb_down":
      decreaseVolume();
      break;
    case "pinch":
      // El gesto de pinza se usa para hacer clic en el menú (si está abierto)
      // o para buscar en la línea de tiempo (si está cerrado)
      if (!configMenu.classList.contains("hidden")) {
        simulateClick(cursorX, cursorY);
      } else {
        seekVideo(cursorX);
      }
      break;
    case "victory":
      // Abrir/cerrar menú de configuración
      configMenu.classList.toggle("hidden");
      break;
  }

  gestureCooldown = now;
}

// ---------- RECONOCIMIENTO DE GESTOS ----------
function recognizeGesture(landmarks) {
  // Calcular si los dedos están extendidos
  const fingers = {
    thumb: isFingerExtended(landmarks, 4, 2),
    index: isFingerExtended(landmarks, 8, 6),
    middle: isFingerExtended(landmarks, 12, 10),
    ring: isFingerExtended(landmarks, 16, 14),
    pinky: isFingerExtended(landmarks, 20, 18)
  };

  // Puño: todos los dedos cerrados
  if (!fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky) {
    return "fist";
  }

  // Palma abierta: todos los dedos extendidos
  if (fingers.index && fingers.middle && fingers.ring && fingers.pinky) {
    return "open_palm";
  }

  // Pulgar arriba: pulgar extendido, resto cerrados, y pulgar apuntando hacia arriba
  if (fingers.thumb && !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky) {
    const thumbTip = landmarks[4];
    const wrist = landmarks[0];
    if (thumbTip.y < wrist.y) return "thumb_up";
    else return "thumb_down";
  }

  // Pinza: pulgar e índice cerca
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const dist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  const handSize = getHandSize(landmarks);
  if (dist < 0.25 * handSize * sensitivity) {
    return "pinch";
  }

  // Victoria: índice y medio extendidos, resto cerrados
  if (fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) {
    return "victory";
  }

  return "none";
}

function isFingerExtended(landmarks, tipIndex, pipIndex) {
  // Comparar la distancia desde la muñeca (0) a la punta y a la articulación PIP
  const wrist = landmarks[0];
  const tip = landmarks[tipIndex];
  const pip = landmarks[pipIndex];

  const distTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
  const distPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);

  return distTip > distPip * 1.1; // Margen para evitar falsos positivos
}

function getHandSize(landmarks) {
  // Distancia entre la muñeca y la punta del dedo medio como referencia de tamaño
  const wrist = landmarks[0];
  const middleTip = landmarks[12];
  return Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y);
}

// ---------- DETECCIÓN DE CORAZÓN (DOS MANOS) ----------
function isHeartGesture(hand1, hand2) {
  // Puntas de los índices y pulgares de ambas manos
  const index1 = hand1[8];
  const thumb1 = hand1[4];
  const index2 = hand2[8];
  const thumb2 = hand2[4];

  // Distancias entre puntas correspondientes
  const distIndex = Math.hypot(index1.x - index2.x, index1.y - index2.y);
  const distThumb = Math.hypot(thumb1.x - thumb2.x, thumb1.y - thumb2.y);

  // Umbral ajustable (0.15 es bastante estricto; usamos 0.2 para más flexibilidad)
  return distIndex < 0.2 && distThumb < 0.2;
}

// ---------- ACCIONES DE VIDEO ----------
function togglePlayPause() {
  if (mainVideo.paused) {
    mainVideo.play();
  } else {
    mainVideo.pause();
  }
}

function stopVideo() {
  mainVideo.pause();
  mainVideo.currentTime = 0;
}

function increaseVolume() {
  mainVideo.volume = Math.min(1, mainVideo.volume + 0.1);
}

function decreaseVolume() {
  mainVideo.volume = Math.max(0, mainVideo.volume - 0.1);
}

function seekVideo(x) {
  // Mapear la posición X del cursor (0 a window.innerWidth) al tiempo del video
  const progress = x / window.innerWidth;
  mainVideo.currentTime = progress * mainVideo.duration;
}

// ---------- SIMULACIÓN DE CLIC (PARA EL MENÚ) ----------
function simulateClick(x, y) {
  // Crear un evento de clic en las coordenadas del cursor
  const element = document.elementFromPoint(x, y);
  if (element) {
    const clickEvent = new MouseEvent("click", {
      view: window,
      bubbles: true,
      cancelable: true,
      clientX: x,
      clientY: y
    });
    element.dispatchEvent(clickEvent);
  }
}

// ---------- FOTO DEL CORAZÓN ----------
function takeHeartPhoto() {
  // Crear un canvas temporal con el tamaño de la pantalla
  const photoCanvas = document.createElement("canvas");
  photoCanvas.width = window.innerWidth;
  photoCanvas.height = window.innerHeight;
  const ctx = photoCanvas.getContext("2d");

  // Dibujar el frame actual del video principal
  ctx.drawImage(mainVideo, 0, 0, photoCanvas.width, photoCanvas.height);

  // Superponer un degradado semitransparente para que el texto sea legible
  const gradient = ctx.createLinearGradient(0, 0, 0, photoCanvas.height);
  gradient.addColorStop(0, "rgba(0,0,0,0.4)");
  gradient.addColorStop(1, "rgba(0,0,0,0.8)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, photoCanvas.width, photoCanvas.height);

  // Texto principal
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 36px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Gracias por visitar nuestro stand", photoCanvas.width / 2, photoCanvas.height / 2 - 30);

  // Nombre de la escuela
  ctx.font = "bold 24px system-ui, sans-serif";
  ctx.fillStyle = "#6c5ce7";
  ctx.fillText(
    "CEJA LIDORO QUINTEROS SUBSEDE N°1 LOS RALOS",
    photoCanvas.width / 2,
    photoCanvas.height / 2 + 30
  );

  // Emoji de corazón
  ctx.font = "64px system-ui, sans-serif";
  ctx.fillText("❤️", photoCanvas.width / 2, photoCanvas.height / 2 + 110);

  // Convertir a imagen y descargar
  const dataURL = photoCanvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.download = `recuerdo-stand-${Date.now()}.png`;
  link.href = dataURL;
  link.click();

  // Mostrar mensaje en pantalla
  gestureHint.textContent = "📸 ¡Foto guardada! Gracias por visitarnos ❤️";
  setTimeout(() => {
    gestureHint.textContent = "";
  }, 4000);
}

// ---------- NOMBRES DE GESTOS ----------
function getGestureName(gesture) {
  const names = {
    open_palm: "✋ Palma abierta → Play/Pausa",
    fist: "✊ Puño → Detener",
    thumb_up: "👍 Pulgar arriba → Subir volumen",
    thumb_down: "👎 Pulgar abajo → Bajar volumen",
    pinch: "🤏 Pinza → Clic / Buscar",
    victory: "✌️ Victoria → Menú",
    heart: "❤️ Corazón → Foto",
    none: ""
  };
  return names[gesture] || "";
}

// ---------- INICIO ----------
init().catch((err) => {
  console.error("Error al iniciar:", err);
  alert("No se pudo iniciar la cámara o MediaPipe. Revisa los permisos.");
});
