document.addEventListener("DOMContentLoaded", () => {
    const socket = io({ transports: ["websocket", "polling"] });

    const videoElement = document.getElementById("videoElement");
    const overlayCanvas = document.getElementById("overlayCanvas");
    const overlayCtx = overlayCanvas.getContext("2d");

    const startStopButton = document.getElementById("startStopButton");
    const retryButton = document.getElementById("retryButton");
    const smoothToggle = document.getElementById("smoothToggle");
    const consentCheckbox = document.getElementById("consentCheckbox");
    const clearEmotionMemoryButton = document.getElementById("clearEmotionMemoryButton");

    const emotionStatus = document.getElementById("emotionStatus");
    const confidenceNotice = document.getElementById("confidenceNotice");
    const confidenceMeterFill = document.getElementById("confidenceMeterFill");
    const streamStatus = document.getElementById("streamStatus");
    const privacyNotice = document.getElementById("privacyNotice");

    const loadingSpinner = document.getElementById("loadingSpinner");
    const errorMessage = document.getElementById("errorMessage");
    const errorText = document.getElementById("errorText");

    const socketState = document.getElementById("socketState");
    const backendStatus = document.getElementById("backendStatus");
    const readinessStatus = document.getElementById("readinessStatus");
    const trackerStatus = document.getElementById("trackerStatus");

    const chatbox = document.getElementById("chatbox");
    const messageInput = document.getElementById("messageInput");
    const sendMessageButton = document.getElementById("sendMessageButton");

    const emotionColors = {
        happy: "#4CAF50",
        sad: "#2196F3",
        angry: "#F44336",
        surprise: "#FF9800",
        neutral: "#9E9E9E",
        fear: "#673AB7",
        disgust: "#3F51B5",
        unknown: "#9d7cc4",
        error: "#be185d",
    };

    let localStream = null;
    let sendingFrames = false;
    let aiTypingIndicatorElement = null;

    let currentTargetFps = 5;
    let frameIntervalMs = 1000 / currentTargetFps;
    let currentJpegQuality = 0.7;
    let lastMetricsSnapshot = null;

    const adaptiveStreaming = {
        enabled: true,
        minFps: 3,
        maxFps: 8,
        minJpegQuality: 0.55,
        maxJpegQuality: 0.85,
    };
    const LOW_CONFIDENCE_FALLBACK_THRESHOLD = 0.55;

    const frameCaptureCanvas = document.createElement("canvas");
    const frameCaptureCtx = frameCaptureCanvas.getContext("2d");

    const emotionChart = new Chart(document.getElementById("emotionChart"), {
        type: "radar",
        data: {
            labels: ["Happy", "Sad", "Angry", "Surprise", "Neutral", "Fear", "Disgust"],
            datasets: [
                {
                    label: "Emotion Profile",
                    data: [0, 0, 0, 0, 0, 0, 0],
                    fill: true,
                    backgroundColor: "rgba(168, 85, 247, 0.23)",
                    borderColor: "rgba(147, 51, 234, 0.95)",
                    pointBorderColor: "#ffffff",
                    pointBackgroundColor: "rgba(236, 72, 153, 0.95)",
                    borderWidth: 2,
                },
            ],
        },
        options: {
            animation: { duration: 180 },
            scales: {
                r: {
                    beginAtZero: true,
                    min: 0,
                    max: 100,
                    ticks: { display: false },
                    pointLabels: {
                        color: "#6f4e8c",
                        font: { family: "Manrope", size: 11, weight: "600" },
                    },
                    grid: {
                        color: "rgba(143, 76, 201, 0.19)",
                    },
                    angleLines: {
                        color: "rgba(143, 76, 201, 0.19)",
                    },
                },
            },
            plugins: { legend: { display: false } },
        },
    });
    let currentEmotionData = [0, 0, 0, 0, 0, 0, 0];

    function setChipStatus(element, text, level) {
        if (!element) return;
        element.textContent = text;
        element.classList.remove("ok", "warn", "error");
        if (level) {
            element.classList.add(level);
        }
    }

    function getEmotionColor(emotion) {
        return emotionColors[(emotion || "unknown").toLowerCase()] || emotionColors.unknown;
    }

    function updateEmotionTheme(emotion) {
        const safeEmotion = (emotion || "neutral").toLowerCase();
        document.body.setAttribute("data-emotion", safeEmotion);
    }

    function updateConfidenceMeter(confidence, isLowConfidence = false) {
        if (!confidenceMeterFill) return;
        const clamped = Math.max(0, Math.min(1, Number(confidence || 0)));
        confidenceMeterFill.style.width = `${(clamped * 100).toFixed(1)}%`;
        if (isLowConfidence) {
            confidenceMeterFill.style.background = "linear-gradient(90deg, #be185d 0%, #ec4899 95%)";
        } else {
            confidenceMeterFill.style.background = "linear-gradient(90deg, #a855f7 0%, #ec4899 95%)";
        }
    }

    function showLoading() {
        loadingSpinner.style.display = "flex";
        errorMessage.style.display = "none";
        videoElement.style.opacity = "0.55";
    }

    function hideLoading() {
        loadingSpinner.style.display = "none";
        videoElement.style.opacity = "1";
    }

    function showError(message) {
        hideLoading();
        errorMessage.style.display = "flex";
        errorText.textContent = message;
        startStopButton.textContent = "Start Camera";
    }

    function hideError() {
        errorMessage.style.display = "none";
    }

    function updateStreamStatus(extraText = "") {
        if (!streamStatus) return;
        const qualityPercent = Math.round(currentJpegQuality * 100);
        const suffix = extraText ? ` | ${extraText}` : "";
        streamStatus.textContent = `Stream: ${currentTargetFps.toFixed(0)} FPS | JPEG ${qualityPercent}%${suffix}`;
    }

    function emitConsentState() {
        if (!socket.connected || !consentCheckbox) return;
        socket.emit("camera_consent", { consent: Boolean(consentCheckbox.checked) });
    }

    function updateCanvasDimensions() {
        if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
            overlayCanvas.width = videoElement.videoWidth;
            overlayCanvas.height = videoElement.videoHeight;
            frameCaptureCanvas.width = videoElement.videoWidth;
            frameCaptureCanvas.height = videoElement.videoHeight;
        }
    }

    async function pollBackendHealth() {
        try {
            const [healthRes, readinessRes] = await Promise.all([
                fetch("/api/health", { cache: "no-store" }),
                fetch("/api/readiness", { cache: "no-store" }),
            ]);

            if (!healthRes.ok) {
                setChipStatus(backendStatus, `HTTP ${healthRes.status}`, "warn");
                return;
            }
            const health = await healthRes.json();
            const runtime = health.runtime || {};
            const checks = health.checks || {};

            const device = runtime.gpu_available ? "GPU" : "CPU";
            const backendLevel = health.status === "ok" ? "ok" : "warn";
            setChipStatus(backendStatus, `${device} ${String(health.status || "unknown").toUpperCase()}`, backendLevel);

            if (readinessRes.ok) {
                const readiness = await readinessRes.json();
                const ready = Boolean(readiness.ready);
                setChipStatus(readinessStatus, ready ? "Ready" : "Degraded", ready ? "ok" : "warn");
            } else {
                setChipStatus(readinessStatus, "Unavailable", "warn");
            }

            if (!checks.detector_ready) {
                setChipStatus(backendStatus, `${device} Detector Missing`, "error");
            }
        } catch (_err) {
            setChipStatus(backendStatus, "Offline", "error");
            setChipStatus(readinessStatus, "Unknown", "warn");
        }
    }

    async function fetchAndAdaptStreamMetrics() {
        if (!adaptiveStreaming.enabled || !sendingFrames) return;

        try {
            const response = await fetch("/api/metrics", { cache: "no-store" });
            if (!response.ok) return;
            const metrics = await response.json();
            if (!metrics || typeof metrics !== "object") return;

            if (lastMetricsSnapshot) {
                const receivedDelta = Math.max(
                    0,
                    (metrics.frames_received || 0) - (lastMetricsSnapshot.frames_received || 0)
                );
                const droppedBusyDelta = Math.max(
                    0,
                    (metrics.frames_dropped_busy || 0) - (lastMetricsSnapshot.frames_dropped_busy || 0)
                );
                const decodeErrDelta = Math.max(
                    0,
                    (metrics.frames_decode_error || 0) - (lastMetricsSnapshot.frames_decode_error || 0)
                );
                const dropRatio = receivedDelta > 0 ? (droppedBusyDelta + decodeErrDelta) / receivedDelta : 0;
                const avgProcessingMs = Number(metrics.avg_processing_ms || 0);

                if (dropRatio > 0.2 || avgProcessingMs > 180) {
                    currentTargetFps = Math.max(adaptiveStreaming.minFps, currentTargetFps - 1);
                    currentJpegQuality = Math.max(adaptiveStreaming.minJpegQuality, currentJpegQuality - 0.05);
                } else if (dropRatio < 0.05 && avgProcessingMs < 110) {
                    currentTargetFps = Math.min(adaptiveStreaming.maxFps, currentTargetFps + 1);
                    currentJpegQuality = Math.min(adaptiveStreaming.maxJpegQuality, currentJpegQuality + 0.03);
                }

                frameIntervalMs = 1000 / currentTargetFps;
                updateStreamStatus(`Server ${avgProcessingMs.toFixed(1)}ms`);
            }
            lastMetricsSnapshot = metrics;
        } catch (_err) {
            // Ignore polling errors silently to avoid log noise.
        }
    }

    async function startCamera() {
        showLoading();
        hideError();
        if (consentCheckbox && !consentCheckbox.checked) {
            showError("Please enable consent before starting camera analysis.");
            return;
        }
        emitConsentState();

        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("Camera API not supported in this browser (try Chrome/Edge).");
            }

            localStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: currentTargetFps },
                    facingMode: "user",
                },
                audio: false,
            });
            videoElement.srcObject = localStream;

            await new Promise((resolve, reject) => {
                const timeoutId = setTimeout(() => {
                    if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                        resolve();
                    } else {
                        reject(new Error("Camera initialization timed out."));
                    }
                }, 15000);

                videoElement.onloadedmetadata = () => {
                    clearTimeout(timeoutId);
                    resolve();
                };
                videoElement.onloadeddata = () => {
                    clearTimeout(timeoutId);
                    resolve();
                };
                videoElement.onerror = () => {
                    clearTimeout(timeoutId);
                    reject(new Error("Video element error."));
                };
            });

            await videoElement.play();

            await new Promise((resolve, reject) => {
                setTimeout(() => {
                    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
                        reject(
                            new Error(
                                "Camera started but no frames are being delivered. Check camera privacy settings."
                            )
                        );
                    } else {
                        resolve();
                    }
                }, 900);
            });

            updateCanvasDimensions();
            startStopButton.textContent = "Stop Camera";
            sendingFrames = true;
            hideLoading();
            sendFrameLoop();
        } catch (error) {
            let msg = "Could not access camera.";
            if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
                msg = "Camera permission denied. Please allow access.";
            } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
                msg = "No camera found.";
            } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
                msg = "Camera is in use by another application.";
            } else if (error.message) {
                msg = `Error: ${error.message}`;
            }
            showError(msg);
            stopCamera(false);
        }
    }

    function stopCamera(updateUI = true) {
        if (localStream) {
            localStream.getTracks().forEach((track) => track.stop());
            videoElement.srcObject = null;
            localStream = null;
        }
        sendingFrames = false;

        if (updateUI) {
            startStopButton.textContent = "Start Camera";
            emotionStatus.textContent = "N/A";
            emotionStatus.style.color = getEmotionColor("unknown");
            updateConfidenceMeter(0, true);
            updateEmotionTheme("neutral");
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            hideLoading();
            hideError();
            setChipStatus(trackerStatus, "Idle", "warn");
        }
    }

    function sendFrameLoop() {
        if (!sendingFrames || !localStream || videoElement.paused || videoElement.ended) {
            return;
        }

        const started = performance.now();
        if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
            setTimeout(sendFrameLoop, 100);
            return;
        }
        if (!socket.connected) {
            setTimeout(sendFrameLoop, 500);
            return;
        }

        if (frameCaptureCanvas.width !== videoElement.videoWidth) {
            updateCanvasDimensions();
        }

        try {
            frameCaptureCtx.drawImage(videoElement, 0, 0, frameCaptureCanvas.width, frameCaptureCanvas.height);
            const dataUrl = frameCaptureCanvas.toDataURL("image/jpeg", currentJpegQuality);
            socket.emit("frame", dataUrl);
        } catch (_err) {
            // Ignore transient frame capture errors.
        }

        const elapsed = performance.now() - started;
        const delay = Math.max(0, frameIntervalMs - elapsed);
        setTimeout(sendFrameLoop, delay);
    }

    function processTherapyResponse(response) {
        let cleaned = String(response || "").replace(/^["'\s]*|["'\s]*$/g, "");
        cleaned = cleaned.replace(/^\\s+|\\s+$/g, "");
        return cleaned;
    }

    function showAiTypingIndicator() {
        if (aiTypingIndicatorElement) {
            aiTypingIndicatorElement.remove();
            aiTypingIndicatorElement = null;
        }

        aiTypingIndicatorElement = document.createElement("div");
        aiTypingIndicatorElement.classList.add("chat-message", "ai-message");
        aiTypingIndicatorElement.id = "ai-typing-indicator";

        const senderSpan = document.createElement("strong");
        senderSpan.textContent = "Eternix";
        aiTypingIndicatorElement.appendChild(senderSpan);

        const typingDots = document.createElement("div");
        typingDots.classList.add("chat-typing");
        typingDots.innerHTML = "<span></span><span></span><span></span>";
        aiTypingIndicatorElement.appendChild(typingDots);

        chatbox.appendChild(aiTypingIndicatorElement);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function removeAiTypingIndicator() {
        if (aiTypingIndicatorElement) {
            aiTypingIndicatorElement.remove();
            aiTypingIndicatorElement = null;
        }
    }

    function simulateTypingEffect(element, message) {
        const contentDiv = document.createElement("div");
        let index = 0;
        const typingSpeed = 20;

        const interval = setInterval(() => {
            if (index < message.length) {
                if (index === 0) {
                    const senderSpan = document.createElement("strong");
                    senderSpan.textContent = "Eternix";
                    contentDiv.appendChild(senderSpan);
                }
                contentDiv.appendChild(document.createTextNode(message.charAt(index)));
                element.appendChild(contentDiv);
                index += 1;
                chatbox.scrollTop = chatbox.scrollHeight;
                return;
            }
            clearInterval(interval);
        }, typingSpeed);
    }

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("chat-message", sender === "user" ? "user-message" : "ai-message");

        if (sender === "user") {
            const content = document.createElement("div");
            const senderSpan = document.createElement("strong");
            senderSpan.textContent = "You";
            content.appendChild(senderSpan);
            content.appendChild(document.createTextNode(message));
            messageElement.appendChild(content);
            chatbox.appendChild(messageElement);
        } else {
            removeAiTypingIndicator();
            chatbox.appendChild(messageElement);
            simulateTypingEffect(messageElement, message);
        }
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function renderFaceOverlay(payload, emotion) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        if (!payload.face_detected || !payload.face_location) {
            return;
        }

        const { x, y, w, h } = payload.face_location;
        const videoW = payload.frame_width || videoElement.videoWidth;
        const videoH = payload.frame_height || videoElement.videoHeight;
        const scaleX = overlayCanvas.width > 0 && videoW > 0 ? overlayCanvas.width / videoW : 1;
        const scaleY = overlayCanvas.height > 0 && videoH > 0 ? overlayCanvas.height / videoH : 1;

        const color = getEmotionColor(emotion);
        overlayCtx.strokeStyle = color;
        overlayCtx.fillStyle = color;
        overlayCtx.lineWidth = 2.5;
        overlayCtx.setLineDash(payload.face_tracked ? [6, 3] : []);
        overlayCtx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
        overlayCtx.setLineDash([]);
        overlayCtx.font = "600 14px Manrope";
        overlayCtx.fillText(
            `${emotion}${payload.face_tracked ? " (tracked)" : ""}`,
            (x * scaleX) + 6,
            Math.max(18, (y * scaleY) - 8)
        );
    }

    function updateEmotionChart(displayEmotion, displayConfidence, smoothedScores) {
        const emotionLabels = ["Happy", "Sad", "Angry", "Surprise", "Neutral", "Fear", "Disgust"];

        if (smoothToggle && smoothToggle.checked && smoothedScores) {
            currentEmotionData = emotionLabels.map((label) => smoothedScores[label.toLowerCase()] || 0);
        } else {
            currentEmotionData = emotionLabels.map((label, idx) =>
                label.toLowerCase() === String(displayEmotion || "").toLowerCase()
                    ? Math.min(100, displayConfidence * 100)
                    : Math.max(0, currentEmotionData[idx] * 0.94)
            );
        }

        emotionChart.data.datasets[0].data = currentEmotionData;
        emotionChart.update();
    }

    socket.on("emotion_update", (data) => {
        let displayEmotion = data.emotion || "unknown";
        let displayConfidence = Number(data.confidence || 0);
        let displayConfidenceRaw = Number(data.confidence_raw || displayConfidence);
        let chartDataScores = null;

        if (smoothToggle && smoothToggle.checked && data.smoothed) {
            displayEmotion = data.smoothed.emotion || displayEmotion;
            displayConfidence = Number(data.smoothed.confidence || displayConfidence);
            displayConfidenceRaw = Number(data.smoothed.confidence_raw || displayConfidenceRaw);
            chartDataScores = data.smoothed.scores || null;
        }

        const lowConfThreshold = Number(
            (data.quality && data.quality.low_confidence_threshold) || LOW_CONFIDENCE_FALLBACK_THRESHOLD
        );
        const isLowConfidence = Boolean(data.quality && data.quality.low_confidence) ||
            (displayConfidence < lowConfThreshold);

        emotionStatus.textContent = `${displayEmotion} (${(displayConfidence * 100).toFixed(1)}%)`;
        emotionStatus.style.color = getEmotionColor(displayEmotion);
        updateEmotionTheme(displayEmotion);
        updateConfidenceMeter(displayConfidence, isLowConfidence);

        if (!data.face_detected) {
            setChipStatus(trackerStatus, "No Face", "warn");
            confidenceNotice.textContent = "No confident face detected. Move closer and improve lighting.";
        } else if (data.face_tracked) {
            setChipStatus(trackerStatus, "Tracking Hold", "warn");
            confidenceNotice.textContent = isLowConfidence
                ? `Low confidence (${(displayConfidenceRaw * 100).toFixed(1)}% raw). Treat as approximate.`
                : "";
        } else {
            setChipStatus(trackerStatus, "Live Face", "ok");
            confidenceNotice.textContent = isLowConfidence
                ? `Low confidence (${(displayConfidenceRaw * 100).toFixed(1)}% raw). Treat as approximate.`
                : "";
        }

        renderFaceOverlay(data, displayEmotion);
        updateEmotionChart(displayEmotion, displayConfidence, chartDataScores);
    });

    socket.on("ai_typing", () => {
        showAiTypingIndicator();
    });

    socket.on("ai_response", (data) => {
        if (data.error) {
            appendMessage("ai", `I hit an issue: ${data.error}`);
            return;
        }
        if (data.response) {
            const processed = processTherapyResponse(data.response);
            if (processed && processed.trim()) {
                appendMessage("ai", processed);
            } else {
                appendMessage("ai", "I received an empty response after processing. Please try again.");
            }
            return;
        }
        appendMessage("ai", "I received an unexpected response. Please try rephrasing.");
    });

    socket.on("emotion_memory_cleared", () => {
        emotionStatus.textContent = "N/A";
        emotionStatus.style.color = getEmotionColor("unknown");
        updateConfidenceMeter(0, true);
        confidenceNotice.textContent = "Emotion memory cleared for this session.";
        setChipStatus(trackerStatus, "Idle", "warn");
    });

    socket.on("camera_consent_required", () => {
        showError("Camera consent is required. Enable consent and start camera again.");
        stopCamera(false);
    });

    socket.on("connect", () => {
        setChipStatus(socketState, "Connected", "ok");
        emitConsentState();
        pollBackendHealth();
    });

    socket.on("disconnect", () => {
        setChipStatus(socketState, "Disconnected", "error");
        stopCamera();
    });

    socket.on("connect_error", () => {
        setChipStatus(socketState, "Connection Error", "error");
    });

    startStopButton.addEventListener("click", () => {
        if (sendingFrames) {
            stopCamera();
        } else {
            startCamera();
        }
    });

    retryButton.addEventListener("click", () => {
        startCamera();
    });

    if (clearEmotionMemoryButton) {
        clearEmotionMemoryButton.addEventListener("click", () => {
            socket.emit("clear_emotion_state");
        });
    }

    sendMessageButton.addEventListener("click", () => {
        const message = messageInput.value.trim();
        if (!message) return;
        appendMessage("user", message);
        socket.emit("user_message", {
            message,
            raw_text: message,
            context: {
                current_emotion: emotionStatus.textContent,
                timestamp: new Date().toISOString(),
            },
        });
        messageInput.value = "";
    });

    messageInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendMessageButton.click();
        }
    });

    if (privacyNotice && consentCheckbox) {
        privacyNotice.textContent = consentCheckbox.checked
            ? "Camera consent enabled for this session."
            : "Enable consent before camera analysis.";
        consentCheckbox.addEventListener("change", () => {
            privacyNotice.textContent = consentCheckbox.checked
                ? "Camera consent enabled for this session."
                : "Enable consent before camera analysis.";
            emitConsentState();
        });
    }

    if (messageInput) {
        messageInput.disabled = false;
        messageInput.readOnly = false;
        messageInput.removeAttribute("disabled");
        messageInput.removeAttribute("readonly");
    }
    if (chatbox && messageInput) {
        chatbox.addEventListener("click", () => {
            messageInput.focus();
        });
    }

    updateEmotionTheme("neutral");
    updateConfidenceMeter(0, true);
    setChipStatus(socketState, "Connecting", "warn");
    setChipStatus(backendStatus, "Checking...", "warn");
    setChipStatus(readinessStatus, "Checking...", "warn");
    setChipStatus(trackerStatus, "Idle", "warn");
    updateStreamStatus();

    pollBackendHealth();
    setInterval(pollBackendHealth, 9000);
    setInterval(fetchAndAdaptStreamMetrics, 4000);

    stopCamera();
});
