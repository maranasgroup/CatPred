(function () {
  const form = document.getElementById("predictForm");
  const rowContainer = document.getElementById("rowContainer");
  const outputCard = document.querySelector(".output-card");

  const addRowBtn = document.getElementById("addRowBtn");
  const loadSampleBtn = document.getElementById("loadSampleBtn");
  const runBtn = document.getElementById("runBtn");

  const statusBox = document.getElementById("statusBox");
  const resultCards = document.getElementById("resultCards");
  const previewTable = document.getElementById("previewTable");

  const serviceBadge = document.getElementById("serviceBadge");
  const serviceHint = document.getElementById("serviceHint");

  const presetButtons = document.querySelectorAll(".preset-btn");
  const predictionTimeoutMs = 120000;

  const sampleRows = [
    {
      SMILES: "CCO",
      sequence: "ACDEFGHIK",
      pdbpath: "seq_001",
    },
    {
      SMILES: "CCN",
      sequence: "LMNPQRSTV",
      pdbpath: "seq_002",
    },
  ];

  const supportedParameters = ["kcat", "km", "ki"];
  let availableCheckpointParams = new Set(supportedParameters);
  let localCheckpointParams = new Set(supportedParameters);
  let selectedParameter = "kcat";
  const runtimeState = {
    defaultBackend: "local",
    modalReady: false,
    localReady: false,
    fallbackToLocalEnabled: false,
  };
  const runningPhaseMessages = [
    "validating input",
    "building protein records",
    "running model ensemble",
    "aggregating uncertainty",
  ];
  const defaultRunButtonLabel = runBtn ? runBtn.textContent : "Run prediction";
  let runningStatusInterval = null;
  let runStartedAtMs = null;

  function jsonPretty(data) {
    return JSON.stringify(data, null, 2);
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatNumber(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return "—";
    }
    return n.toFixed(1);
  }

  function truncateText(value, maxLength) {
    const text = String(value || "");
    if (text.length <= maxLength) {
      return text;
    }
    return text.slice(0, maxLength - 1) + "…";
  }

  function setStatus(text, kind) {
    if (!statusBox) return;
    statusBox.textContent = text;
    statusBox.classList.remove("ok", "error", "running");
    if (kind) {
      statusBox.classList.add(kind);
    }
  }

  function formatElapsed(seconds) {
    const safeSeconds = Math.max(0, Math.floor(seconds));
    const minutes = Math.floor(safeSeconds / 60);
    const remainder = safeSeconds % 60;
    return String(minutes).padStart(2, "0") + ":" + String(remainder).padStart(2, "0");
  }

  function getElapsedSeconds() {
    if (!runStartedAtMs) return 0;
    return Math.floor((Date.now() - runStartedAtMs) / 1000);
  }

  function setRunButtonState(isRunning, elapsedLabel) {
    if (!runBtn) return;
    if (isRunning) {
      const suffix = elapsedLabel ? " " + elapsedLabel : "";
      runBtn.textContent = "Running" + suffix;
      runBtn.classList.add("is-running");
      runBtn.disabled = true;
      runBtn.setAttribute("aria-busy", "true");
      return;
    }

    runBtn.textContent = defaultRunButtonLabel || "Run prediction";
    runBtn.classList.remove("is-running");
    runBtn.disabled = false;
    runBtn.removeAttribute("aria-busy");
  }

  function setBusyUi(isBusy) {
    if (form) {
      form.classList.toggle("is-busy", isBusy);
      form.setAttribute("aria-busy", isBusy ? "true" : "false");
    }
    if (outputCard) {
      outputCard.classList.toggle("is-busy", isBusy);
      outputCard.setAttribute("aria-busy", isBusy ? "true" : "false");
    }

    if (addRowBtn) {
      addRowBtn.disabled = isBusy;
    }
    if (loadSampleBtn) {
      loadSampleBtn.disabled = isBusy;
    }
    if (presetButtons && presetButtons.length) {
      presetButtons.forEach((btn) => {
        if (!isBusy && !isParameterAvailable(btn.dataset.param)) {
          btn.disabled = true;
          return;
        }
        btn.disabled = isBusy;
      });
    }
  }

  function startRunningFeedback(payload) {
    stopRunningFeedback();
    runStartedAtMs = Date.now();
    setBusyUi(true);

    const parameterLabel = String(payload.parameter || "kcat").toUpperCase();
    const rowCount = Array.isArray(payload.input_rows) ? payload.input_rows.length : 0;
    const rowLabel = rowCount === 1 ? "1 row" : String(rowCount) + " rows";

    const updateRunningStatus = () => {
      const elapsed = getElapsedSeconds();
      const elapsedLabel = formatElapsed(elapsed);
      const phaseIndex = Math.floor(elapsed / 3) % runningPhaseMessages.length;
      const phaseText = runningPhaseMessages[phaseIndex];
      setRunButtonState(true, elapsedLabel);
      setStatus(
        "Running " + parameterLabel + " on " + rowLabel + " • " + elapsedLabel + " • " + phaseText,
        "running"
      );
    };

    updateRunningStatus();
    runningStatusInterval = window.setInterval(updateRunningStatus, 1000);
  }

  function stopRunningFeedback() {
    if (runningStatusInterval) {
      window.clearInterval(runningStatusInterval);
      runningStatusInterval = null;
    }
    setBusyUi(false);
    setRunButtonState(false);
  }

  function setServiceState(text, hint, kind) {
    if (serviceBadge) {
      serviceBadge.textContent = text;
      serviceBadge.classList.remove("ok", "error");
      if (kind) {
        serviceBadge.classList.add(kind);
      }
    }
    if (serviceHint) {
      serviceHint.textContent = hint || "";
    }
  }

  function setActivePreset(paramValue) {
    if (!presetButtons || !presetButtons.length) return;
    presetButtons.forEach((btn) => {
      if (!(btn instanceof HTMLElement)) return;
      btn.classList.toggle("active", btn.dataset.param === paramValue);
    });
  }

  function getSelectedParameter() {
    return String(selectedParameter || "kcat").toLowerCase();
  }

  function firstAvailableParameter() {
    for (const param of supportedParameters) {
      if (availableCheckpointParams.has(param)) {
        return param;
      }
    }
    return null;
  }

  function parseAvailableCheckpointParams(availableCheckpoints) {
    if (availableCheckpoints && typeof availableCheckpoints === "object") {
      return new Set(
        Object.keys(availableCheckpoints)
          .map((key) => String(key).toLowerCase())
          .filter((key) => supportedParameters.includes(key))
      );
    }
    return new Set(supportedParameters);
  }

  function setParameterAvailability(availableCheckpoints) {
    availableCheckpointParams = parseAvailableCheckpointParams(availableCheckpoints);

    if (presetButtons && presetButtons.length) {
      presetButtons.forEach((btn) => {
        const paramValue = String(btn.dataset.param || "").toLowerCase();
        const enabled = availableCheckpointParams.has(paramValue);
        btn.disabled = !enabled;
        btn.setAttribute("aria-disabled", String(!enabled));
      });
    }

    const fallbackParam = firstAvailableParameter();
    if (fallbackParam && !availableCheckpointParams.has(getSelectedParameter())) {
      selectedParameter = fallbackParam;
      setActivePreset(fallbackParam);
    }
  }

  function isParameterAvailable(paramValue) {
    return availableCheckpointParams.has(String(paramValue || "").toLowerCase());
  }

  function chooseRequestBackend() {
    if (runtimeState.defaultBackend === "modal" && runtimeState.modalReady) {
      return "modal";
    }
    if (runtimeState.defaultBackend === "local" && localCheckpointParams.size > 0) {
      return "local";
    }
    if (runtimeState.modalReady) {
      return "modal";
    }
    if (runtimeState.localReady) {
      return "local";
    }
    return runtimeState.defaultBackend || "local";
  }

  function shouldFallbackToLocal(requestBackend, targetParam) {
    if (requestBackend !== "modal") {
      return false;
    }
    if (!runtimeState.fallbackToLocalEnabled) {
      return false;
    }
    if (!runtimeState.localReady) {
      return false;
    }
    return localCheckpointParams.has(String(targetParam || "").toLowerCase());
  }

  function formatSequenceId(index) {
    const safeIndex = Math.max(1, Number(index) || 1);
    return "seq_" + String(safeIndex).padStart(3, "0");
  }

  function getNextSequenceId() {
    if (!rowContainer) {
      return formatSequenceId(1);
    }

    let maxSeen = 0;
    const idInputs = rowContainer.querySelectorAll('input[name="pdbpath"]');
    idInputs.forEach((input) => {
      const raw = String(input.value || "").trim();
      const match = raw.match(/^seq_(\d+)$/i);
      if (!match) return;
      const parsed = Number(match[1]);
      if (Number.isFinite(parsed)) {
        maxSeen = Math.max(maxSeen, parsed);
      }
    });

    if (maxSeen > 0) {
      return formatSequenceId(maxSeen + 1);
    }

    const currentRowCount = rowContainer.querySelectorAll(".row-item").length;
    return formatSequenceId(currentRowCount + 1);
  }

  function rowTemplate(index, values) {
    const smiles = values && values.SMILES ? values.SMILES : "";
    const seq = values && values.sequence ? values.sequence : "";
    const pdb = values && values.pdbpath ? values.pdbpath : "";

    return (
      '<article class="row-item">' +
      '<div class="row-item-head">' +
      "<h4>Entry " +
      (index + 1) +
      "</h4>" +
      '<button type="button" class="icon-btn" data-remove-row="1">Remove</button>' +
      "</div>" +
      '<div class="row-grid">' +
      '<label><span class="field-label">Sequence ID</span><input name="pdbpath" placeholder="seq_001" value="' +
      escapeHtml(pdb) +
      '" required /></label>' +
      '<label><span class="field-label">Substrate (SMILES)</span><input name="SMILES" placeholder="CCO" value="' +
      escapeHtml(smiles) +
      '" required /></label>' +
      '<label><span class="field-label">Enzyme sequence</span><textarea name="sequence" rows="3" placeholder="ACDEFGHIK" required>' +
      escapeHtml(seq) +
      "</textarea></label>" +
      "</div>" +
      "</article>"
    );
  }

  function renumberRows() {
    if (!rowContainer) return;
    const items = rowContainer.querySelectorAll(".row-item");
    const shouldShowHeader = items.length > 1;
    items.forEach((item, idx) => {
      const head = item.querySelector(".row-item-head");
      if (head instanceof HTMLElement) {
        head.hidden = !shouldShowHeader;
      }
      const heading = item.querySelector("h4");
      if (heading) {
        heading.textContent = "Entry " + String(idx + 1);
      }
    });
  }

  function addRow(values) {
    if (!rowContainer) return;
    const nextValues = values ? { ...values } : { SMILES: "", sequence: "", pdbpath: "" };
    if (!String(nextValues.pdbpath || "").trim()) {
      nextValues.pdbpath = getNextSequenceId();
    }
    const index = rowContainer.querySelectorAll(".row-item").length;
    rowContainer.insertAdjacentHTML("beforeend", rowTemplate(index, nextValues));
    const last = rowContainer.lastElementChild;
    if (last) {
      last.animate(
        [
          { opacity: 0, transform: "translateY(8px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        { duration: 220, easing: "ease-out" }
      );
    }
    renumberRows();
  }

  function clearRows() {
    if (!rowContainer) return;
    rowContainer.innerHTML = "";
  }

  function loadSampleRows() {
    clearRows();
    sampleRows.forEach((row) => addRow(row));
  }

  function collectRows() {
    if (!rowContainer) return [];

    const rows = [];
    const items = rowContainer.querySelectorAll(".row-item");

    items.forEach((item) => {
      const smilesInput = item.querySelector('input[name="SMILES"]');
      const sequenceInput = item.querySelector('textarea[name="sequence"]');
      const pdbpathInput = item.querySelector('input[name="pdbpath"]');

      if (!smilesInput || !sequenceInput || !pdbpathInput) {
        return;
      }

      const smiles = smilesInput.value.trim();
      const sequence = sequenceInput.value.trim().toUpperCase();
      const pdbpath = pdbpathInput.value.trim();

      if (!smiles || !sequence || !pdbpath) {
        return;
      }

      rows.push({
        SMILES: smiles,
        sequence: sequence,
        pdbpath: pdbpath,
      });
    });

    return rows;
  }

  function validateRows(rows) {
    if (!rows.length) {
      return "Please add at least one complete input row.";
    }

    const mapping = new Map();
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const key = row.pdbpath;
      if (mapping.has(key) && mapping.get(key) !== row.sequence) {
        return "Each Sequence ID must map to one unique enzyme sequence.";
      }
      mapping.set(key, row.sequence);
    }

    return "";
  }

  function buildPayload(rows) {
    const target = getSelectedParameter();
    const requestBackend = chooseRequestBackend();

    return {
      parameter: target,
      checkpoint_dir: target,
      input_rows: rows,
      use_gpu: false,
      results_dir: "web-app",
      backend: requestBackend,
      fallback_to_local: shouldFallbackToLocal(requestBackend, target),
    };
  }

  function parsePrediction(row) {
    const keys = Object.keys(row || {});
    const linearKey = keys.find((key) => key.startsWith("Prediction_("));
    const unitMatch = linearKey ? linearKey.match(/^Prediction_\((.*)\)$/) : null;
    const unit = unitMatch ? unitMatch[1] : "";

    return {
      linear: linearKey ? row[linearKey] : null,
      linearKey: linearKey || "Prediction",
      unit: unit,
      log10: row.Prediction_log10,
      sdTotal: row.SD_total,
      sdAleatoric: row.SD_aleatoric,
      sdEpistemic: row.SD_epistemic,
    };
  }

  function renderResultCards(previewRows, selectedParam) {
    if (!resultCards) return;

    if (!previewRows || !previewRows.length) {
      resultCards.innerHTML =
        '<article class="empty-result"><p>No preview rows were returned for this run.</p></article>';
      return;
    }

    const cardsHtml = previewRows
      .map((row, idx) => {
        const p = parsePrediction(row);

        return (
          '<article class="result-card">' +
          '<div class="result-card-head">' +
          "<h4>Result " +
          String(idx + 1) +
          "</h4>" +
          '<span class="result-chip">' +
          escapeHtml(selectedParam.toUpperCase()) +
          "</span>" +
          "</div>" +
          '<div class="result-main">' +
          "<strong>" +
          escapeHtml(formatNumber(p.linear)) +
          "</strong>" +
          "<span>" +
          escapeHtml(p.unit || "predicted unit") +
          "</span>" +
          "</div>" +
          '<dl class="metric-grid">' +
          "<div><dt>log10</dt><dd>" +
          escapeHtml(formatNumber(p.log10)) +
          "</dd></div>" +
          "<div><dt>Total SD</dt><dd>" +
          escapeHtml(formatNumber(p.sdTotal)) +
          "</dd></div>" +
          "<div><dt>Epistemic SD</dt><dd>" +
          escapeHtml(formatNumber(p.sdEpistemic)) +
          "</dd></div>" +
          "</dl>" +
          '<div class="result-meta">' +
          "<span>SMILES: " +
          escapeHtml(truncateText(row.SMILES, 24)) +
          "</span>" +
          "<span>Sequence ID: " +
          escapeHtml(row.pdbpath || "—") +
          "</span>" +
          "</div>" +
          "</article>"
        );
      })
      .join("");

    resultCards.innerHTML = cardsHtml;
  }

  function renderPreviewTable(previewRows) {
    if (!previewTable) return;

    if (!previewRows || !previewRows.length) {
      previewTable.innerHTML = "<tbody><tr><td>No rows to show.</td></tr></tbody>";
      return;
    }

    const keys = Object.keys(previewRows[0]);
    const head =
      "<thead><tr>" +
      keys.map((k) => "<th>" + escapeHtml(k) + "</th>").join("") +
      "</tr></thead>";

    const bodyRows = previewRows
      .map((row) => {
        const cells = keys
          .map((k) => {
            const value = row[k];
            if (value === null || value === undefined) {
              return "<td></td>";
            }
            const displayValue = typeof value === "number" ? formatNumber(value) : value;
            return "<td>" + escapeHtml(displayValue) + "</td>";
          })
          .join("");
        return "<tr>" + cells + "</tr>";
      })
      .join("");

    previewTable.innerHTML = head + "<tbody>" + bodyRows + "</tbody>";
  }

  async function fetchReady() {
    setServiceState("Checking...", "", "");

    try {
      const response = await fetch("/ready", {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      let data;
      try {
        data = await response.json();
      } catch (_err) {
        data = {};
      }

      if (!response.ok) {
        setServiceState("Offline", "Service not reachable", "error");
        return;
      }

      const backends = data && data.backends ? data.backends : {};
      runtimeState.defaultBackend = data && data.default_backend ? String(data.default_backend) : "local";
      runtimeState.modalReady = Boolean(backends.modal && backends.modal.ready);
      runtimeState.localReady = Boolean(backends.local && backends.local.ready);
      runtimeState.fallbackToLocalEnabled = Boolean(data && data.fallback_to_local_enabled);

      localCheckpointParams = parseAvailableCheckpointParams(
        data && data.api ? data.api.available_checkpoints : null
      );

      if (localCheckpointParams.size > 0) {
        setParameterAvailability(Object.fromEntries(Array.from(localCheckpointParams).map((key) => [key, key])));
      } else if (runtimeState.modalReady) {
        setParameterAvailability({ kcat: "kcat", km: "km", ki: "ki" });
      } else {
        setParameterAvailability({});
      }

      const availableParams = Array.from(availableCheckpointParams.values());
      const localParams = Array.from(localCheckpointParams.values());

      if (data && data.ready) {
        const backend = data.default_backend ? String(data.default_backend) : "default";
        let hint = "";
        if (localParams.length) {
          hint = "Backend: " + backend + " | Checkpoints: " + localParams.join(", ");
        } else if (runtimeState.modalReady) {
          hint = "Backend: " + backend + " | Remote checkpoints: " + availableParams.join(", ");
        } else {
          hint = "Backend: " + backend + " | No local checkpoints found";
        }
        setServiceState("Online", hint, "ok");
      } else {
        const hint = runtimeState.modalReady
          ? "Backend available in limited mode"
          : "Backend configuration needed | No local checkpoints found";
        setServiceState("Limited", hint, "error");
      }
    } catch (_err) {
      setServiceState("Offline", "Could not contact service", "error");
    }
  }

  async function submitPrediction(event) {
    event.preventDefault();

    const rows = collectRows();
    const rowError = validateRows(rows);
    if (rowError) {
      setStatus(rowError, "error");
      return;
    }

    const selectedParameter = getSelectedParameter();
    if (!isParameterAvailable(selectedParameter)) {
      setStatus("No local checkpoint available for " + selectedParameter.toUpperCase() + ".", "error");
      return;
    }

    const payload = buildPayload(rows);
    startRunningFeedback(payload);
    await new Promise((resolve) => window.requestAnimationFrame(resolve));

    const requestController = new AbortController();
    const requestTimeout = window.setTimeout(() => {
      requestController.abort();
    }, predictionTimeoutMs);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        signal: requestController.signal,
        body: JSON.stringify(payload),
      });

      let data;
      try {
        data = await response.json();
      } catch (_err) {
        data = { detail: "Unexpected response format." };
      }

      if (!response.ok) {
        const message = data && data.detail ? String(data.detail) : "Prediction could not be completed.";
        renderResultCards([], payload.parameter);
        renderPreviewTable([]);
        const elapsedLabel = formatElapsed(getElapsedSeconds());
        setStatus(message + " (" + elapsedLabel + ")", "error");
        return;
      }

      renderResultCards(data.preview_rows || [], payload.parameter);
      renderPreviewTable(data.preview_rows || []);
      const elapsedLabel = formatElapsed(getElapsedSeconds());
      setStatus("Prediction complete • " + elapsedLabel, "ok");
    } catch (err) {
      renderResultCards([], payload.parameter);
      renderPreviewTable([]);
      const elapsedLabel = formatElapsed(getElapsedSeconds());
      if (err && err.name === "AbortError") {
        const timeoutLabel = formatElapsed(Math.floor(predictionTimeoutMs / 1000));
        setStatus(
          "Prediction timed out after " +
            timeoutLabel +
            ". This is often a cold-start delay; retry once or check backend logs. (" +
            elapsedLabel +
            ")",
          "error"
        );
      } else {
        setStatus("Network error while running prediction. (" + elapsedLabel + ")", "error");
      }
    } finally {
      window.clearTimeout(requestTimeout);
      stopRunningFeedback();
      runStartedAtMs = null;
    }
  }

  function setupFaq() {
    const faqRoot = document.getElementById("faqList");
    if (!faqRoot) return;

    const items = faqRoot.querySelectorAll(".faq-item");
    items.forEach((item) => {
      const button = item.querySelector("button");
      if (!button) return;

      button.addEventListener("click", function () {
        const isOpen = item.classList.contains("open");
        items.forEach((row) => {
          row.classList.remove("open");
          const rowButton = row.querySelector("button");
          if (rowButton) {
            rowButton.setAttribute("aria-expanded", "false");
          }
        });
        if (!isOpen) {
          item.classList.add("open");
          button.setAttribute("aria-expanded", "true");
        }
      });
    });
  }

  function setupReveal() {
    const revealItems = document.querySelectorAll(".reveal");
    if (!revealItems.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("show");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.12,
        rootMargin: "0px 0px -36px 0px",
      }
    );

    revealItems.forEach((item) => observer.observe(item));
  }

  function setupEvents() {
    if (presetButtons && presetButtons.length) {
      presetButtons.forEach((btn) => {
        btn.addEventListener("click", function () {
          if (btn.disabled) return;
          const chosenParam = btn.dataset.param || "kcat";
          selectedParameter = String(chosenParam).toLowerCase();
          setActivePreset(chosenParam);
        });
      });
    }

    if (addRowBtn) {
      addRowBtn.addEventListener("click", function () {
        addRow({ SMILES: "", sequence: "", pdbpath: "" });
      });
    }

    if (loadSampleBtn) {
      loadSampleBtn.addEventListener("click", function () {
        loadSampleRows();
        setStatus("Sample inputs loaded", "ok");
      });
    }

    if (rowContainer) {
      rowContainer.addEventListener("click", function (event) {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        if (!target.matches("[data-remove-row]")) return;

        const row = target.closest(".row-item");
        if (!row) return;

        if (rowContainer.querySelectorAll(".row-item").length === 1) {
          setStatus("At least one row is required", "error");
          return;
        }

        row.remove();
        renumberRows();
      });
    }

    if (form) {
      form.addEventListener("submit", submitPrediction);
    }
  }

  function bootstrap() {
    clearRows();
    addRow(sampleRows[0]);
    setActivePreset("kcat");

    renderResultCards([], "kcat");
    renderPreviewTable([]);

    setStatus("Ready when you are.", "ok");
    setupEvents();
    setupFaq();
    setupReveal();
    fetchReady();
  }

  bootstrap();
})();
