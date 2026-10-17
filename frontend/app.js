// LogoErase Pro - Client Application
document.addEventListener("DOMContentLoaded", () => {
  // Initialize Lucide icons
  if (window.lucide) lucide.createIcons();

  // Application State
  const state = {
    batchId: null,
    batchData: null,
    pollInterval: null,
    currentFilter: "all",
    activeModalImgId: null,
    sliderPos: 50,
    isDraggingSlider: false,
    fetchingStatus: false,
    cardsSignature: ""
  };

  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const folderInput = document.getElementById("folderInput");
  const browseFilesBtn = document.getElementById("browseFilesBtn");
  const browseFolderBtn = document.getElementById("browseFolderBtn");

  const batchControlsPanel = document.getElementById("batchControlsPanel");
  const batchCountBadge = document.getElementById("batchCountBadge");
  const algoModeSelect = document.getElementById("algoModeSelect");
  const parallelModeSelect = document.getElementById("parallelModeSelect");
  const startBatchBtn = document.getElementById("startBatchBtn");
  const downloadZipBtn = document.getElementById("downloadZipBtn");
  const clearAllBtn = document.getElementById("clearAllBtn");

  const progressBar = document.getElementById("progressBar");
  const progressPercent = document.getElementById("progressPercent");
  const progressStatusText = document.getElementById("progressStatusText");
  const currentFilenameText = document.getElementById("currentFilenameText");
  const speedStat = document.getElementById("speedStat");
  const timeStat = document.getElementById("timeStat");

  const gallerySection = document.getElementById("gallerySection");
  const imageGrid = document.getElementById("imageGrid");
  const filterTabs = document.querySelectorAll(".filter-tab");
  const countAll = document.getElementById("countAll");
  const countCompleted = document.getElementById("countCompleted");
  const countFlagged = document.getElementById("countFlagged");

  // Modal Elements
  const compareModal = document.getElementById("compareModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const modalPrevBtn = document.getElementById("modalPrevBtn");
  const modalNextBtn = document.getElementById("modalNextBtn");
  const modalDownloadBtn = document.getElementById("modalDownloadBtn");
  const modalFilename = document.getElementById("modalFilename");
  const modalMeta = document.getElementById("modalMeta");
  const comparisonContainer = document.getElementById("comparisonContainer");
  const originalImg = document.getElementById("originalImg");
  const processedImg = document.getElementById("processedImg");
  const comparisonOverlay = document.getElementById("comparisonOverlay");
  const sliderHandle = document.getElementById("sliderHandle");
  const bboxMarker = document.getElementById("bboxMarker");
  const showBboxToggle = document.getElementById("showBboxToggle");
  const detectedCoords = document.getElementById("detectedCoords");
  const recleanModeSelect = document.getElementById("recleanModeSelect");
  const recleanBtn = document.getElementById("recleanBtn");

  // Helper: Format bytes
  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + " " + sizes[i];
  }

  function getParallelWorkers() {
    const parsed = Number.parseInt(parallelModeSelect.value, 10);
    if (!Number.isFinite(parsed)) return 4;
    return Math.max(1, Math.min(10, parsed));
  }

  // --- 1. UPLOAD & DRAG/DROP ---
  browseFilesBtn.addEventListener("click", () => fileInput.click());
  browseFolderBtn.addEventListener("click", () => folderInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadFiles(Array.from(e.target.files));
    }
  });

  folderInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadFiles(Array.from(e.target.files));
    }
  });

  // Drag and drop events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dropzone-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dropzone-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      uploadFiles(Array.from(dt.files));
    }
  });

  async function uploadFiles(files) {
    if (!files || files.length === 0) return;
    if (state.batchData?.status === "processing") {
      alert("Please wait for the current batch to finish before uploading another.");
      return;
    }
    const formData = new FormData();
    const paths = [];

    files.forEach((f) => {
      formData.append("files", f);
      paths.push(f.webkitRelativePath || f.name);
    });
    formData.append("paths", paths.join(","));

    try {
      progressStatusText.textContent = `Uploading ${files.length} images...`;
      batchControlsPanel.classList.remove("hidden");

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      state.batchId = data.batch_id;
      await fetchBatchStatus();
    } catch (err) {
      alert("Upload error: " + err.message);
    }
  }

  // --- 2. BATCH STATUS & POLLING ---
  async function fetchBatchStatus() {
    if (!state.batchId || state.fetchingStatus) return;
    const batchId = state.batchId;
    state.fetchingStatus = true;
    try {
      const res = await fetch(`/api/batch/${batchId}/status`);
      if (batchId !== state.batchId) return;
      if (res.status === 404) {
        clearInterval(state.pollInterval);
        state.pollInterval = null;
        state.batchId = null;
        throw new Error("This batch has expired or the server restarted. Please upload your images again.");
      }
      if (!res.ok) throw new Error("Unable to read progress. Retrying...");
      const data = await res.json();
      if (batchId !== state.batchId) return;
      state.batchData = data;
      renderUI();

      if (data.status === "processing") {
        if (!state.pollInterval) {
          state.pollInterval = setInterval(fetchBatchStatus, 400);
        }
      } else {
        if (state.pollInterval) {
          clearInterval(state.pollInterval);
          state.pollInterval = null;
        }
      }
    } catch (err) {
      console.error("Poll status error:", err);
      progressStatusText.textContent = err.message === "Failed to fetch"
        ? "Server disconnected. Reopen LogoErase Preview to reconnect."
        : err.message;
    } finally {
      state.fetchingStatus = false;
    }
  }

  // Start batch processing
  startBatchBtn.addEventListener("click", async () => {
    if (!state.batchId) return;
    try {
      startBatchBtn.disabled = true;
      startBatchBtn.classList.add("opacity-50");

      const mode = algoModeSelect.value;
      const parallelWorkers = getParallelWorkers();
      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          batch_id: state.batchId,
          mode: mode,
          sensitivity: 1.0,
          feather: 2,
          parallel_workers: parallelWorkers
        })
      });
      if (!res.ok) throw new Error("Failed to start processing");

      if (!state.pollInterval) {
        state.pollInterval = setInterval(fetchBatchStatus, 400);
      }
    } catch (err) {
      alert("Processing error: " + err.message);
      startBatchBtn.disabled = false;
      startBatchBtn.classList.remove("opacity-50");
    }
  });

  // Download ZIP
  downloadZipBtn.addEventListener("click", () => {
    if (!state.batchId) return;
    window.location.href = `/api/download-zip/${state.batchId}`;
  });

  // Clear All
  clearAllBtn.addEventListener("click", () => {
    if (confirm("Clear current batch?")) {
      if (state.pollInterval) clearInterval(state.pollInterval);
      state.pollInterval = null;
      state.batchId = null;
      state.batchData = null;
      batchControlsPanel.classList.add("hidden");
      gallerySection.classList.add("hidden");
      imageGrid.innerHTML = "";
      state.cardsSignature = "";
      fileInput.value = "";
      folderInput.value = "";
      closeCompareModal();
    }
  });

  // --- 3. UI RENDERING ---
  function renderUI() {
    const data = state.batchData;
    if (!data) return;

    batchControlsPanel.classList.remove("hidden");
    gallerySection.classList.remove("hidden");

    // Top counters
    batchCountBadge.textContent = `${data.total_count} Images`;
    progressBar.style.width = `${data.progress || 0}%`;
    progressPercent.textContent = `${Math.round(data.progress || 0)}%`;
    speedStat.textContent = `Speed: ${data.speed || 0} img/s`;
    timeStat.textContent = `Elapsed: ${data.elapsed_sec || 0}s`;

    if (data.status === "processing") {
      progressStatusText.textContent = `Processing ${data.current_index || 0} of ${data.total_count} with ${data.parallel_workers || 1} parallel worker${(data.parallel_workers || 1) === 1 ? "" : "s"}...`;
      currentFilenameText.textContent = data.current_filename || "";
      startBatchBtn.disabled = true;
      startBatchBtn.classList.add("opacity-50");
      downloadZipBtn.disabled = true;
    } else if (data.status === "completed") {
      progressStatusText.textContent = `Finished: ${data.completed_count} completed, ${data.error_count || 0} failed.`;
      currentFilenameText.textContent = "";
      startBatchBtn.disabled = false;
      startBatchBtn.classList.remove("opacity-50");
      downloadZipBtn.disabled = false;
    } else {
      progressStatusText.textContent = `Ready: ${data.total_count} images queued.`;
      startBatchBtn.disabled = false;
      startBatchBtn.classList.remove("opacity-50");
      downloadZipBtn.disabled = (data.completed_count === 0);
    }

    // Filter counts
    const imagesList = Object.values(data.images);
    countAll.textContent = imagesList.length;
    countCompleted.textContent = imagesList.filter((im) => im.status === "completed").length;
    countFlagged.textContent = imagesList.filter((im) => im.flagged).length;

    renderCards();
  }

  function renderCards() {
    const data = state.batchData;
    if (!data) return;

    const signature = JSON.stringify([data.batch_id, state.currentFilter,
      Object.values(data.images).map(im => [im.id, im.status, im.flagged, im.confidence])]);
    if (signature === state.cardsSignature) return;
    state.cardsSignature = signature;

    const imagesList = Object.values(data.images);
    const filtered = imagesList.filter((im) => {
      if (state.currentFilter === "completed") return im.status === "completed";
      if (state.currentFilter === "flagged") return im.flagged;
      return true;
    });

    imageGrid.innerHTML = "";

    filtered.forEach((im) => {
      const card = document.createElement("div");
      card.className = "glass-panel rounded-xl overflow-hidden border border-slate-800 hover:border-indigo-500/50 transition-all duration-200 group flex flex-col cursor-pointer";
      card.dataset.id = im.id;

      // Status pill
      let statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-400">Pending</span>`;
      if (im.status === "processing") {
        statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 animate-pulse">Processing...</span>`;
      } else if (im.status === "completed") {
        if (im.flagged) {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">Review Flag</span>`;
        } else {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Cleaned</span>`;
        }
      } else if (im.status === "error") {
        statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-500/20 text-red-300">Failed</span>`;
      }

      // Confidence badge
      const confVal = im.confidence !== null ? Math.round(im.confidence * 100) : null;
      const confHtml = confVal !== null
        ? `<div class="text-[10px] text-slate-400 flex items-center space-x-1"><span>Conf:</span><span class="font-bold ${confVal > 60 ? "text-emerald-400" : "text-amber-400"}">${confVal}%</span></div>`
        : "";

      // Image thumbnail URL
      const imgUrl = (im.status === "completed")
        ? `/api/image/${data.batch_id}/${im.id}?type=processed`
        : `/api/image/${data.batch_id}/${im.id}?type=original`;

      card.innerHTML = `
        <div class="relative aspect-video bg-slate-950 overflow-hidden">
          <img loading="lazy" src="${imgUrl}" alt="" class="w-full h-full object-cover group-hover:scale-105 transition-all duration-300">
          <div class="absolute top-2 left-2 flex items-center space-x-1">
            ${statusBadge}
          </div>
          <div class="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-all flex items-end justify-between p-3">
            <span class="text-xs font-semibold text-white flex items-center space-x-1"><i data-lucide="split" class="w-3.5 h-3.5"></i><span>Inspect Slider</span></span>
            <span class="text-[11px] text-slate-300">${formatBytes(im.size_bytes)}</span>
          </div>
        </div>
        <div class="p-3 flex-1 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between">
            <span class="card-filename text-xs font-medium text-slate-200 truncate"></span>
            ${confHtml}
          </div>
          <div class="flex items-center justify-between pt-1 border-t border-slate-800 text-[11px] text-slate-400">
            <span>${im.dimensions ? `${im.dimensions[0]}×${im.dimensions[1]}` : "1376×768"}</span>
            <button class="card-dl-btn text-indigo-400 hover:text-indigo-300 flex items-center space-x-1" data-id="${im.id}">
              <i data-lucide="download" class="w-3 h-3"></i>
              <span>Download</span>
            </button>
          </div>
        </div>
      `;
      const filenameLabel = card.querySelector(".card-filename");
      filenameLabel.textContent = im.filename;
      filenameLabel.title = im.error || im.filename;

      // Click card to open inspection modal
      card.addEventListener("click", (e) => {
        if (e.target.closest(".card-dl-btn")) {
          e.stopPropagation();
          window.location.href = `/api/download/${data.batch_id}/${im.id}`;
          return;
        }
        openCompareModal(im.id);
      });

      imageGrid.appendChild(card);
    });

    if (window.lucide) lucide.createIcons();
  }

  // Filter tabs click
  filterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      filterTabs.forEach((t) => {
        t.classList.remove("active", "bg-indigo-600", "text-white");
        t.classList.add("text-slate-400");
      });
      tab.classList.add("active", "bg-indigo-600", "text-white");
      tab.classList.remove("text-slate-400");
      state.currentFilter = tab.dataset.filter;
      renderCards();
    });
  });

  // --- 4. INTERACTIVE BEFORE / AFTER COMPARISON MODAL ---
  function openCompareModal(imgId) {
    state.activeModalImgId = imgId;
    const batch = state.batchData;
    if (!batch || !batch.images[imgId]) return;

    const img = batch.images[imgId];
    modalFilename.textContent = img.filename;
    modalMeta.textContent = `${img.dimensions ? `${img.dimensions[0]}×${img.dimensions[1]}` : "1376×768"} • ${img.message || "Watermark detection"}`;

    const origUrl = `/api/image/${batch.batch_id}/${img.id}?type=original`;
    const procUrl = `/api/image/${batch.batch_id}/${img.id}?type=processed`;

    // Load images
    originalImg.src = origUrl;
    processedImg.src = procUrl;

    // Set coordinates
    if (img.bbox) {
      detectedCoords.textContent = `BBox: [${img.bbox.join(", ")}]`;
    }

    setSliderPosition(50);
    compareModal.classList.remove("hidden");
    updateBboxMarker(img);

    // Sync image sizes once loaded
    processedImg.onload = () => {
      syncOverlayImageSize();
      updateBboxMarker(img);
    };
  }

  function syncOverlayImageSize() {
    // Synchronize inner overlay img dimensions with base image
    const rect = processedImg.getBoundingClientRect();
    originalImg.style.width = `${rect.width}px`;
    originalImg.style.height = `${rect.height}px`;
  }

  window.addEventListener("resize", syncOverlayImageSize);

  function setSliderPosition(pos) {
    const clamped = Math.max(0, Math.min(100, pos));
    state.sliderPos = clamped;
    comparisonOverlay.style.width = `${clamped}%`;
    sliderHandle.style.left = `${clamped}%`;
  }

  function updateBboxMarker(img) {
    if (!showBboxToggle.checked || !img.bbox) {
      bboxMarker.classList.add("hidden");
      return;
    }

    const [bx, by, bw, bh] = img.bbox;
    const origW = img.dimensions ? img.dimensions[0] : 1376;
    const origH = img.dimensions ? img.dimensions[1] : 768;

    const rect = processedImg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const scaleX = rect.width / origW;
    const scaleY = rect.height / origH;

    bboxMarker.style.left = `${bx * scaleX}px`;
    bboxMarker.style.top = `${by * scaleY}px`;
    bboxMarker.style.width = `${bw * scaleX}px`;
    bboxMarker.style.height = `${bh * scaleY}px`;
    bboxMarker.classList.remove("hidden");
  }

  showBboxToggle.addEventListener("change", () => {
    if (state.activeModalImgId && state.batchData) {
      updateBboxMarker(state.batchData.images[state.activeModalImgId]);
    }
  });

  // Slider Drag Events
  function handleSliderMove(e) {
    if (!state.isDraggingSlider) return;
    const rect = comparisonContainer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const offsetX = clientX - rect.left;
    const pos = (offsetX / rect.width) * 100;
    setSliderPosition(pos);
  }

  sliderHandle.addEventListener("mousedown", () => (state.isDraggingSlider = true));
  sliderHandle.addEventListener("touchstart", () => (state.isDraggingSlider = true));

  window.addEventListener("mousemove", handleSliderMove);
  window.addEventListener("touchmove", handleSliderMove);

  window.addEventListener("mouseup", () => (state.isDraggingSlider = false));
  window.addEventListener("touchend", () => (state.isDraggingSlider = false));

  comparisonContainer.addEventListener("click", (e) => {
    const rect = comparisonContainer.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const pos = (offsetX / rect.width) * 100;
    setSliderPosition(pos);
  });

  // Modal navigation
  modalPrevBtn.addEventListener("click", () => navigateModal(-1));
  modalNextBtn.addEventListener("click", () => navigateModal(1));

  function navigateModal(dir) {
    if (!state.batchData || !state.activeModalImgId) return;
    const keys = Object.keys(state.batchData.images);
    const currIdx = keys.indexOf(state.activeModalImgId);
    if (currIdx === -1) return;
    const newIdx = (currIdx + dir + keys.length) % keys.length;
    openCompareModal(keys[newIdx]);
  }

  // Keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (compareModal.classList.contains("hidden")) return;
    if (e.key === "ArrowLeft") navigateModal(-1);
    if (e.key === "ArrowRight") navigateModal(1);
    if (e.key === "Escape") closeCompareModal();
  });

  closeModalBtn.addEventListener("click", closeCompareModal);
  compareModal.addEventListener("click", (e) => {
    if (e.target === compareModal) closeCompareModal();
  });

  function closeCompareModal() {
    compareModal.classList.add("hidden");
    state.activeModalImgId = null;
  }

  // Single download from modal
  modalDownloadBtn.addEventListener("click", () => {
    if (state.batchId && state.activeModalImgId) {
      window.location.href = `/api/download/${state.batchId}/${state.activeModalImgId}`;
    }
  });

  // Re-clean single image
  recleanBtn.addEventListener("click", async () => {
    if (!state.batchId || !state.activeModalImgId) return;
    try {
      recleanBtn.disabled = true;
      recleanBtn.classList.add("opacity-50");

      const img = state.batchData.images[state.activeModalImgId];
      const res = await fetch("/api/reprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          batch_id: state.batchId,
          img_id: state.activeModalImgId,
          mode: recleanModeSelect.value,
          bbox: img.bbox,
          feather: 2
        })
      });
      if (!res.ok) throw new Error("Reprocess failed");

      // Refresh modal and card
      const cacheBust = `&t=${Date.now()}`;
      processedImg.src = `/api/image/${state.batchId}/${state.activeModalImgId}?type=processed${cacheBust}`;
      state.cardsSignature = "";
      await fetchBatchStatus();
    } catch (err) {
      alert("Re-clean error: " + err.message);
    } finally {
      recleanBtn.disabled = false;
      recleanBtn.classList.remove("opacity-50");
    }
  });
});
