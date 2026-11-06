const $ = id => document.getElementById(id);
let scale = 4, result = null;
let threeRenderer = null, threeScene = null, threeCamera = null, threeControls = null, threeMesh = null, threePoints = null;
let isRotating = false;

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}

function setFile(input, meta, drop) {
  const f = input.files[0];
  $(meta).textContent = f ? `${f.name} · ${(f.size/1024/1024).toFixed(2)} MB` : '';
  drop.classList.toggle('has-file', !!f);
  const previewId = input.id === 'rgbInput' ? 'rgbPreview' : input.id === 'depthInput' ? 'depthPreview' : null;
  if (previewId) {
    const im = $(previewId);
    if (f && !f.name.toLowerCase().endsWith('.npy')) {
      im.src = URL.createObjectURL(f);
      im.hidden = false;
    } else {
      im.hidden = true;
      im.removeAttribute('src');
    }
  }
  $('runBtn').disabled = !($('rgbInput').files[0] && $('depthInput').files[0]);
}

[['rgbInput','rgbMeta','rgbDrop'],['depthInput','depthMeta','depthDrop'],['gtInput','gtMeta','gtDrop']].forEach(([i,m,d]) => {
  $(i).addEventListener('change', () => setFile($(i), m, $(d)));
  $(d).addEventListener('dragover', e => { e.preventDefault(); $(d).classList.add('drag'); });
  $(d).addEventListener('dragleave', () => $(d).classList.remove('drag'));
  $(d).addEventListener('drop', e => {
    e.preventDefault();
    $(d).classList.remove('drag');
    if (e.dataTransfer.files.length) {
      $(i).files = e.dataTransfer.files;
      setFile($(i), m, $(d));
    }
  });
});

document.querySelectorAll('#scaleChoices button').forEach(b => {
  b.addEventListener('click', () => {
    scale = +b.dataset.value;
    document.querySelectorAll('#scaleChoices button').forEach(x => x.classList.toggle('selected', x === b));
  });
});

$('alpha').addEventListener('input', e => $('alphaValue').textContent = e.target.value + '%');

async function health() {
  try {
    const r = await fetch('/api/health');
    const j = await r.json();
    $('serverStatus').textContent = 'Engine Active';
    $('serverDot').style.background = '#62d6a9';
    $('modelState').innerHTML = j.model_trained
      ? '<span class="state-icon">✦</span><div><b>Trained Residual U-Net Active</b><small>Learned edge-preserving refinement ready.</small></div>'
      : '<span class="state-icon">◌</span><div><b>Classical Engine Ready · DL Checkpoint Training</b><small>Trained weights will automatically link upon completion.</small></div>';
  } catch (e) {
    $('serverStatus').textContent = 'Server unavailable';
    $('serverDot').style.background = '#e5a45e';
    $('modelState').innerHTML = '<span class="state-icon">!</span><div><b>Backend is not responding</b><small>Ensure the FastAPI server is running.</small></div>';
  }
}
health();

// 1-Click Fast Benchmarking Presets
async function loadPreset(name, config) {
  toast(`Loading ${config.label} benchmark preset...`);
  try {
    const fetchBlob = async url => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Preset asset missing: ${url}`);
      return await res.blob();
    };

    const [rgbBlob, depthBlob, gtBlob] = await Promise.all([
      fetchBlob(config.rgbUrl),
      fetchBlob(config.depthUrl),
      fetchBlob(config.gtUrl)
    ]);

    const dtRgb = new DataTransfer(); dtRgb.items.add(new File([rgbBlob], `${name}_rgb.png`, {type: 'image/png'}));
    $('rgbInput').files = dtRgb.files; setFile($('rgbInput'), 'rgbMeta', $('rgbDrop'));

    const dtDepth = new DataTransfer(); dtDepth.items.add(new File([depthBlob], `${name}_depth.png`, {type: 'image/png'}));
    $('depthInput').files = dtDepth.files; setFile($('depthInput'), 'depthMeta', $('depthDrop'));

    const dtGt = new DataTransfer(); dtGt.items.add(new File([gtBlob], `${name}_gt.png`, {type: 'image/png'}));
    $('gtInput').files = dtGt.files; setFile($('gtInput'), 'gtMeta', $('gtDrop'));

    $('depthProfile').value = config.profile;
    $('depthUnits').value = config.units;
    $('depthInputType').value = config.inputType;
    $('mode').value = config.mode;
    $('method').value = config.method;
    scale = config.scale;
    document.querySelectorAll('#scaleChoices button').forEach(b => b.classList.toggle('selected', +b.dataset.value === scale));

    toast(`${config.label} preset loaded. Starting reconstruction...`);
    $('runBtn').click();
  } catch (err) {
    toast(`Preset error: ${err.message}`);
  }
}

$('presetNYU').addEventListener('click', () => {
  loadPreset('nyu', {
    label: 'NYU Indoor 4x',
    rgbUrl: '/assets/presets/nyu_rgb.png',
    depthUrl: '/assets/presets/nyu_depth.png',
    gtUrl: '/assets/presets/nyu_gt.png',
    profile: 'NYU indoor · 10 m',
    units: 'millimetres',
    inputType: 'low_resolution',
    mode: 'Hybrid',
    method: 'guided',
    scale: 4
  });
});

$('presetKITTI').addEventListener('click', () => {
  loadPreset('kitti', {
    label: 'KITTI Outdoor LiDAR',
    rgbUrl: '/assets/presets/kitti_rgb.png',
    depthUrl: '/assets/presets/kitti_depth.png',
    gtUrl: '/assets/presets/kitti_gt.png',
    profile: 'KITTI outdoor · 80 m',
    units: 'kitti_256',
    inputType: 'full_resolution_sparse',
    mode: 'Classical',
    method: 'joint_bilateral',
    scale: 4
  });
});

$('presetMid').addEventListener('click', () => {
  loadPreset('middlebury', {
    label: 'Middlebury Calibrated Stereo',
    rgbUrl: '/assets/presets/mid_rgb.png',
    depthUrl: '/assets/presets/mid_depth.png',
    gtUrl: '/assets/presets/mid_gt.png',
    profile: 'NYU indoor · 10 m',
    units: 'millimetres',
    inputType: 'low_resolution',
    mode: 'Hybrid',
    method: 'guided',
    scale: 4
  });
});

// 24-Scene Interactive Benchmark Gallery Logic
let galleryData = [];

async function initGallery() {
  const grid = $('galleryGrid');
  if (!grid) return;
  try {
    const res = await fetch('/assets/presets/gallery_manifest.json');
    if (!res.ok) throw new Error('Could not load gallery manifest');
    galleryData = await res.json();
    renderGallery('all');
  } catch (err) {
    grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;color:var(--muted);padding:24px;">Unable to load benchmark gallery: ${err.message}</div>`;
  }
}

function renderGallery(filterCategory) {
  const grid = $('galleryGrid');
  if (!grid) return;
  grid.innerHTML = '';
  
  const items = filterCategory === 'all' 
    ? galleryData 
    : galleryData.filter(item => item.category === filterCategory);

  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.title = `Click to reconstruct: ${item.title}`;
    card.innerHTML = `
      <div class="gallery-thumb-wrap">
        <img src="${item.thumbUrl}" alt="${item.title}" loading="lazy">
        <span class="gallery-badge ${item.badgeClass}">${item.badge}</span>
      </div>
      <div class="gallery-card-body">
        <div class="gallery-scene-title" title="${item.title}">${item.title}</div>
        <div class="gallery-meta-tags">
          <span>${item.dataset}</span>
          <span>${item.resolution}</span>
          <span>${item.range}</span>
        </div>
        <button class="gallery-run-btn" data-id="${item.id}">
          <span>⚡ Run Reconstruction</span>
          <span>↗</span>
        </button>
      </div>
    `;

    card.addEventListener('click', () => {
      loadPreset(item.id, {
        label: `${item.title} (${item.dataset})`,
        rgbUrl: item.rgbUrl,
        depthUrl: item.depthUrl,
        gtUrl: item.gtUrl,
        profile: item.profile,
        units: item.units,
        inputType: item.inputType,
        mode: item.mode,
        method: item.method,
        scale: item.scale
      });
    });

    grid.appendChild(card);
  });
}

// Category filter buttons
document.querySelectorAll('#galleryFilters .filter-pill').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#galleryFilters .filter-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderGallery(btn.dataset.category);
  });
});

// Hero showcase specimen click
const heroShowcase = $('heroShowcase');
if (heroShowcase) {
  heroShowcase.addEventListener('click', () => {
    $('presetNYU').click();
  });
}

// Initialize gallery on load
initGallery();

// View mode tabs (Grid vs Split Slider vs 3D)
$('tabGrid').addEventListener('click', () => switchView('grid'));
$('tabSlider').addEventListener('click', () => switchView('slider'));
$('tab3D').addEventListener('click', () => switchView('3d'));

function switchView(mode) {
  $('tabGrid').classList.toggle('active', mode === 'grid');
  $('tabSlider').classList.toggle('active', mode === 'slider');
  $('tab3D').classList.toggle('active', mode === '3d');

  $('viewGrid').hidden = mode !== 'grid';
  $('viewSlider').hidden = mode !== 'slider';
  $('view3D').hidden = mode !== '3d';

  if (mode === '3d' && result) {
    initOrUpdate3D();
  }
}

// Reconstruction Execution
$('runBtn').addEventListener('click', async () => {
  const btn = $('runBtn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.innerHTML = '<span>Reconstructing edges…</span><span>◌</span>';

  const fd = new FormData();
  fd.append('rgb_file', $('rgbInput').files[0]);
  fd.append('depth_file', $('depthInput').files[0]);
  fd.append('scale', scale);
  fd.append('method', $('method').value);
  fd.append('interpolation', $('interpolation').value);
  fd.append('mode', $('mode').value);
  fd.append('alpha', +$('alpha').value / 100);
  fd.append('depth_profile', $('depthProfile').value);
  fd.append('depth_units', $('depthUnits').value);
  fd.append('depth_input_type', $('depthInputType').value);
  if ($('gtInput').files[0]) fd.append('gt_file', $('gtInput').files[0]);

  try {
    const res = await fetch('/api/reconstruct', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Reconstruction failed');
    result = data;

    $('interpolationImg').src = data.interpolation;
    $('knnImg').src = data.knn;
    $('classicalImg').src = data.classical;
    $('deepImg').src = data.deep;
    $('hybridImg').src = data.hybrid;
    $('boundaryImg').src = data.boundary;
    $('outputImg').src = data.output;
    
    // Set split-slider images
    $('sliderImgBefore').src = data.interpolation;
    $('sliderImgAfter').src = data.output;

    $('errorCard').hidden = !data.error_map;
    if (data.error_map) $('errorImg').src = data.error_map;

    $('latency').textContent = data.latency_ms + ' ms';
    $('scaleMetric').textContent = data.scale + '×';
    $('device').textContent = data.device.toUpperCase();
    $('density').textContent = (data.valid_fraction * 100).toFixed(1) + '%';
    $('resultMeta').textContent = `${data.input_size[0]} × ${data.input_size[1]} PX · ${$('mode').value.toUpperCase()} · SCALE ${scale}×`;
    $('modelWarning').hidden = data.trained || !['Hybrid','Deep learning'].includes($('mode').value);
    
    $('results').hidden = false;
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

    if (data.metrics) {
      $('metricPanel').innerHTML = `
        <table class="metric-table">
          <thead><tr><th>METRIC</th><th>VALUE</th><th>BENCHMARK INTERPRETATION</th></tr></thead>
          <tbody>
            ${Object.entries(data.metrics).map(([k,v]) => `
              <tr>
                <td><b>${k}</b></td>
                <td style="color:#7ee0d4;font-family:var(--mono);font-weight:700">${v === null ? '—' : Number(v).toFixed(5)}</td>
                <td>${({
                  MAE_m: 'Physical mean error in metres (Lower is better)',
                  RMSE_m: 'Root mean square error in metres (Lower is better)',
                  PSNR_dB: 'Peak signal-to-noise ratio in dB (Higher is better)',
                  SSIM: 'Structural similarity index (Closer to 1 is better)',
                  Edge_F1: 'High-frequency edge preservation F1 score (Higher is better)',
                  Boundary_F1_2px: 'Boundary accuracy at 2-pixel tolerance band (Higher is better)',
                  Boundary_MAE_m: 'Error strictly concentrated on geometric boundaries (Lower is better)'
                })[k] || 'Reported dataset statistic'}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
        <p style="font-size:9px;color:#f0cf7e;padding:12px 16px;background:#18160e;border-radius:6px;margin:12px">${data.metric_note}</p>
      `;
    } else {
      $('metricPanel').innerHTML = '<div class="eval-empty"><div>⌁</div><b>No ground truth supplied</b><p>Provide a ground truth depth map or pick a 1-click preset to evaluate exact RMSE and Boundary F1.</p></div>';
    }

    if (!$('view3D').hidden) {
      initOrUpdate3D();
    }
    renderEdgeOverlay();
    toast('Depth reconstruction completed successfully.');
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.innerHTML = '<span>Run reconstruction</span><span>↗</span>';
  }
});

// Edge Discontinuity High-Contrast Overlay
function renderEdgeOverlay() {
  const checkbox = $('toggleEdgeOverlay');
  const canvas = $('outputEdgeCanvas');
  if (!checkbox || !canvas) return;
  if (!result || !result.boundary || !checkbox.checked) {
    canvas.hidden = true;
    return;
  }
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    canvas.width = img.naturalWidth || 640;
    canvas.height = img.naturalHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const off = document.createElement('canvas');
    off.width = canvas.width;
    off.height = canvas.height;
    const octx = off.getContext('2d');
    octx.drawImage(img, 0, 0);
    const idata = octx.getImageData(0, 0, off.width, off.height);
    const d = idata.data;

    const outData = ctx.createImageData(canvas.width, canvas.height);
    const outD = outData.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i];
      if (v > 30) {
        // High-contrast fluorescent cyan edge glow (#73e0dc)
        outD[i] = 115;
        outD[i + 1] = 224;
        outD[i + 2] = 220;
        outD[i + 3] = Math.min(255, v * 2);
      } else {
        outD[i + 3] = 0;
      }
    }
    ctx.putImageData(outData, 0, 0);
    canvas.hidden = false;
  };
  img.src = result.boundary;
}

$('toggleEdgeOverlay').addEventListener('change', renderEdgeOverlay);

// Interactive Split-Comparison Slider Logic
const compareContainer = $('compareContainer');
const sliderOverlay = $('sliderOverlay');
const sliderHandle = $('sliderHandle');
let isSliding = false;

function setSliderPosition(x) {
  const rect = compareContainer.getBoundingClientRect();
  let pos = (x - rect.left) / rect.width;
  pos = Math.max(0.02, Math.min(0.98, pos));
  sliderOverlay.style.width = (pos * 100) + '%';
  sliderHandle.style.left = (pos * 100) + '%';
}

compareContainer.addEventListener('mousedown', e => { isSliding = true; setSliderPosition(e.clientX); });
window.addEventListener('mousemove', e => { if (isSliding) setSliderPosition(e.clientX); });
window.addEventListener('mouseup', () => { isSliding = false; });
compareContainer.addEventListener('touchstart', e => { isSliding = true; setSliderPosition(e.touches[0].clientX); });
window.addEventListener('touchmove', e => { if (isSliding) setSliderPosition(e.touches[0].clientX); });
window.addEventListener('touchend', () => { isSliding = false; });

// Interactive 3D Point Cloud & Surface Mesh with Three.js
function initOrUpdate3D() {
  if (!window.THREE || !result) return;
  const wrap = $('threeCanvasWrap');
  const canvas = $('threeCanvas');
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;

  if (!threeRenderer) {
    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x080d18);
    threeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    threeCamera.position.set(0, -18, 22);

    threeRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    threeControls = new THREE.OrbitControls(threeCamera, canvas);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    threeScene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0x73e0dc, 0.6);
    dirLight.position.set(5, 10, 15);
    threeScene.add(dirLight);

    function animate() {
      requestAnimationFrame(animate);
      if (isRotating && threeMesh) {
        threeMesh.rotation.z += 0.005;
        if (threePoints) threePoints.rotation.z += 0.005;
      }
      threeControls.update();
      threeRenderer.render(threeScene, threeCamera);
    }
    animate();

    window.addEventListener('resize', () => {
      if (!threeCamera || !threeRenderer) return;
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      threeCamera.aspect = w / h;
      threeCamera.updateProjectionMatrix();
      threeRenderer.setSize(w, h);
    });
  }

  // Load Depth and RGB onto 3D Geometry
  const depthImg = new Image();
  depthImg.crossOrigin = "anonymous";
  depthImg.onload = () => {
    const sw = 160, sh = 120;
    const offCanvas = document.createElement('canvas');
    offCanvas.width = sw; offCanvas.height = sh;
    const ctx = offCanvas.getContext('2d');
    ctx.drawImage(depthImg, 0, 0, sw, sh);
    const dData = ctx.getImageData(0, 0, sw, sh).data;

    if (threeMesh) threeScene.remove(threeMesh);
    if (threePoints) threeScene.remove(threePoints);

    const geom = new THREE.PlaneGeometry(24, 18, sw - 1, sh - 1);
    const pos = geom.attributes.position;
    const elev = +$('threeElevation').value / 10;

    for (let i = 0; i < pos.count; i++) {
      const depthVal = dData[i * 4] / 255.0; // Normalized 0..1
      pos.setZ(i, (1.0 - depthVal) * elev);
    }
    geom.computeVertexNormals();

    const texture = new THREE.TextureLoader().load(result.output);
    const rgbInput = $('rgbPreview');
    const rgbTexture = (rgbInput && rgbInput.src) ? new THREE.TextureLoader().load(rgbInput.src) : texture;

    const mat = new THREE.MeshStandardMaterial({
      map: rgbTexture,
      roughness: 0.4,
      metalness: 0.1,
      side: THREE.DoubleSide
    });
    threeMesh = new THREE.Mesh(geom, mat);
    threeScene.add(threeMesh);

    // Points Cloud representation
    const pMat = new THREE.PointsMaterial({
      size: 0.12,
      map: rgbTexture,
      color: 0x73e0dc
    });
    threePoints = new THREE.Points(geom, pMat);
    threePoints.visible = false;
    threeScene.add(threePoints);
  };
  depthImg.src = result.output;
}

$('threeElevation').addEventListener('input', e => {
  if (!threeMesh) return;
  const pos = threeMesh.geometry.attributes.position;
  const elev = +e.target.value / 10;
  // Dynamic elevation update
  initOrUpdate3D();
});

$('btnModeMesh').addEventListener('click', () => {
  if (threeMesh) { threeMesh.visible = true; threeMesh.material.wireframe = false; }
  if (threePoints) threePoints.visible = false;
  $('btnModeMesh').classList.add('active');
  $('btnModePoints').classList.remove('active');
  $('btnModeWire').classList.remove('active');
});

$('btnModePoints').addEventListener('click', () => {
  if (threeMesh) threeMesh.visible = false;
  if (threePoints) threePoints.visible = true;
  $('btnModeMesh').classList.remove('active');
  $('btnModePoints').classList.add('active');
  $('btnModeWire').classList.remove('active');
});

$('btnModeWire').addEventListener('click', () => {
  if (threeMesh) { threeMesh.visible = true; threeMesh.material.wireframe = true; }
  if (threePoints) threePoints.visible = false;
  $('btnModeMesh').classList.remove('active');
  $('btnModePoints').classList.remove('active');
  $('btnModeWire').classList.add('active');
});

$('btnAutoRotate').addEventListener('click', () => {
  isRotating = !isRotating;
  $('btnAutoRotate').classList.toggle('active', isRotating);
});

$('btnResetCamera').addEventListener('click', () => {
  if (threeCamera && threeControls) {
    threeCamera.position.set(0, -18, 22);
    threeControls.target.set(0, 0, 0);
    threeControls.update();
  }
});

// Download Handlers
function downloadBlob(blob, name) {
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('downloadOutput').onclick = () => {
  if (!result) return;
  fetch(result.output_png).then(r => r.blob()).then(b => downloadBlob(b, 'depthlab_refined_uint16.png'));
};

$('downloadNpy').onclick = () => {
  if (!result) return;
  const raw = atob(result.output_npy);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  downloadBlob(new Blob([bytes], { type: 'application/octet-stream' }), 'depthlab_refined_metres.npy');
};
