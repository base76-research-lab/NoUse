/**
 * Nous City View — Isometrisk manga-stad
 * Varje domän = kvarter, varje koncept = byggnad
 * Driven av systemklockan (dag/natt-cykel)
 */

'use strict';

// ── Konstanter ────────────────────────────────────────────────────
const CITY_BLOCK_SIZE  = 120;   // bredd/djup per domänkvarter (units)
const CITY_STREET_GAP  = 40;    // mellanrum (gata) mellan kvarter
const BUILDING_SPACING = 18;    // avstånd mellan byggnader inom kvarter
const MAX_BUILDING_H   = 80;    // max byggnads-höjd
const MIN_BUILDING_H   = 8;     // min byggnads-höjd
const PAGODA_TIERS     = 4;     // antal pagoda-våningar för meta-axiom

// Färgpalett — varm fantasistad, manga-noir
const C = {
  skyDay:    0xc4d8f0,
  skyDusk:   0x2a0e3a,    // djup lila skymning
  skyNight:  0x08050f,    // nästan svart natt
  skyDawn:   0x1a0820,

  ambDay:    { color: 0xffe8c8, intensity: 1.6 },
  ambDusk:   { color: 0xff7a20, intensity: 0.8 },
  ambNight:  { color: 0x1a1040, intensity: 0.28 },
  ambDawn:   { color: 0xff8050, intensity: 0.5 },

  streetLight: 0xff9a30,   // varm bärnstenslykta (som bild 1)
  neonActive:  0xff6a00,   // aktiv byggnad — orange glöd
  windowNight: 0xffb830,   // varmt amber fönsterljus (som bild 2)
  windowDay:   0xfff8f0,
  bridgeColor: 0xffd700,   // guld för bisociation-broar
  groundDay:   0x3a342a,   // varm gråbrun gatsten (dag)
  groundNight: 0x120e08,   // djup varm svart (natt)
  outline:     0x080408,   // nästans svart outline
};

// Fasövergångar (i decimala timmar)
const PHASES = [
  { start:  5, end:  7, name: 'dawn'  },
  { start:  7, end: 17, name: 'day'   },
  { start: 17, end: 20, name: 'dusk'  },
  { start: 20, end: 29, name: 'night' }, // 29 = 5 nästa dag (wrap)
];

const CITY_UI_FONT = "'Yu Mincho', 'Hiragino Mincho ProN', 'Noto Serif JP', Georgia, serif";

// ── Hjälpare ──────────────────────────────────────────────────────
function lerp(a, b, t) { return a + (b - a) * Math.clamp(t, 0, 1); }
Math.clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function hexLerp(hexA, hexB, t) {
  const ra = (hexA >> 16) & 0xff, ga = (hexA >> 8) & 0xff, ba = hexA & 0xff;
  const rb = (hexB >> 16) & 0xff, gb = (hexB >> 8) & 0xff, bb = hexB & 0xff;
  return (
    (Math.round(lerp(ra, rb, t)) << 16) |
    (Math.round(lerp(ga, gb, t)) << 8)  |
     Math.round(lerp(ba, bb, t))
  );
}

function currentHour() {
  const now = new Date();
  return now.getHours() + now.getMinutes() / 60;
}

function getTimePhase(h) {
  // normalise 0..24
  const h24 = ((h % 24) + 24) % 24;
  if (h24 >= 5  && h24 < 7)  return { phase: 'dawn',  t: (h24 - 5)  / 2  };
  if (h24 >= 7  && h24 < 17) return { phase: 'day',   t: (h24 - 7)  / 10 };
  if (h24 >= 17 && h24 < 20) return { phase: 'dusk',  t: (h24 - 17) / 3  };
  // night: 20..5 (wrapping)
  const nightH = h24 >= 20 ? h24 - 20 : h24 + 4;
  return { phase: 'night', t: nightH / 9 };
}

function cityFallbackHumanize(value, fallback = 'Unknown') {
  const raw = String(value || '').trim();
  if (!raw) return fallback;
  return raw
    .replace(/^META::/i, 'Meta ')
    .replace(/::/g, ' / ')
    .replace(/[_.]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .map(token => {
      if (!token) return token;
      if (/[0-9:/-]/.test(token)) return token;
      return token.charAt(0).toUpperCase() + token.slice(1);
    })
    .join(' ');
}

function cityFormatDomainName(domain) {
  if (typeof formatDomainName === 'function') return formatDomainName(domain);
  return cityFallbackHumanize(domain, 'Unknown');
}

function cityFormatNodeName(id) {
  if (typeof formatNodeName === 'function') return formatNodeName(id);
  return cityFallbackHumanize(id, 'Unknown node');
}

function cityWrapLines(text, maxLen = 18, maxLines = 2) {
  if (typeof wrapDisplayLines === 'function') return wrapDisplayLines(text, maxLen, maxLines);
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  if (!clean) return ['Unknown'];
  const words = clean.split(' ');
  const lines = [];
  let current = '';
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxLen) {
      current = candidate;
      continue;
    }
    if (current) lines.push(current);
    current = word;
    if (lines.length >= maxLines - 1) break;
  }
  if (current && lines.length < maxLines) lines.push(current);
  return lines.slice(0, maxLines);
}

function cityRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function makeTextBadge(text, color, opts = {}) {
  const {
    maxLen = 18,
    maxLines = 2,
    fontSize = 26,
    minWidth = 170,
    maxWidth = 420,
    opacity = 0.96,
  } = opts;

  const lines = cityWrapLines(text, maxLen, maxLines);
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = lines.length > 1 ? 132 : 104;
  const ctx = canvas.getContext('2d');
  const lineHeight = fontSize + 6;
  const padX = 26;
  const padY = 16;

  ctx.font = `700 ${fontSize}px ${CITY_UI_FONT}`;
  const widths = lines.map(line => ctx.measureText(line).width);
  const cardWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(...widths) + padX * 2));
  const cardHeight = padY * 2 + lines.length * lineHeight;
  const x = (canvas.width - cardWidth) / 2;
  const y = (canvas.height - cardHeight) / 2;

  ctx.save();
  ctx.shadowColor = 'rgba(25, 17, 16, 0.28)';
  ctx.shadowBlur = 20;
  ctx.fillStyle = 'rgba(245, 236, 220, 0.96)';
  ctx.strokeStyle = `#${color.toString(16).padStart(6, '0')}`;
  ctx.lineWidth = 3;
  cityRoundedRect(ctx, x, y, cardWidth, cardHeight, 20);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = '#1c1517';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `700 ${fontSize}px ${CITY_UI_FONT}`;
  lines.forEach((line, idx) => {
    const ly = y + padY + lineHeight / 2 + idx * lineHeight + 2;
    ctx.fillText(line, canvas.width / 2, ly);
  });

  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(lines.length > 1 ? 42 : 34, lines.length > 1 ? 10.5 : 8.2, 1);
  sprite.renderOrder = 12;
  sprite.userData.lineCount = lines.length;
  return sprite;
}

// ── Gatsten-textur (cobblestone, varm medievalstil) ───────────────
function makeScreentoneTexture(size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Mörkbrun bas
  ctx.fillStyle = '#1a140a';
  ctx.fillRect(0, 0, size, size);

  // Gatsten-block
  const cols = 8, rows = 10;
  const bw = size / cols, bh = size / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const offset = (r % 2) * bw * 0.5; // förskjutning varannan rad
      const x = c * bw + offset;
      const y = r * bh;
      const brightness = 0.14 + Math.random() * 0.06;
      ctx.fillStyle = `rgba(${Math.round(brightness*180)},${Math.round(brightness*140)},${Math.round(brightness*100)},1)`;
      ctx.fillRect(x + 1.5, y + 1.5, bw - 3, bh - 3);
    }
  }

  // Fogar (mörka linjer)
  ctx.strokeStyle = 'rgba(8,4,2,0.9)';
  ctx.lineWidth = 2;
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * bh);
    ctx.lineTo(size, r * bh);
    ctx.stroke();
  }
  for (let r = 0; r < rows; r++) {
    const offset = (r % 2) * bw * 0.5;
    for (let c = 0; c <= cols; c++) {
      ctx.beginPath();
      ctx.moveTo(c * bw + offset, r * bh);
      ctx.lineTo(c * bw + offset, (r + 1) * bh);
      ctx.stroke();
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(12, 12);
  return tex;
}

// ── Outline-trick (back-face scale) ──────────────────────────────
function addOutline(mesh, thickness = 0.06) {
  const outMat = new THREE.MeshBasicMaterial({
    color: C.outline,
    side: THREE.BackSide,
  });
  const outMesh = new THREE.Mesh(mesh.geometry, outMat);
  outMesh.scale.setScalar(1 + thickness);
  mesh.add(outMesh);
}

// ── Byggnadsgeometri ──────────────────────────────────────────────
function makeBuildingMesh(w, h, d, color, isMeta) {
  const group = new THREE.Group();

  if (isMeta) {
    // Kunskapstorn (som bild 2): cylindriskt torn med flera balkonger + spira
    const baseR = w * 0.55;
    // Cylindrisk kropp
    const bodyGeo = new THREE.CylinderGeometry(baseR * 0.75, baseR, h, 12);
    const bodyMat = new THREE.MeshToonMaterial({ color });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h / 2;
    addOutline(body, 0.06);
    group.add(body);

    // Balkonger (utskjutande ringar på 3 höjder)
    for (let i = 1; i <= 3; i++) {
      const balkH = (h / 4) * i;
      const balkGeo = new THREE.CylinderGeometry(baseR + 3, baseR + 3, 1.5, 12);
      const balkMat = new THREE.MeshToonMaterial({ color: 0x8a6030 });
      const balk = new THREE.Mesh(balkGeo, balkMat);
      balk.position.y = balkH;
      addOutline(balk, 0.05);
      group.add(balk);
    }

    // Spira i toppen
    const spireGeo = new THREE.ConeGeometry(baseR * 0.4, h * 0.35, 8);
    const spireMat = new THREE.MeshToonMaterial({ color: 0x2a1a50 });
    const spire = new THREE.Mesh(spireGeo, spireMat);
    spire.position.y = h + h * 0.175;
    addOutline(spire, 0.06);
    group.add(spire);

    // Gyllene glöd vid torntoppen
    const glowLight = new THREE.PointLight(0xffd700, 2.5, 120);
    glowLight.position.y = h + 10;
    group.add(glowLight);
    group.userData.glowLight = glowLight;

    // Glow ring runt botten
    const ringGeo = new THREE.RingGeometry(baseR + 2, baseR + 5, 16);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffd700, side: THREE.DoubleSide,
      transparent: true, opacity: 0.45,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.5;
    group.add(ring);
  } else {
    // Vanlig byggnad: box med tak-kant (manga-stil)
    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const bodyMat = new THREE.MeshToonMaterial({ color });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = h / 2;
    addOutline(body, 0.05);
    group.add(body);

    // Tak-list
    const rimGeo = new THREE.BoxGeometry(w + 1.5, 1.2, d + 1.5);
    const rimMat = new THREE.MeshToonMaterial({ color: 0x111111 });
    const rim = new THREE.Mesh(rimGeo, rimMat);
    rim.position.y = h + 0.6;
    group.add(rim);

    // Fönster (emissive plan)
    const winCols = Math.max(1, Math.floor(w / 5));
    const winRows = Math.max(1, Math.floor(h / 7));
    for (let c = 0; c < winCols; c++) {
      for (let r = 0; r < winRows; r++) {
        const wGeo = new THREE.PlaneGeometry(2, 2.5);
        const wMat = new THREE.MeshBasicMaterial({
          color: C.windowDay,
          transparent: true,
          opacity: 0.0, // sätts av dag/natt
        });
        const win = new THREE.Mesh(wGeo, wMat);
        win.position.set(
          -w / 2 + 3 + c * 5,
          4 + r * 7,
          d / 2 + 0.05,
        );
        win.userData.isWindow = true;
        group.add(win);
      }
    }

    // Inre glöd-ljus (blöder ut som varmt fönsterljus)
    const glow = new THREE.PointLight(C.windowNight, 0, 50);
    glow.position.set(0, h * 0.5, 0);
    group.add(glow);
    group.userData.windowLight = glow;
  }

  return group;
}

// ── Gatulyktor ────────────────────────────────────────────────────
function makeStreetLight() {
  const group = new THREE.Group();
  // Stolpe
  const poleGeo = new THREE.CylinderGeometry(0.3, 0.3, 14, 6);
  const poleMat = new THREE.MeshToonMaterial({ color: 0x334455 });
  const pole = new THREE.Mesh(poleGeo, poleMat);
  pole.position.y = 7;
  group.add(pole);
  // Arm
  const armGeo = new THREE.BoxGeometry(5, 0.4, 0.4);
  const armMat = new THREE.MeshToonMaterial({ color: 0x334455 });
  const arm = new THREE.Mesh(armGeo, armMat);
  arm.position.set(2.5, 14, 0);
  group.add(arm);
  // Lampa (glöd)
  const lampGeo = new THREE.SphereGeometry(1.2, 8, 6);
  const lampMat = new THREE.MeshBasicMaterial({ color: C.streetLight });
  const lamp = new THREE.Mesh(lampGeo, lampMat);
  lamp.position.set(5, 14, 0);
  group.add(lamp);
  // Ljuskälla
  const light = new THREE.PointLight(C.streetLight, 0, 60);
  light.position.set(5, 14, 0);
  group.add(light);
  group.userData.pointLight = light;
  group.userData.lamp = lamp;
  return group;
}

// ── Bro (bisociation) ─────────────────────────────────────────────
function makeBridge(startVec, endVec) {
  const mid = startVec.clone().add(endVec).multiplyScalar(0.5);
  mid.y = 50; // bågpunkt uppåt

  const curve = new THREE.QuadraticBezierCurve3(startVec, mid, endVec);
  const points = curve.getPoints(40);
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({
    color: C.bridgeColor,
    transparent: true,
    opacity: 0.7,
    linewidth: 2,
  });
  return new THREE.Line(geo, mat);
}

// ── Layout-algoritm ───────────────────────────────────────────────
function computeLayout(graphData) {
  // Gruppera noder per domän
  const domains = {};
  graphData.nodes.forEach(n => {
    const d = n.group || 'unknown';
    if (!domains[d]) domains[d] = [];
    domains[d].push(n);
  });

  const domainList = Object.keys(domains);
  const cols = Math.ceil(Math.sqrt(domainList.length));
  const step = CITY_BLOCK_SIZE + CITY_STREET_GAP;

  const districtPos = {}; // domän → {x, z} center
  const nodePos = {};     // nodeId → {x, y, z}

  domainList.forEach((dom, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const cx = col * step;
    const cz = row * step;
    districtPos[dom] = { x: cx, z: cz };

    const nodes = domains[dom];
    const perRow = Math.ceil(Math.sqrt(nodes.length));

    nodes.forEach((n, ni) => {
      const nc = ni % perRow;
      const nr = Math.floor(ni / perRow);
      nodePos[n.id] = {
        x: cx - CITY_BLOCK_SIZE / 2 + nc * BUILDING_SPACING + BUILDING_SPACING / 2,
        y: 0,
        z: cz - CITY_BLOCK_SIZE / 2 + nr * BUILDING_SPACING + BUILDING_SPACING / 2,
      };
    });
  });

  return { domains, domainList, districtPos, nodePos };
}

// ── Huvud CityView-klass ──────────────────────────────────────────
class CityView {
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.animId = null;
    this.graphData = { nodes: [], links: [] };
    this.buildingMap = {};   // nodeId → THREE.Group
    this.lightMap = {};      // grupperad per distrikt
    this.streetLights = [];  // alla gatulyktor
    this.districtLabels = [];
    this.ambientLight = null;
    this.dirLight = null;
    this.layout = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.dayNightInterval = null;
    this.arousal = 0.5;
    this.hoveredNodeId = '';
    this.hoverBadge = null;

    this._onMouseMove = this._onMouseMove.bind(this);
    this._onClick = this._onClick.bind(this);
    this._onResize = this._onResize.bind(this);
  }

  // ── Init ──────────────────────────────────────────────────────
  init(graphData) {
    this.graphData = graphData;
    this._setupRenderer();
    this._setupScene();
    this._buildCity();
    this._applyDayNight(currentHour());
    this._setupControls();
    this._startLoop();

    // Dag/natt uppdatering var 60s
    this.dayNightInterval = setInterval(() => {
      this._applyDayNight(currentHour());
    }, 60000);

    // Limbic poll var 30s
    this._pollLimbic();
    this.limbicInterval = setInterval(() => this._pollLimbic(), 30000);
  }

  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.dayNightInterval) clearInterval(this.dayNightInterval);
    if (this.limbicInterval) clearInterval(this.limbicInterval);
    this.container.removeEventListener('mousemove', this._onMouseMove);
    this.container.removeEventListener('click', this._onClick);
    window.removeEventListener('resize', this._onResize);
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }
    this.scene = null;
    this.buildingMap = {};
    this.streetLights = [];
    this.districtLabels = [];
    this.hoveredNodeId = '';
    this.hoverBadge = null;
  }

  // ── Renderer ─────────────────────────────────────────────────
  _setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.container.appendChild(this.renderer.domElement);
  }

  // ── Scen ─────────────────────────────────────────────────────
  _setupScene() {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(C.skyNight, 0.004);

    // Kamera: isometrisk vinkel
    const aspect = window.innerWidth / window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 1, 5000);
    this.camera.position.set(300, 350, 300);
    this.camera.lookAt(0, 0, 0);

    // Ambient
    this.ambientLight = new THREE.AmbientLight(C.ambNight.color, C.ambNight.intensity);
    this.scene.add(this.ambientLight);

    // Direktionellt (sol/måne)
    this.dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    this.dirLight.position.set(200, 400, 150);
    this.dirLight.castShadow = true;
    this.scene.add(this.dirLight);

    // Mark
    const groundTex = makeScreentoneTexture();
    const groundGeo = new THREE.PlaneGeometry(4000, 4000);
    const groundMat = new THREE.MeshToonMaterial({ map: groundTex, color: C.groundNight });
    this.groundMesh = new THREE.Mesh(groundGeo, groundMat);
    this.groundMesh.rotation.x = -Math.PI / 2;
    this.groundMesh.receiveShadow = true;
    this.scene.add(this.groundMesh);
  }

  // ── Bygg staden ───────────────────────────────────────────────
  _buildCity() {
    this.layout = computeLayout(this.graphData);
    const { domains, domainList, districtPos, nodePos } = this.layout;

    // Centrera scenen
    const totalW = (Math.ceil(Math.sqrt(domainList.length))) * (CITY_BLOCK_SIZE + CITY_STREET_GAP);
    const offset = totalW / 2;

    // Kvartersgolv + distrikt-label
    domainList.forEach(dom => {
      const { x, z } = districtPos[dom];
      const color = this._domainColorHex(dom);

      // Kvarters-markering
      const blockGeo = new THREE.PlaneGeometry(CITY_BLOCK_SIZE - 4, CITY_BLOCK_SIZE - 4);
      const blockMat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.06,
        side: THREE.DoubleSide,
      });
      const block = new THREE.Mesh(blockGeo, blockMat);
      block.rotation.x = -Math.PI / 2;
      block.position.set(x - offset, 0.3, z - offset);
      this.scene.add(block);

      // Kvarters-kant (outline)
      const edgeGeo = new THREE.EdgesGeometry(
        new THREE.BoxGeometry(CITY_BLOCK_SIZE - 4, 0.5, CITY_BLOCK_SIZE - 4)
      );
      const edgeMat = new THREE.LineBasicMaterial({ color, opacity: 0.3, transparent: true });
      const edges = new THREE.LineSegments(edgeGeo, edgeMat);
      edges.position.set(x - offset, 0.3, z - offset);
      this.scene.add(edges);

      // Gatulyktor i hörnen
      const corners = [
        [x - CITY_BLOCK_SIZE/2, z - CITY_BLOCK_SIZE/2],
        [x + CITY_BLOCK_SIZE/2, z - CITY_BLOCK_SIZE/2],
        [x - CITY_BLOCK_SIZE/2, z + CITY_BLOCK_SIZE/2],
        [x + CITY_BLOCK_SIZE/2, z + CITY_BLOCK_SIZE/2],
      ];
      corners.forEach(([lx, lz]) => {
        const sl = makeStreetLight();
        sl.position.set(lx - offset, 0, lz - offset);
        this.scene.add(sl);
        this.streetLights.push(sl);
      });

      const districtLabel = makeTextBadge(cityFormatDomainName(dom), color, {
        fontSize: 24,
        maxLen: 16,
        maxLines: 2,
        minWidth: 180,
        maxWidth: 360,
        opacity: 0.92,
      });
      districtLabel.position.set(x - offset, 20, z - offset);
      this.scene.add(districtLabel);
      this.districtLabels.push(districtLabel);
    });

    // Byggnader
    this.graphData.nodes.forEach(n => {
      const pos = nodePos[n.id];
      if (!pos) return;

      const isMeta = n.isMeta || (n.id || '').startsWith('META::');
      const ev = n.evidence || 0.3;
      const deg = n._degree || 1;
      const h = isMeta
        ? MAX_BUILDING_H
        : Math.max(MIN_BUILDING_H, Math.min(MAX_BUILDING_H, ev * 60 + deg * 2));
      const w = isMeta ? 14 : Math.max(8, Math.min(16, deg * 1.5 + 7));
      const color = this._domainColorHex(n.group);

      const building = makeBuildingMesh(w, h, w, color, isMeta);
      building.position.set(pos.x - offset, 0, pos.z - offset);
      building.userData.nodeId = n.id;
      building.userData.nodeData = n;
      building.userData.labelY = h + (isMeta ? 22 : 14);
      building.userData.baseScale = 1;
      building.castShadow = true;
      this.scene.add(building);
      this.buildingMap[n.id] = building;
    });

    // Bisociation-broar (hämtas asynkront)
    this._loadBridges(offset);

    // Centrera kamera på mitten
    this.camera.position.set(offset * 0.3, 350, offset * 1.2);
    this.camera.lookAt(0, 0, 0);
  }

  async _loadBridges(offset) {
    try {
      const resp = await fetch('/api/bisoc?max_domains=10&tau_threshold=0.7');
      if (!resp.ok) return;
      const data = await resp.json();
      const candidates = data.candidates || data;
      if (!Array.isArray(candidates)) return;

      candidates.slice(0, 8).forEach(c => {
        const pa = this.layout.districtPos[c.domain_a];
        const pb = this.layout.districtPos[c.domain_b];
        if (!pa || !pb) return;
        const bridge = makeBridge(
          new THREE.Vector3(pa.x - offset, 5, pa.z - offset),
          new THREE.Vector3(pb.x - offset, 5, pb.z - offset),
        );
        this.scene.add(bridge);
      });
    } catch { /* bisoc ej tillgänglig */ }
  }

  // ── Dag/natt ──────────────────────────────────────────────────
  _applyDayNight(h) {
    const { phase, t } = getTimePhase(h);
    let skyColor, ambColor, ambInt, dirInt, fogDensity, groundColor;
    let lightsOn = false;

    if (phase === 'dawn') {
      skyColor    = hexLerp(C.skyNight, C.skyDay, t);
      ambColor    = hexLerp(C.ambNight.color, C.ambDawn.color, t);
      ambInt      = lerp(C.ambNight.intensity, C.ambDawn.intensity, t);
      dirInt      = lerp(0.1, 0.6, t);
      fogDensity  = lerp(0.006, 0.003, t);
      groundColor = hexLerp(C.groundNight, C.groundDay, t);
      lightsOn    = t < 0.5;
    } else if (phase === 'day') {
      skyColor    = C.skyDay;
      ambColor    = C.ambDay.color;
      ambInt      = C.ambDay.intensity;
      dirInt      = 1.8;
      fogDensity  = 0.002;
      groundColor = C.groundDay;
      lightsOn    = false;
    } else if (phase === 'dusk') {
      skyColor    = hexLerp(C.skyDay, C.skyDusk, t);
      ambColor    = hexLerp(C.ambDay.color, C.ambDusk.color, t);
      ambInt      = lerp(C.ambDay.intensity, C.ambDusk.intensity, t);
      dirInt      = lerp(1.8, 0.4, t);
      fogDensity  = lerp(0.002, 0.005, t);
      groundColor = hexLerp(C.groundDay, C.groundNight, t);
      lightsOn    = t > 0.4;
    } else { // night
      skyColor    = C.skyNight;
      ambColor    = C.ambNight.color;
      ambInt      = C.ambNight.intensity + this.arousal * 0.2;
      dirInt      = 0.1;
      fogDensity  = 0.005;
      groundColor = C.groundNight;
      lightsOn    = true;
    }

    this.scene.background = new THREE.Color(skyColor);
    this.scene.fog.color.setHex(skyColor);
    this.scene.fog.density = fogDensity;
    this.ambientLight.color.setHex(ambColor);
    this.ambientLight.intensity = ambInt;
    this.dirLight.intensity = dirInt;
    if (this.groundMesh) {
      this.groundMesh.material.color.setHex(groundColor);
    }

    // Gatulyktor
    const lampIntensity = lightsOn ? (1.2 + this.arousal * 0.8) : 0;
    this.streetLights.forEach(sl => {
      sl.userData.pointLight.intensity = lampIntensity;
      if (sl.userData.lamp) {
        sl.userData.lamp.material.opacity = lightsOn ? 1 : 0.15;
      }
    });

    // Fönster + inre glöd
    const winOpacity  = lightsOn ? 0.9 : 0.05;
    const winColor    = lightsOn ? C.windowNight : C.windowDay;
    const glowInt     = lightsOn ? (0.6 + this.arousal * 0.5) : 0;
    Object.values(this.buildingMap).forEach(building => {
      building.traverse(child => {
        if (child.userData.isWindow) {
          child.material.opacity = winOpacity;
          child.material.color.setHex(winColor);
        }
      });
      // Inre fönsterglöd
      if (building.userData.windowLight) {
        building.userData.windowLight.intensity = glowInt;
        building.userData.windowLight.color.setHex(C.windowNight);
      }
      // Tornskimmer (meta-axiom)
      if (building.userData.glowLight) {
        building.userData.glowLight.intensity = lightsOn ? 3.0 : 0.8;
      }
    });
  }

  // ── Limbic ────────────────────────────────────────────────────
  async _pollLimbic() {
    try {
      const resp = await fetch('/api/limbic');
      if (!resp.ok) return;
      const data = await resp.json();
      this.arousal = data.arousal ?? 0.5;
      // Uppdatera ambient live (inte bara vid dag/natt-skifte)
      this.ambientLight.intensity = Math.max(
        this.ambientLight.intensity,
        this.arousal * 0.3,
      );
    } catch { /* ok */ }
  }

  // ── Pulsanimation (SSE-event) ─────────────────────────────────
  pulseBuilding(nodeId) {
    const b = this.buildingMap[nodeId];
    if (!b) return;
    const origScale = b.scale.clone();
    b.scale.setScalar(1.12);
    b.traverse(child => {
      if (child.isMesh && child.material.emissive) {
        child.material.emissive.setHex(C.neonActive);
        child.material.emissiveIntensity = 0.8;
      }
    });
    setTimeout(() => {
      b.scale.copy(origScale);
      b.traverse(child => {
        if (child.isMesh && child.material.emissive) {
          child.material.emissiveIntensity = 0;
        }
      });
    }, 1500);
  }

  // ── Orbit-kontroll (förenklad) ────────────────────────────────
  _setupControls() {
    // Enkel orbit: mouse-drag för rotation + scroll för zoom
    let isDragging = false;
    let lastX = 0, lastY = 0;
    let theta = Math.PI / 4;  // start-vinkel
    let phi   = Math.PI / 4;  // elevation
    let radius = 600;
    const target = new THREE.Vector3(0, 0, 0);

    const updateCamera = () => {
      const x = radius * Math.sin(phi) * Math.sin(theta);
      const y = radius * Math.cos(phi);
      const z = radius * Math.sin(phi) * Math.cos(theta);
      this.camera.position.set(x, y, z);
      this.camera.lookAt(target);
    };
    updateCamera();

    const el = this.renderer.domElement;

    el.addEventListener('mousedown', e => {
      isDragging = true;
      lastX = e.clientX; lastY = e.clientY;
    });
    el.addEventListener('mouseup', () => { isDragging = false; });
    el.addEventListener('mousemove', e => {
      if (!isDragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      theta -= dx * 0.005;
      phi    = Math.clamp(phi + dy * 0.005, 0.15, Math.PI / 2.2);
      lastX = e.clientX; lastY = e.clientY;
      updateCamera();
    });
    el.addEventListener('wheel', e => {
      radius = Math.clamp(radius + e.deltaY * 0.5, 80, 1800);
      updateCamera();
      e.preventDefault();
    }, { passive: false });

    this._updateCamera = updateCamera;
    this._orbitState = { theta, phi, radius, target, updateCamera };

    // Hoist för extern uppdatering
    this.container.addEventListener('mousemove', this._onMouseMove);
    this.container.addEventListener('click', this._onClick);
    window.addEventListener('resize', this._onResize);
  }

  _onMouseMove(e) {
    this.mouse.x =  (e.clientX / window.innerWidth)  * 2 - 1;
    this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    this._updateHoverState();
  }

  _onClick(e) {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.scene.children, true);
    for (const hit of hits) {
      let obj = hit.object;
      while (obj && !obj.userData.nodeId) obj = obj.parent;
      if (obj && obj.userData.nodeId) {
        // Anropa global showInspector med node-data
        if (typeof showInspector === 'function') {
          showInspector(obj.userData.nodeData);
        }
        return;
      }
    }
    if (typeof closeInspector === 'function') closeInspector();
  }

  _clearHoverBadge() {
    if (!this.hoveredNodeId) return;
    const current = this.buildingMap[this.hoveredNodeId];
    if (current) {
      current.scale.setScalar(1);
      if (this.hoverBadge && this.hoverBadge.parent === current) {
        current.remove(this.hoverBadge);
      }
    }
    this.hoverBadge = null;
    this.hoveredNodeId = '';
  }

  _applyHoverBadge(building) {
    const nodeId = String((building.userData || {}).nodeId || '').trim();
    if (!nodeId) return;
    if (this.hoveredNodeId === nodeId) return;

    this._clearHoverBadge();

    const nodeData = building.userData.nodeData || {};
    const color = this._domainColorHex(nodeData.group || 'unknown');
    const label = makeTextBadge(cityFormatNodeName(nodeId), color, {
      fontSize: 26,
      maxLen: 18,
      maxLines: 2,
      minWidth: 190,
      maxWidth: 380,
      opacity: 0.98,
    });
    label.position.set(0, building.userData.labelY || 18, 0);
    building.add(label);
    building.scale.setScalar(1.05);
    this.hoverBadge = label;
    this.hoveredNodeId = nodeId;
  }

  _updateHoverState() {
    if (!this.scene || !this.camera) return;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.scene.children, true);
    for (const hit of hits) {
      let obj = hit.object;
      while (obj && !obj.userData.nodeId) obj = obj.parent;
      if (obj && obj.userData.nodeId) {
        this._applyHoverBadge(obj);
        return;
      }
    }
    this._clearHoverBadge();
  }

  _onResize() {
    if (!this.camera || !this.renderer) return;
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  // ── Render-loop ───────────────────────────────────────────────
  _startLoop() {
    const animate = () => {
      this.animId = requestAnimationFrame(animate);
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  // ── Hjälp: domänfärg som hex-int ────────────────────────────
  _domainColorHex(domain) {
    // Återanvänder domainColor() från index.html om tillgänglig
    if (typeof domainColor === 'function') {
      return parseInt(domainColor(domain).replace('#', ''), 16);
    }
    return 0x4e9af1;
  }
}

// ── Globalt singleton ─────────────────────────────────────────────
window._nouseCityView = null;

window.cityViewInit = function(graphData) {
  if (window._nouseCityView) {
    window._nouseCityView.destroy();
  }
  const container = document.getElementById('city-container');
  if (!container) return;
  window._nouseCityView = new CityView(container);
  window._nouseCityView.init(graphData);
};

window.cityViewDestroy = function() {
  if (window._nouseCityView) {
    window._nouseCityView.destroy();
    window._nouseCityView = null;
  }
};

window.cityViewPulse = function(nodeId) {
  if (window._nouseCityView) {
    window._nouseCityView.pulseBuilding(nodeId);
  }
};
