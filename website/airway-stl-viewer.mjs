import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const VIEWER_BACKGROUND = "#eef4f7";
const MESH_COLOR = "#0b3558";
const UPRIGHT_ROTATION_X = -Math.PI / 2;
const TARGET_MODEL_SIZE = 3.05;

const container = document.getElementById("airway-three-viewer");
const fallback = document.getElementById("model-fallback");
const description = document.getElementById("model-description");
const tabs = Array.from(document.querySelectorAll(".model-tab"));

if (container && tabs.length) {
  initAirwayViewer();
}

function initAirwayViewer() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(VIEWER_BACKGROUND);

  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 1000);
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance"
  });
  renderer.setClearColor(VIEWER_BACKGROUND, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const pivot = new THREE.Group();
  scene.add(pivot);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.minDistance = 1.7;
  controls.maxDistance = 8.5;

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(MESH_COLOR),
    roughness: 0.52,
    metalness: 0.04,
    side: THREE.DoubleSide
  });

  scene.add(new THREE.HemisphereLight(0xffffff, 0xcddde6, 1.35));

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
  keyLight.position.set(3.2, 4.4, 5.2);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xe3f1f6, 0.85);
  fillLight.position.set(-4.2, 1.8, 3.0);
  scene.add(fillLight);

  const loader = new STLLoader();
  let currentMesh = null;
  let loadToken = 0;
  let animationFrame = 0;
  let previousTime = performance.now();

  function showFallback(show) {
    if (fallback) fallback.hidden = !show;
  }

  function resize() {
    const rect = container.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    renderer.render(scene, camera);
  }

  function clearCurrentMesh() {
    if (!currentMesh) return;
    pivot.remove(currentMesh);
    currentMesh.geometry.dispose();
    currentMesh = null;
  }

  function frameMesh(mesh, size) {
    const maxDimension = Math.max(size.x, size.y, size.z) || 1;
    mesh.scale.setScalar(TARGET_MODEL_SIZE / maxDimension);

    const distance = TARGET_MODEL_SIZE * 2.05;
    camera.position.set(0, TARGET_MODEL_SIZE * 0.06, distance);
    camera.near = Math.max(0.01, distance / 100);
    camera.far = distance * 100;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.minDistance = TARGET_MODEL_SIZE * 0.55;
    controls.maxDistance = TARGET_MODEL_SIZE * 3.4;
    controls.update();
  }

  function loadModel(modelPath) {
    const token = ++loadToken;
    clearCurrentMesh();
    showFallback(false);

    loader.load(
      modelPath,
      (geometry) => {
        if (token !== loadToken) {
          geometry.dispose();
          return;
        }

        geometry.computeVertexNormals();
        geometry.computeBoundingBox();

        const bounds = geometry.boundingBox;
        const center = new THREE.Vector3();
        const size = new THREE.Vector3();
        bounds.getCenter(center);
        bounds.getSize(size);
        geometry.translate(-center.x, -center.y, -center.z);

        const mesh = new THREE.Mesh(geometry, material);
        // The STL's main airway axis is authored along Z; map it to vertical Y.
        mesh.rotation.x = UPRIGHT_ROTATION_X;

        pivot.rotation.y = 0.35;
        pivot.add(mesh);
        currentMesh = mesh;
        frameMesh(mesh, size);
        showFallback(false);
      },
      undefined,
      () => {
        if (token !== loadToken) return;
        clearCurrentMesh();
        showFallback(true);
      }
    );
  }

  function activateTab(tab) {
    tabs.forEach((button) => {
      const isActive = button === tab;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    });

    if (description && tab.dataset.description) {
      description.textContent = tab.dataset.description;
    }

    if (tab.dataset.model) {
      loadModel(tab.dataset.model);
    }
  }

  function animate(now) {
    animationFrame = window.requestAnimationFrame(animate);
    const delta = Math.min(40, now - previousTime);
    previousTime = now;
    pivot.rotation.y += 0.0021 * (delta / 16.67);
    controls.update();
    renderer.render(scene, camera);
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab));
  });

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();
  activateTab(tabs.find((tab) => tab.classList.contains("active")) || tabs[0]);
  animationFrame = window.requestAnimationFrame(animate);

  window.addEventListener("pagehide", () => {
    window.cancelAnimationFrame(animationFrame);
    resizeObserver.disconnect();
    clearCurrentMesh();
    material.dispose();
    renderer.dispose();
  }, { once: true });
}
