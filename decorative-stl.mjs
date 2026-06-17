import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";


const HERO_SCREEN = window.matchMedia("(min-width: 1201px)");
const UPRIGHT_ROTATION_X = -Math.PI / 2;

function initHeroSTL({
  containerId,
  modelPath,
  color,
  targetSize = 2.05,
  speed = 0.0032,
  initialY = 0,
  cameraZ = 4.2
}) {
  const container = document.getElementById(containerId);
  if (!container || !HERO_SCREEN.matches) return null;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 0.03, cameraZ);

  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    powerPreference: "low-power"
  });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const pivot = new THREE.Group();
  pivot.rotation.y = initialY;
  scene.add(pivot);

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color),
    roughness: 0.52,
    metalness: 0.04,
    side: THREE.DoubleSide
  });

  const hemiLight = new THREE.HemisphereLight(0xffffff, 0xc9e7e5, 1.45);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.05);
  keyLight.position.set(2.6, 3.4, 4.8);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xdff8f6, 0.75);
  fillLight.position.set(-3.0, 1.5, 2.4);
  scene.add(fillLight);

  const loader = new STLLoader();
  loader.load(
    modelPath,
    (geometry) => {
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();

      const bounds = geometry.boundingBox;
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      bounds.getCenter(center);
      bounds.getSize(size);
      geometry.translate(-center.x, -center.y, -center.z);

      const maxDimension = Math.max(size.x, size.y, size.z) || 1;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.scale.setScalar(targetSize / maxDimension);

      // STL exports lie mostly along the depth Z axis. Rotate the mesh once
      // so the trachea appears upright, then spin only the parent pivot about Y.
      mesh.rotation.x = UPRIGHT_ROTATION_X;
      pivot.add(mesh);
    },
    undefined,
    (error) => {
      console.warn(`Could not load hero STL: ${modelPath}`, error);
    }
  );

  function resize() {
    const rect = container.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }

  let animationFrame = 0;
  let previousTime = performance.now();

  function animate(now) {
    animationFrame = window.requestAnimationFrame(animate);
    const delta = Math.min(40, now - previousTime);
    previousTime = now;
    pivot.rotation.y += speed * (delta / 16.67);
    renderer.render(scene, camera);
  }

  resize();
  animationFrame = window.requestAnimationFrame(animate);

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);

  return {
    destroy() {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.dispose();
      material.dispose();
      container.replaceChildren();
    }
  };
}

let heroViewers = [];

function setupHeroModels() {
  if (heroViewers.length) return;

  heroViewers = [
    initHeroSTL({
      containerId: "hero-stl-right",
      modelPath: "assets/figures/airway_closed_validation_dilation.stl",
      color: "#0b3558",
      targetSize: 2.08,
      speed: 0.0032,
      initialY: 0.38
    })
  ].filter(Boolean);
}

function teardownHeroModels() {
  heroViewers.forEach((viewer) => viewer.destroy());
  heroViewers = [];
}

setupHeroModels();

HERO_SCREEN.addEventListener("change", (event) => {
  if (event.matches) {
    setupHeroModels();
  } else {
    teardownHeroModels();
  }
});
