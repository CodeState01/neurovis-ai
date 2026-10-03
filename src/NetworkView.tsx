import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
export type Snapshot = {
  sizes: number[];
  weights: number[][][];
  biases: number[][];
  activations: number[][];
  sums: number[][];
  probabilities: number[];
  epoch: number;
  running: boolean;
  metrics: { loss: number; accuracy: number };
  validation: { loss: number; accuracy: number };
  history: { epoch: number; loss: number }[];
  points: { x: number; y: number; label: number }[];
  sample: number[];
  grid: number[];
  kind: string;
};
export default function NetworkView({
  data,
  selected,
  onSelect,
  showEdges,
  rotate,
  resetView,
}: {
  data: Snapshot;
  selected: [number, number];
  onSelect: (s: [number, number]) => void;
  showEdges: boolean;
  rotate: boolean;
  resetView: number;
}) {
  const host = useRef<HTMLDivElement>(null),
    live = useRef({ data, selected, onSelect, showEdges, rotate }),
    cameraReset = useRef<() => void>(() => {});
  live.current = { data, selected, onSelect, showEdges, rotate };
  useEffect(() => {
    cameraReset.current();
  }, [resetView]);
  useEffect(() => {
    const container = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      container.textContent =
        "A visualização 3D precisa de WebGL. Ative a aceleração de hardware do navegador. O treinamento continua disponível.";
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene(),
      camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 7;
    controls.maxDistance = 32;
    controls.enablePan = false;
    controls.autoRotateSpeed = 0.5;
    cameraReset.current = () => {
      camera.position.set(1.8, 1.1, Math.max(12.5, 16 / camera.aspect));
      controls.target.set(0, 0, 0);
      controls.update();
    };
    cameraReset.current();
    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const light = new THREE.PointLight(0x8af4dc, 40);
    light.position.set(0, 5, 6);
    scene.add(light);
    const sizes = live.current.data.sizes,
      nodes: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>[][] =
        [];
    const geometry = new THREE.SphereGeometry(0.115, 14, 10),
      positions: THREE.Vector3[][] = [];
    sizes.forEach((n, l) => {
      nodes[l] = [];
      positions[l] = [];
      for (let j = 0; j < n; j++) {
        const columns = n > 24 ? 4 : n > 12 ? 2 : 1,
          row = Math.floor(j / columns),
          rows = Math.ceil(n / columns);
        const pos = new THREE.Vector3(
          (l - (sizes.length - 1) / 2) * 2.5,
          (row - (rows - 1) / 2) * 0.52,
          ((j % columns) - (columns - 1) / 2) * 0.72,
        );
        positions[l].push(pos);
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: 0x5ce2c1,
            emissive: 0x215b4b,
            roughness: 0.25,
            metalness: 0.2,
          }),
        );
        mesh.position.copy(pos);
        mesh.userData = { layer: l, index: j };
        scene.add(mesh);
        nodes[l].push(mesh);
      }
    });
    const links: {
      l: number;
      j: number;
      k: number;
      a: THREE.Vector3;
      b: THREE.Vector3;
    }[] = [];
    for (let l = 1; l < sizes.length; l++)
      for (let j = 0; j < sizes[l]; j++)
        for (let k = 0; k < sizes[l - 1]; k++)
          links.push({ l, j, k, a: positions[l - 1][k], b: positions[l][j] });
    const points = new Float32Array(links.length * 6),
      colors = new Float32Array(links.length * 6);
    links.forEach((e, i) => {
      e.a.toArray(points, i * 6);
      e.b.toArray(points, i * 6 + 3);
    });
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(points, 3));
    lineGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const lines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.55,
      }),
    );
    scene.add(lines);
    const pulsesGeometry = new THREE.BufferGeometry(),
      pulsePositions = new Float32Array(links.length * 3),
      pulseColors = new Float32Array(links.length * 3);
    pulsesGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(pulsePositions, 3),
    );
    pulsesGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(pulseColors, 3),
    );
    const pulses = new THREE.Points(
      pulsesGeometry,
      new THREE.PointsMaterial({
        size: 0.036,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
      }),
    );
    scene.add(pulses);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.2, 0.012, 8, 40),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    scene.add(ring);
    const raycaster = new THREE.Raycaster(),
      pointer = new THREE.Vector2();
    let down = [0, 0];
    const onDown = (e: PointerEvent) => {
      down = [e.clientX, e.clientY];
    };
    const onClick = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const r = container.getBoundingClientRect();
      pointer.set(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        (-(e.clientY - r.top) / r.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(nodes.flat())[0];
      if (hit) {
        const { layer, index } = hit.object.userData;
        live.current.onSelect([layer, index]);
      }
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onClick);
    const resize = new ResizeObserver(() => {
      const w = container.clientWidth,
        h = container.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      cameraReset.current();
    });
    resize.observe(container);
    const positive = new THREE.Color("#46d9bd"),
      negative = new THREE.Color("#ef976f"),
      color = new THREE.Color();
    let frame = 0;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const animate = (time: number) => {
      frame = requestAnimationFrame(animate);
      const {
        data: d,
        selected: s,
        showEdges: edges,
        rotate: rot,
      } = live.current;
      if (reduced) time = 0;
      controls.autoRotate = rot && !reduced;
      controls.update();
      lines.visible = edges;
      pulses.visible = edges;
      nodes.forEach((layer, l) =>
        layer.forEach((node, j) => {
          const value = d.activations[l]?.[j] || 0;
          node.material.color.copy(value >= 0 ? positive : negative);
          node.material.emissive
            .copy(node.material.color)
            .multiplyScalar(0.12 + Math.min(1, Math.abs(value)) * 0.7);
          node.scale.setScalar(0.75 + Math.min(1, Math.abs(value)) * 0.7);
        }),
      );
      const chosen = nodes[s[0]]?.[s[1]];
      ring.visible = !!chosen;
      if (chosen) {
        ring.position.copy(chosen.position);
        ring.quaternion.copy(camera.quaternion);
      }
      links.forEach((e, i) => {
        const w = d.weights[e.l - 1][e.j][e.k],
          strength = Math.min(1, Math.abs(w) / 2.5),
          active = e.l === s[0] && e.j === s[1];
        color
          .copy(w >= 0 ? positive : negative)
          .multiplyScalar(active ? 0.8 : 0.055 + strength * 0.24);
        color.toArray(colors, i * 6);
        color.toArray(colors, i * 6 + 3);
        const flow = (time * 0.00022 + e.l * 0.21 + (i % 17) * 0.037) % 1,
          contribution = Math.min(
            1,
            Math.abs(w * (d.activations[e.l - 1][e.k] || 0)),
          );
        pulsePositions[i * 3] = e.a.x + (e.b.x - e.a.x) * flow;
        pulsePositions[i * 3 + 1] = e.a.y + (e.b.y - e.a.y) * flow;
        pulsePositions[i * 3 + 2] = e.a.z + (e.b.z - e.a.z) * flow;
        color
          .copy(w >= 0 ? positive : negative)
          .multiplyScalar(contribution * (active ? 1.4 : 0.65));
        color.toArray(pulseColors, i * 3);
      });
      lineGeometry.attributes.color.needsUpdate = true;
      pulsesGeometry.attributes.position.needsUpdate = true;
      pulsesGeometry.attributes.color.needsUpdate = true;
      renderer.render(scene, camera);
    };
    frame = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onClick);
      geometry.dispose();
      nodes.flat().forEach((n) => n.material.dispose());
      lineGeometry.dispose();
      lines.material.dispose();
      pulsesGeometry.dispose();
      pulses.material.dispose();
      ring.geometry.dispose();
      ring.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [data.sizes.join(",")]);
  return (
    <div
      className="network-canvas"
      ref={host}
      role="img"
      aria-label="Rede neural em 3D. Arraste para girar, use a roda para aproximar e clique em um neurônio para inspecionar. Os mesmos valores estão disponíveis no painel de inspeção."
    />
  );
}
