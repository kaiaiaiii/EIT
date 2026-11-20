from __future__ import annotations
import json
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


EPS0 = 8.8541878128e-12


@dataclass
class Config:
    # Rohrgeometrie
    pipe_radius: float = 0.12       # m
    pipe_height: float = 0.500       # m
    grid_spacing: float = 0.005      # m; kleinere Werte -> genauer, aber deutlich langsamer

    # Elektroden: 3 Ringe mit je 16 Elektroden
    ring_heights: tuple = (0.47, 0.50, 0.53)
    electrodes_per_ring: int = 16

    # Elektrische Materialparameter (beliebig anpassbar)
    water_conductivity: float = 0.05       # S/m
    air_conductivity: float = 1.0e-14      # S/m
    water_relative_permittivity: float = 80.0
    air_relative_permittivity: float = 1.0006
    electrode_voltage: float = 1.0          # V, Amplitude

    # Frequenzbelegung: Elektrode k sendet f_start + k*f_step
    frequency_start: float = 1_000.0        # Hz
    frequency_step: float = 1_000.0         # Hz

    # Blasenmodell (bewusst einfaches kinematisches Modell)
    water_velocity: float = 0.20            # m/s nach oben
    bubble_rate: float = 4.0                # mittlere Blasenzahl pro Sekunde (Poisson)
    bubble_radius_min: float = 0.006        # m
    bubble_radius_max: float = 0.016        # m
    lateral_drift: float = 0.012            # m/s, Stärke des zufälligen Seitwärtsdrifts

    # Zeitintegration und elektrische Berechnung
    dt: float = 0.02                        # s
    total_time: float = 10.0                 # s
    electrical_every_n_steps: int = 1      # elektrische Lösung nur alle N Schritte
    random_seed: int = 7

    # Ausgabe
    output_dir: str = "output"
    live_visualization: bool = True


@dataclass
class Bubble:
    x: float
    y: float
    z: float
    radius: float
    vx: float
    vy: float


class BubbleSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.bubbles: list[Bubble] = []

    def step(self, dt: float):
        c = self.cfg
        # Neue Blasen werden als Poisson-Prozess am Rohreintritt erzeugt.
        n_new = self.rng.poisson(c.bubble_rate * dt)
        for _ in range(n_new):
            r = self.rng.uniform(c.bubble_radius_min, c.bubble_radius_max)
            max_center_r = max(0.0, c.pipe_radius - r - 0.004)
            rr = max_center_r * math.sqrt(self.rng.random())
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            self.bubbles.append(Bubble(
                x=rr * math.cos(theta),
                y=rr * math.sin(theta),
                z=r,
                radius=r,
                vx=self.rng.normal(0.0, c.lateral_drift * 0.35),
                vy=self.rng.normal(0.0, c.lateral_drift * 0.35),
            ))

        kept = []
        for b in self.bubbles:
            b.z += c.water_velocity * dt
            # Kleiner zufälliger Seitwärtsdrift (Random Walk).
            b.vx += self.rng.normal(0.0, c.lateral_drift * math.sqrt(dt) * 0.25)
            b.vy += self.rng.normal(0.0, c.lateral_drift * math.sqrt(dt) * 0.25)
            b.x += b.vx * dt
            b.y += b.vy * dt

            # Blasen bleiben innerhalb des Rohres.
            radial = math.hypot(b.x, b.y)
            max_r = c.pipe_radius - b.radius - 0.003
            if radial > max_r and radial > 0:
                scale = max_r / radial
                b.x *= scale
                b.y *= scale
                b.vx *= -0.4
                b.vy *= -0.4

            if b.z - b.radius <= c.pipe_height:
                kept.append(b)
        self.bubbles = kept


class CylindricalGrid:
    """Zellzentriertes kartesisches Finite-Volumen-Gitter innerhalb eines Zylinders."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        h = cfg.grid_spacing
        self.h = h
        self.x = np.arange(-cfg.pipe_radius, cfg.pipe_radius + 0.5*h, h)
        self.y = np.arange(-cfg.pipe_radius, cfg.pipe_radius + 0.5*h, h)
        self.z = np.arange(0.0, cfg.pipe_height + 0.5*h, h)
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        mask = X**2 + Y**2 <= cfg.pipe_radius**2
        self.shape = mask.shape
        self.X, self.Y, self.Z = X, Y, Z
        self.mask = mask
        self.flat_active = np.flatnonzero(mask.ravel())
        self.n = len(self.flat_active)
        self.full_to_active = np.full(mask.size, -1, dtype=np.int32)
        self.full_to_active[self.flat_active] = np.arange(self.n, dtype=np.int32)
        self.coords = np.column_stack((X.ravel()[self.flat_active],
                                       Y.ravel()[self.flat_active],
                                       Z.ravel()[self.flat_active]))

        # Alle Nachbarschaftskanten in +x, +y, +z Richtung.
        edges = []
        nx, ny, nz = self.shape
        for axis, stride in enumerate((ny*nz, nz, 1)):
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_a[axis] = slice(0, -1)
            sl_b[axis] = slice(1, None)
            a_full = np.arange(mask.size).reshape(self.shape)[tuple(sl_a)].ravel()
            b_full = np.arange(mask.size).reshape(self.shape)[tuple(sl_b)].ravel()
            valid = mask.ravel()[a_full] & mask.ravel()[b_full]
            aa = self.full_to_active[a_full[valid]]
            bb = self.full_to_active[b_full[valid]]
            edges.append((aa, bb))
        self.edge_i = np.concatenate([e[0] for e in edges])
        self.edge_j = np.concatenate([e[1] for e in edges])

        # Wandzellen: aktive Zellen mit mindestens einem Nachbarn außerhalb des Rohres.
        boundary = np.zeros(mask.shape, dtype=bool)
        for axis in range(2):  # nur Zylinder-Seitenwand, nicht Rohrenden
            for shift in (-1, 1):
                shifted = np.roll(mask, shift, axis=axis)
                boundary |= mask & ~shifted
        self.boundary_active = self.full_to_active[np.flatnonzero(boundary.ravel())]
        self.boundary_active = self.boundary_active[self.boundary_active >= 0]

    def build_electrodes(self):
        c = self.cfg
        result = []
        used = set()
        wall = self.coords[self.boundary_active]
        for ring, z0 in enumerate(c.ring_heights):
            for k in range(c.electrodes_per_ring):
                theta = 2.0 * math.pi * k / c.electrodes_per_ring
                target = np.array([c.pipe_radius * math.cos(theta),
                                   c.pipe_radius * math.sin(theta), z0])
                # Gewichtete Suche auf der Seitenwand; vermeidet Elektroden an den Rohrenden.
                dist = np.sum((wall - target)**2, axis=1)
                order = np.argsort(dist)
                chosen = None
                for oi in order:
                    idx = int(self.boundary_active[oi])
                    if idx not in used:
                        chosen = idx
                        break
                if chosen is None:
                    raise RuntimeError("Elektrodenpositionen kollidieren. Gitterweite verkleinern.")
                used.add(chosen)
                result.append(chosen)
        return np.asarray(result, dtype=np.int32)

    def material_map(self, bubbles: list[Bubble]):
        air = np.zeros(self.n, dtype=bool)
        if not bubbles:
            return air
        xyz = self.coords
        for b in bubbles:
            d2 = ((xyz[:, 0] - b.x)**2 +
                  (xyz[:, 1] - b.y)**2 +
                  (xyz[:, 2] - b.z)**2)
            air |= d2 <= b.radius**2
        return air


class ElectricalSolver:
    """
    Komplexwertige quasi-statische Leitfähigkeitsrechnung:
        div((sigma + j*omega*epsilon) grad(phi)) = 0

    Pro Frequenz wird eine Elektrode auf U0 gesetzt, alle anderen Elektroden
    liegen bei dieser Frequenz auf 0 V. Die komplexen Elektrodenströme bilden
    die Übertragungsmatrix. Die Diagonale (Eigenanregung) wird nicht ausgegeben.
    """

    def __init__(self, cfg: Config, grid: CylindricalGrid, electrode_nodes: np.ndarray):
        self.cfg = cfg
        self.grid = grid
        self.electrode_nodes = electrode_nodes
        self.n_electrodes = len(electrode_nodes)
        self.frequencies = (cfg.frequency_start +
                            cfg.frequency_step * np.arange(self.n_electrodes))
        self.fixed_nodes = electrode_nodes.copy()
        self.is_fixed = np.zeros(grid.n, dtype=bool)
        self.is_fixed[self.fixed_nodes] = True
        self.free_nodes = np.flatnonzero(~self.is_fixed)
        self.global_to_free = np.full(grid.n, -1, dtype=np.int32)
        self.global_to_free[self.free_nodes] = np.arange(len(self.free_nodes), dtype=np.int32)

    def _edge_admittance(self, air_mask, frequency):
        c = self.cfg
        omega = 2.0 * math.pi * frequency
        gamma_w = c.water_conductivity + 1j * omega * EPS0 * c.water_relative_permittivity
        gamma_a = c.air_conductivity + 1j * omega * EPS0 * c.air_relative_permittivity
        gamma = np.where(air_mask, gamma_a, gamma_w).astype(np.complex128)
        gi = gamma[self.grid.edge_i]
        gj = gamma[self.grid.edge_j]
        # Harmonisches Mittel für die Zellgrenze; A/V = gamma * Fläche / Abstand.
        gamma_face = 2.0 * gi * gj / (gi + gj + 1e-300)
        return gamma_face * self.grid.h

    def _assemble(self, edge_g, fixed_values):
        ei, ej = self.grid.edge_i, self.grid.edge_j
        mi = self.global_to_free[ei]
        mj = self.global_to_free[ej]
        fi = mi >= 0
        fj = mj >= 0

        rows, cols, vals = [], [], []
        rhs = np.zeros(len(self.free_nodes), dtype=np.complex128)

        # Freier Knoten gegen freien Knoten.
        both = fi & fj
        i = mi[both]; j = mj[both]; g = edge_g[both]
        rows.extend([i, j, i, j])
        cols.extend([i, j, j, i])
        vals.extend([g, g, -g, -g])

        # Freier Knoten gegen festes Elektrodenpotential.
        i_free_j_fixed = fi & ~fj
        if np.any(i_free_j_fixed):
            i = mi[i_free_j_fixed]
            g = edge_g[i_free_j_fixed]
            fixed_global = ej[i_free_j_fixed]
            rows.append(i); cols.append(i); vals.append(g)
            rhs += np.bincount(i, weights=np.real(g * fixed_values[fixed_global]),
                               minlength=len(rhs)) + 1j*np.bincount(
                                   i, weights=np.imag(g * fixed_values[fixed_global]),
                                   minlength=len(rhs))

        j_free_i_fixed = ~fi & fj
        if np.any(j_free_i_fixed):
            j = mj[j_free_i_fixed]
            g = edge_g[j_free_i_fixed]
            fixed_global = ei[j_free_i_fixed]
            rows.append(j); cols.append(j); vals.append(g)
            rhs += np.bincount(j, weights=np.real(g * fixed_values[fixed_global]),
                               minlength=len(rhs)) + 1j*np.bincount(
                                   j, weights=np.imag(g * fixed_values[fixed_global]),
                                   minlength=len(rhs))

        row = np.concatenate([np.asarray(v, dtype=np.int32) for v in rows])
        col = np.concatenate([np.asarray(v, dtype=np.int32) for v in cols])
        data = np.concatenate([np.asarray(v, dtype=np.complex128) for v in vals])
        A = sp.coo_matrix((data, (row, col)),
                          shape=(len(self.free_nodes), len(self.free_nodes))).tocsr()
        return A, rhs

    def solve(self, bubbles: list[Bubble]):
        air = self.grid.material_map(bubbles)
        H = np.zeros((self.n_electrodes, self.n_electrodes), dtype=np.complex128)
        H[:] = np.nan + 1j*np.nan

        # Die Materialverteilung bleibt über alle Frequenzen identisch,
        # die komplexe Admittivität hängt jedoch von omega ab.
        for tx, freq in enumerate(self.frequencies):
            edge_g = self._edge_admittance(air, freq)
            fixed_values = np.zeros(self.grid.n, dtype=np.complex128)
            fixed_values[self.electrode_nodes] = 0.0
            fixed_values[self.electrode_nodes[tx]] = self.cfg.electrode_voltage

            A, rhs = self._assemble(edge_g, fixed_values)
            phi_free = spsolve(A, rhs)
            phi = fixed_values.copy()
            phi[self.free_nodes] = phi_free

            # Elektrodenstrom: Summe der Ströme von der Elektrode in alle
            # benachbarten Fluidelemente. Positive Richtung in das Fluid.
            currents = np.zeros(self.n_electrodes, dtype=np.complex128)
            ei, ej = self.grid.edge_i, self.grid.edge_j
            for rx, node in enumerate(self.electrode_nodes):
                mask_i = ei == node
                mask_j = ej == node
                if np.any(mask_i):
                    currents[rx] += np.sum(edge_g[mask_i] * (phi[node] - phi[ej[mask_i]]))
                if np.any(mask_j):
                    currents[rx] += np.sum(edge_g[mask_j] * (phi[node] - phi[ei[mask_j]]))
            H[tx, :] = currents
            H[tx, tx] = np.nan + 1j*np.nan
        return H, air


def save_matrix_csv(path: Path, matrix: np.ndarray, frequencies: np.ndarray, quantity: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = matrix.shape[0]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TX/RX"] + [f"RX_{j+1}" for j in range(n)])
        for i in range(n):
            w.writerow([f"TX_{i+1}"] + [
                "" if not np.isfinite(matrix[i, j].real) else
                f"{matrix[i,j].real:.8e}{matrix[i,j].imag:+.8e}j"
                for j in range(n)
            ])
    # Begleitende Frequenzliste
    with path.with_name("frequencies_hz.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["electrode", "frequency_Hz"])
        for i, fr in enumerate(frequencies):
            w.writerow([i + 1, fr])


def save_amplitude_phase(outdir: Path, H: np.ndarray):
    amp = np.abs(H)
    phase = np.angle(H, deg=True)
    np.savetxt(outdir / "current_amplitude_A.csv", amp, delimiter=",", fmt="%.8e")
    np.savetxt(outdir / "current_phase_deg.csv", phase, delimiter=",", fmt="%.6f")


def visualize(cfg, bubbles, H, time_s, step, outdir):
    plt.ion()
    fig = plt.figure("Rohrstroemung und elektrische Uebertragung", figsize=(12, 6))
    fig.clf()

    ax = fig.add_subplot(121, projection="3d")
    # Zylinderoberfläche als Drahtgitter
    theta = np.linspace(0, 2*np.pi, 28)
    z = np.linspace(0, cfg.pipe_height, 12)
    TH, ZZ = np.meshgrid(theta, z)
    XX = cfg.pipe_radius * np.cos(TH)
    YY = cfg.pipe_radius * np.sin(TH)
    ax.plot_wireframe(XX, YY, ZZ, rstride=2, cstride=2, linewidth=0.35, alpha=0.25)

    # Elektrodenpunkte
    for zh in cfg.ring_heights:
        for k in range(cfg.electrodes_per_ring):
            th = 2*np.pi*k/cfg.electrodes_per_ring
            ax.scatter([cfg.pipe_radius*np.cos(th)],
                       [cfg.pipe_radius*np.sin(th)], [zh],
                       s=9, marker="s")

    # Blasen als Kugeln
    for b in bubbles:
        u = np.linspace(0, 2*np.pi, 10)
        v = np.linspace(0, np.pi, 7)
        xs = b.x + b.radius*np.outer(np.cos(u), np.sin(v))
        ys = b.y + b.radius*np.outer(np.sin(u), np.sin(v))
        zs = b.z + b.radius*np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.45, linewidth=0)
    ax.set_xlim(-cfg.pipe_radius, cfg.pipe_radius)
    ax.set_ylim(-cfg.pipe_radius, cfg.pipe_radius)
    ax.set_zlim(0, cfg.pipe_height)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Blasen im Rohr, t = {time_s:.2f} s")
    ax.set_box_aspect((1, 1, 5))

    ax2 = fig.add_subplot(122)
    amp = np.abs(H)
    # Normierung zur Darstellung; absolute Werte bleiben in CSV erhalten.
    finite = amp[np.isfinite(amp)]
    vmax = np.max(finite) if finite.size else 1.0
    im = ax2.imshow(amp, origin="lower", aspect="auto",
                    vmin=0, vmax=max(vmax, 1e-30))
    ax2.set_xlabel("Empfangselektrode")
    ax2.set_ylabel("Sendeelektrode")
    ax2.set_title("Betrag des Empfangsstroms [A]\nDiagonale = Eigenanregung ausgeblendet")
    fig.colorbar(im, ax=ax2, label="|I| [A]")
    fig.suptitle(f"3 Ringe × 16 Elektroden = 48 Elektroden | Schritt {step}")
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
    fig.savefig(outdir / "latest_result.png", dpi=130)
    return fig


def main():
    cfg = Config()
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Erzeuge 3D-Gitter ...")
    grid = CylindricalGrid(cfg)
    electrodes = grid.build_electrodes()
    print(f"Gitterzellen im Rohr: {grid.n:,}")
    print(f"Elektroden: {len(electrodes)}")

    bubbles = BubbleSystem(cfg)
    solver = ElectricalSolver(cfg, grid, electrodes)

    # Konfiguration dokumentieren
    (outdir / "run_config.json").write_text(
        json.dumps(cfg.__dict__, indent=2), encoding="utf-8"
    )

    n_steps = int(round(cfg.total_time / cfg.dt))
    H = np.full((len(electrodes), len(electrodes)), np.nan + 1j*np.nan)
    fig = None
    start = time.time()

    for step in range(n_steps + 1):
        t = step * cfg.dt
        if step > 0:
            bubbles.step(cfg.dt)

        do_electric = (step % cfg.electrical_every_n_steps == 0) or step == n_steps
        if do_electric:
            print(f"[{step:4d}/{n_steps}] t={t:.2f} s | {len(bubbles.bubbles)} Blasen | loese 48 Frequenzen ...")
            H, air = solver.solve(bubbles.bubbles)
            save_matrix_csv(outdir / f"transfer_current_step_{step:05d}.csv",
                            H, solver.frequencies, "current")
            save_amplitude_phase(outdir, H)
            if cfg.live_visualization:
                if fig is not None:
                    plt.close(fig)
                fig = visualize(cfg, bubbles.bubbles, H, t, step, outdir)

    elapsed = time.time() - start
    print(f"Fertig. Laufzeit: {elapsed:.1f} s")
    print(f"Ergebnisse liegen in: {outdir.resolve()}")
    if cfg.live_visualization:
        print("Schliesse das Plotfenster, um das Programm zu beenden.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
