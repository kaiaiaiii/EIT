"""
Wasser strömt in einem runden Rohr von unten nach oben, darin steigen runde
Luftblasen mit (bewusst unphysikalisch: Blasen = mitgetragene Kugeln).

NEU: Elektroden an der Rohrwand
  * 2 Ringe à 16 Elektroden, gleichmäßig über den Umfang verteilt,
    bündig mit der Rohrinnenwand, Ringe bei Mitte -5 cm und Mitte +5 cm
  * jede der 32 Elektroden hat ihre eigene Frequenz
  * berechnet wird, welches Signal (von welcher Elektrode, mit welcher Frequenz)
    bei jeder anderen Elektrode ankommt

Elektrisches Modell (quasistatisch, Wasser rein ohmsch):
  * Rohr wird in ein 3D-Gitter zerlegt (Finite Volumen, 1 cm Zellen).
  * Leitfähigkeit des Wassers SIGMA_W; Blasen sind Isolatoren. Der Gasanteil
    alpha pro Zelle senkt die Leitfähigkeit nach Maxwell:
        sigma = SIGMA_W * (1 - alpha) / (1 + alpha / 2)
  * Jede Elektrode wird mit Spannung V_i(t) = V * sin(2 pi f_i t) getrieben;
    alle Elektroden sind gleichzeitig Sender und Empfänger. Da das Netzwerk
    linear ist, gilt Superposition: Der Strom an Elektrode j ist
        I_j(t) = sum_i Y_ji * V_i(t)
    mit der 32x32-Admittanzmatrix Y (aus dem Gitter berechnet). Das von
    Elektrode i bei Elektrode j ankommende Signal hat also die Frequenz f_i und
    die Amplitude G_ji = -Y_ji * V.
  * Die Empfangselektrode trennt die Signale per FFT (Frequenzmultiplex).

Benötigt: numpy, scipy, matplotlib
"""
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import EllipseCollection
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

# ======================= Parameter (SI-Einheiten) =======================
# --- Rohr & Strömung ---
D = 0.20                   # Rohrdurchmesser (m)
R = D / 2                  # Rohrradius (m)
H = 1.00                   # Rohrlänge (m)
DT = 0.03                  # Zeitschritt (s)
U_MAX = 0.30               # Wassergeschwindigkeit in Rohrmitte (m/s)

# --- Blasen ---
SPAWN_RATE = 15.0          # mittlere Zahl neuer Blasen pro Sekunde
R_MEDIAN, R_SIGMA = 0.004, 0.35   # Blasenradius: lognormalverteilt (Median 4 mm)
WOBBLE = 0.012             # seitliches Zittern (m/sqrt(s)), 0 = aus
SLIP = 0.0                 # Auftrieb relativ zum Wasser (m/s), 0 = mitgetragen
N_TRACER = 400             # Wasser-Tracer (nur zur Visualisierung)

# --- Elektroden ---
N_PER_RING = 16                          # Elektroden pro Ring
RING_Y = (H / 2 - 0.05, H/2, H / 2 + 0.05)    # Ringhöhen: 5 cm unter / über der Mitte
N_EL = N_PER_RING * len(RING_Y)          # 48 Elektroden; Index = Ring * 16 + k
EL_WIDTH_FRAC = 0.6                      # Elektrodenbreite als Anteil des Umfangs-Abstands
F0, DF = 20e3, 1e3                       # Frequenz der Elektrode k: F0 + k * DF
V_AMP = 1.0                              # Treiberspannung (V)
RX = 0                                   # Empfangselektrode für die Spektrum-Anzeige

# --- Elektrisches Gitter ---
SIGMA_W = 0.05             # Leitfähigkeit Wasser (S/m), ca. 500 uS/cm
GRID_H = 0.01              # Zellgröße (m)
WIN_HALF = 0.25            # Rechenfenster: Mitte +- 25 cm (Rest ist elektrisch unwichtig)
Y_EVERY = 1                # Admittanzmatrix alle n Zeitschritte neu berechnen

# --- Messung ---
FS = 250e3                 # Abtastrate des Empfängers (Hz)
N_SAMP = 250               # Fensterlänge (1 ms -> Frequenzraster 1 kHz, alle f_k orthogonal)

HEATMAP_PERCENT = 1.0      # Farbskala der Kopplungsänderung (+- %)

rng = np.random.default_rng(42)

FREQ = F0 + DF * np.arange(N_EL)
EL_ANGLE = 2 * np.pi * (np.arange(N_EL) % N_PER_RING) / N_PER_RING
EL_Y = np.array(RING_Y)[np.arange(N_EL) // N_PER_RING]


# ======================= Strömung & Blasen =======================
def flow(rho):
    """Aufwärtsgeschwindigkeit des Wassers, parabolisch über den Rohrradius
    (rho = Abstand von der Rohrachse). An der Wand nicht ganz 0."""
    return U_MAX * (0.25 + 0.75 * (1 - (rho / R) ** 2))


def random_in_disk(n, radius):
    """Gleichverteilte Punkte (x, z) in einer Kreisscheibe."""
    rho = radius * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return rho * np.cos(phi), rho * np.sin(phi)


bub_pos = np.empty((0, 3))     # Spalten: x, y (Höhe), z (Tiefe)
bub_r = np.empty(0)
tx, tz = random_in_disk(N_TRACER, R)
tracer = np.column_stack([tx, rng.uniform(0, H, N_TRACER), tz])


def step():
    global bub_pos, bub_r, tracer

    # Wasser-Tracer
    rho_t = np.hypot(tracer[:, 0], tracer[:, 2])
    tracer[:, 1] += flow(rho_t) * DT
    out = tracer[:, 1] > H
    if out.any():
        tracer[out, 1] -= H
        tracer[out, 0], tracer[out, 2] = random_in_disk(out.sum(), R)

    # Blasen bewegen
    if len(bub_r):
        rho = np.hypot(bub_pos[:, 0], bub_pos[:, 2])
        bub_pos[:, 1] += (flow(rho) + SLIP) * DT
        bub_pos[:, 0] += rng.normal(0, WOBBLE * np.sqrt(DT), len(bub_r))
        bub_pos[:, 2] += rng.normal(0, WOBBLE * np.sqrt(DT), len(bub_r))

        # Blase nicht in die Rohrwand: radial zurückschieben
        rho = np.hypot(bub_pos[:, 0], bub_pos[:, 2])
        rho_max = R - bub_r
        scale = np.where(rho > rho_max, rho_max / np.maximum(rho, 1e-12), 1.0)
        bub_pos[:, 0] *= scale
        bub_pos[:, 2] *= scale

        keep = bub_pos[:, 1] - bub_r < H
        bub_pos, bub_r = bub_pos[keep], bub_r[keep]

    # Neue Blasen (Poisson-Prozess) am unteren Rand
    n_new = rng.poisson(SPAWN_RATE * DT)
    if n_new:
        r = np.minimum(R_MEDIAN * rng.lognormal(0, R_SIGMA, n_new), 0.4 * R)
        xz = np.array([[a[0] for a in random_in_disk(1, R - ri)] for ri in r])
        new = np.column_stack([xz[:, 0], -r, xz[:, 1]])
        bub_pos = np.vstack([bub_pos, new])
        bub_r = np.concatenate([bub_r, r])


# ======================= Elektrisches Gitter =======================
h = GRID_H
nx = nz = int(round(D / h))
ny = int(round(2 * WIN_HALF / h)) + 1
xc = -R + (np.arange(nx) + 0.5) * h
zc = -R + (np.arange(nz) + 0.5) * h
yc = H / 2 + (np.arange(ny) - (ny - 1) / 2) * h      # Ringe liegen exakt auf Zellmitten

rho2d = np.hypot(xc[:, None], zc[None, :])
disk = rho2d <= R                                      # Rohrquerschnitt (Treppenstufen-Rand)
inside = np.broadcast_to(disk[:, None, :], (nx, ny, nz))
cell_id = -np.ones((nx, ny, nz), dtype=int)
cell_id[inside] = np.arange(inside.sum())
n_cells = int(inside.sum())

# --- Elektroden: Zellen der äußersten Schicht (bündig mit der Wand) ---
shell = disk & (rho2d > R - h)
ang2d = np.arctan2(zc[None, :], xc[:, None])
half_w = EL_WIDTH_FRAC * np.pi / N_PER_RING
eid3d = -np.ones((nx, ny, nz), dtype=int)
for k in range(N_EL):
    iy = int(np.argmin(np.abs(yc - EL_Y[k])))
    dth = np.angle(np.exp(1j * (ang2d - EL_ANGLE[k])))
    sel = shell & (np.abs(dth) <= half_w)
    if not sel.any():                                  # Sicherheitsnetz: nächste Wandzelle nehmen
        sel = shell & (np.abs(dth) == np.abs(dth)[shell].min())
    layer = eid3d[:, iy, :]
    layer[sel] = k
eid_cell = eid3d[inside]
n_el_cells = np.bincount(eid_cell[eid_cell >= 0], minlength=N_EL)

int_of_cell = -np.ones(n_cells, dtype=int)
is_int = eid_cell < 0
n_int = int(is_int.sum())
int_of_cell[is_int] = np.arange(n_int)

# --- Zellflächen (Nachbarpaare) ---
_pa, _pb = [], []
for axis in range(3):
    sa = [slice(None)] * 3
    sb = [slice(None)] * 3
    sa[axis], sb[axis] = slice(0, -1), slice(1, None)
    a, b = cell_id[tuple(sa)], cell_id[tuple(sb)]
    m = (a >= 0) & (b >= 0)
    _pa.append(a[m])
    _pb.append(b[m])
_pa, _pb = np.concatenate(_pa), np.concatenate(_pb)
_ea, _eb = eid_cell[_pa], eid_cell[_pb]

m_ii = (_ea < 0) & (_eb < 0)                        # innen - innen
CA, CB = _pa[m_ii], _pb[m_ii]
IA, IB = int_of_cell[CA], int_of_cell[CB]

m_ie = (_ea < 0) != (_eb < 0)                       # innen - Elektrode (Elektrode - Elektrode: kein Pfad)
_ci = np.where(_ea[m_ie] < 0, _pa[m_ie], _pb[m_ie])
CE = np.where(_ea[m_ie] < 0, _pb[m_ie], _pa[m_ie])  # Elektrodenzelle
CI = _ci                                            # innere Zelle
II = int_of_cell[CI]
KE = eid_cell[CE]


def gas_alpha(pos, r):
    """Gasanteil pro Gitterzelle (Cloud-in-Cell-Verteilung des Blasenvolumens)."""
    alpha = np.zeros((nx, ny, nz))
    if len(r):
        vol = 4 / 3 * np.pi * r**3 / h**3
        f = [(pos[:, 0] - xc[0]) / h, (pos[:, 1] - yc[0]) / h, (pos[:, 2] - zc[0]) / h]
        i0 = [np.floor(c).astype(int) for c in f]
        w1 = [c - i for c, i in zip(f, i0)]
        for dx, dy, dz in product((0, 1), repeat=3):
            idx = [i0[0] + dx, i0[1] + dy, i0[2] + dz]
            w = vol * np.prod([w1[a] if d else 1 - w1[a] for a, d in enumerate((dx, dy, dz))], axis=0)
            ok = (idx[0] >= 0) & (idx[0] < nx) & (idx[1] >= 0) & (idx[1] < ny) & (idx[2] >= 0) & (idx[2] < nz)
            np.add.at(alpha, (idx[0][ok], idx[1][ok], idx[2][ok]), w[ok])
    return np.clip(alpha, 0, 0.6)[inside]


def face_conductances(alpha):
    """Leitwerte aller Zellflächen (innen-innen, innen-Elektrode) für einen Gasanteil-Vektor."""
    sig = SIGMA_W * (1 - alpha) / (1 + alpha / 2)

    def face(a, b):                                   # harmonisches Mittel zweier Zellen
        return 2 * sig[a] * sig[b] / (sig[a] + sig[b]) * h

    return face(CA, CB), face(CI, CE)


def admittance_matrix(alpha):
    """Exakte 32x32-Admittanzmatrix Y (S): Strom aus Elektrode j in das Wasser bei
    Elektrodenspannungen V:  I = Y @ V.  Off-Diagonale sind <= 0.
    Löst das Gleichungssystem komplett (ca. 0.7 s) -> wird nur für die Basislinie
    (Rohr ohne Blasen) und zur Kontrolle benutzt. Gibt zusätzlich die Potentiale X zurück."""
    g_ii, g_ie = face_conductances(alpha)
    diag = (np.bincount(IA, g_ii, n_int) + np.bincount(IB, g_ii, n_int)
            + np.bincount(II, g_ie, n_int))
    ar = np.arange(n_int)
    L = coo_matrix((np.concatenate([diag, -g_ii, -g_ii]),
                    (np.concatenate([ar, IA, IB]), np.concatenate([ar, IB, IA]))),
                   shape=(n_int, n_int)).tocsc()
    B = np.zeros((n_int, N_EL))
    np.add.at(B, (II, KE), g_ie)
    X = splu(L).solve(B)                              # Potentiale im Wasser pro Elektrode
    return np.diag(B.sum(0)) - B.T @ X, X             # Schur-Komplement


# Schnelle Variante für die Animation: Störungsrechnung 1. Ordnung.
# Es gilt exakt  Y_ji = sum_Flächen g_f * dphi_f(i) * dphi_f(j)  (dphi_f(i) = Spannungsabfall
# über die Fläche, wenn nur Elektrode i auf 1 V liegt). Ändern die Blasen die Leitwerte um
# dg_f, ist in 1. Ordnung  dY = D^T diag(dg) D  - bei Gasanteilen von Promille sehr genau.
_Y0, _X0 = admittance_matrix(np.zeros(n_cells))
_onehot = np.eye(N_EL)[KE]
_D = np.vstack([_X0[IA] - _X0[IB], _X0[II] - _onehot])
_g0 = np.concatenate(face_conductances(np.zeros(n_cells)))


def admittance_fast(alpha):
    dg = np.concatenate(face_conductances(alpha)) - _g0
    return _Y0 + _D.T @ (dg[:, None] * _D)


# ======================= Signale =======================
T_AXIS = np.arange(N_SAMP) / FS
_BINS = np.rint(FREQ / (FS / N_SAMP)).astype(int)


def empfangssignal(Y, j, t=T_AXIS):
    """Zeitsignal (Strom, A) an Elektrode j: Summe der von allen anderen Elektroden i
    ankommenden Signale  G_ji * V * sin(2 pi f_i t)."""
    G = -Y[j].copy()
    G[j] = 0.0                                        # eigenes Treibersignal wird ausgeblendet
    return (G[:, None] * V_AMP * np.sin(2 * np.pi * FREQ[:, None] * t[None, :])).sum(0)


def demodulieren(sig):
    """Amplituden der 32 Frequenzen im Empfangssignal (FFT, Frequenzmultiplex)."""
    return 2 * np.abs(np.fft.rfft(sig))[_BINS] / N_SAMP


# ======================= Darstellung =======================
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.6])
ax = fig.add_subplot(gs[:, 0])
ax_h = fig.add_subplot(gs[0, 1])
ax_s = fig.add_subplot(gs[1, 1])

# --- Rohr ---
ax.set_xlim(-R, R)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.set_facecolor("#1f6fb5")
ax.set_xticks([])
ax.set_yticks([])
for side in ("left", "right"):
    ax.spines[side].set_linewidth(3)
ax.set_xlabel("Ø 20 cm", fontsize=9)
ax.set_ylabel("1 m", fontsize=9)
tracer_plot = ax.scatter(tracer[:, 0], tracer[:, 1], s=2, c="white", alpha=0.25)

el_x = R * np.cos(EL_ANGLE)
el_depth = (R * np.sin(EL_ANGLE) / R + 1) / 2
el_rgba = np.zeros((N_EL, 4))
el_rgba[:, :3] = (1.0, 0.65, 0.1)
el_rgba[:, 3] = 0.25 + 0.75 * el_depth
el_rgba[RX] = (1, 0.1, 0.1, 1)
ax.scatter(el_x, EL_Y, s=55, marker="s", c=el_rgba, edgecolors="k", linewidths=0.6, zorder=5)
for y in RING_Y:
    ax.axhline(y, color="orange", lw=0.6, alpha=0.5)
bubbles = None

# --- Heatmap ---
im = ax_h.imshow(np.full((N_EL, N_EL), np.nan), cmap="RdBu_r",
                 vmin=-HEATMAP_PERCENT, vmax=HEATMAP_PERCENT, origin="lower")
ax_h.axhline(N_PER_RING - 0.5, color="k", lw=0.8)
ax_h.axvline(N_PER_RING - 0.5, color="k", lw=0.8)
ticks = list(range(0, N_EL, 4))
ax_h.set_xticks(ticks)
ax_h.set_yticks(ticks)
ax_h.set_xlabel("sendende Elektrode i  (0-15 unterer Ring, 16-31 oberer Ring)")
ax_h.set_ylabel("empfangende Elektrode j")
fig.colorbar(im, ax=ax_h, label="Änderung der Kopplung ΔG/G₀ (%)")

# --- Spektrum ---
ring_color = ["#4c78a8" if k < N_PER_RING else "#f58518" for k in range(N_EL)]
bars = ax_s.bar(FREQ / 1e3, np.ones(N_EL), width=0.6, color=ring_color)
ax_s.set_yscale("log")
ax_s.set_xlabel("Frequenz (kHz) = sendende Elektrode")
ax_s.set_ylabel("Amplitude (µA)")
ax_s.set_ylim(10, 500)
ax_s.set_title(f"Bei Elektrode {RX} ankommende Signale", fontsize=10)
sec = ax_s.secondary_xaxis("top", functions=(lambda f: f - F0 / 1e3, lambda i: i + F0 / 1e3))
sec.set_xlabel("Elektrode i")

# --- Basislinie (ohne Blasen) ---
Y0 = _Y0
G0 = -Y0
offdiag = ~np.eye(N_EL, dtype=bool)
state = {"n": 0, "Y": Y0}


def update(_frame):
    global bubbles
    step()
    state["n"] += 1
    tracer_plot.set_offsets(tracer[:, :2])

    if bubbles is not None:
        bubbles.remove()
        bubbles = None
    if len(bub_r):
        order = np.argsort(bub_pos[:, 2])
        pos, r = bub_pos[order], bub_r[order]
        depth = (pos[:, 2] / R + 1) / 2
        face = np.column_stack([np.ones((len(r), 3)), 0.12 + 0.28 * depth])
        edge = np.column_stack([np.ones((len(r), 3)), 0.35 + 0.65 * depth])
        d = 2 * r
        bubbles = EllipseCollection(d, d, np.zeros_like(d), units="x", offsets=pos[:, :2],
                                    offset_transform=ax.transData,
                                    facecolors=face, edgecolors=edge, linewidths=1.0)
        ax.add_collection(bubbles)

    if state["n"] % Y_EVERY == 1:
        alpha = gas_alpha(bub_pos, bub_r)
        Y = admittance_fast(alpha)
        state["Y"] = Y
        dG = np.where(offdiag, 100 * (-Y - G0) / np.where(offdiag, G0, 1), np.nan)
        im.set_data(dG)
        amps = demodulieren(empfangssignal(Y, RX)) * 1e6        # µA
        amps[RX] = np.nan
        for b, a in zip(bars, amps):
            b.set_height(a if np.isfinite(a) else 1e-12)
        gas = alpha.sum() * h**3 / (np.pi * R**2 * ny * h)
        ax_h.set_title(f"Kopplung G_ji = -Y_ji relativ zum Rohr ohne Blasen   (mittlerer Gasanteil {100 * gas:.3f} %)",
                       fontsize=10)
    return tracer_plot,


# Vorlauf, damit das Rohr beim Start nicht leer ist
for _ in range(int(H / (U_MAX * DT))):
    step()

anim = FuncAnimation(fig, update, interval=int(DT * 1000), blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()
    # Als GIF speichern (statt plt.show()):
    # anim.save("blasen.gif", writer="pillow", fps=30, frames=300)