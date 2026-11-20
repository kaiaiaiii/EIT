"""
Wasser strömt in einem runden Rohr von unten nach oben, darin steigen runde
Luftblasen mit (bewusst unphysikalisch: Blasen = mitgetragene Kugeln).

Elektroden an der Rohrwand
  * 3 Ringe à 16 Elektroden (48 insgesamt), gleichmäßig über den Umfang verteilt, bündig
    mit der Rohrinnenwand; Ringe bei Mitte -5 cm, genau in der Mitte und Mitte +5 cm
    (Index 0-15 unten, 16-31 Mitte, 32-47 oben)
  * jede der 48 Elektroden hat ihre eigene Frequenz f_k = F0 + k * DF  (20 ... 67 kHz)
  * alle Elektroden senden gleichzeitig und empfangen gleichzeitig;
    das eigene Signal empfängt eine Elektrode nicht

Elektrisches Feldmodell (quasistatisch):
  * Das Rohr wird in ein 3D-Gitter zerlegt (Finite Volumen, 5 mm Zellen). In jeder
    Zelle gilt div(sigma * grad(phi)) = 0; die Rohrwand isoliert, die Elektrodenzellen
    in der Wand liegen auf dem Elektrodenpotential.
  * Blasen sind Isolatoren. Der Gasanteil alpha pro Zelle senkt die Leitfähigkeit
    nach Maxwell:  sigma = SIGMA_W * (1 - alpha) / (1 + alpha / 2)
  * Aus dem Gitter folgt die 48x48-Admittanzmatrix Y_f des Wassers (Elektrodenpotential
    -> Strom ins Wasser). Sie wird bei jeder neuen Blasenverteilung komplett neu
    gelöst (ca. 2-3 s, im Hintergrund-Thread).
  * Kontaktimpedanz: Zwischen Anschluss und Wasser liegt je Elektrode
        Z_c(f) = R_C + 1 / (j 2 pi f C)       (Kontaktwiderstand + Doppelschicht)
    Damit ist die Kopplung frequenzabhängig. Für die Linie bei f_i gilt
        H(f_i) = Y_f (1 + Z_c(f_i) Y_f)^-1       (Spalte i, Diagonale = 0)
    -> bei Elektrode j kommt von Elektrode i die Amplitude |H_ji| * V mit Phase arg(H_ji).
  * Empfangssignal: I_j(t) = sum_{i != j} |H_ji| V sin(2 pi f_i t + arg H_ji); jede
    Empfangselektrode trennt die Signale per FFT (Frequenzmultiplex).

Zwei Messmodi (Knopf "Modus"):
  * Paper-Modus (Standard), wie im Paper (Pechler/Sack/Pöschel): Stromanregung 5 mA / 5 kHz zwischen
    Nachbarelektroden (GND = Elektrode i, Strom bei i+1 im selben Ring), alle anderen Elektroden
    floatend und hochohmig, gemessen werden die Spannungen aller Elektroden. Auswertung per DFT bei
    5 kHz, Normierung auf die Spannung an der Stromelektrode (Gl. 4-6), Referenz = Rohr ohne Blasen,
    Differenzen dU und dphi (Gl. 7-8). Kontaktimpedanz Z_c liegt in Reihe zu jeder Elektrode: Floatende
    Elektroden sehen keinen Spannungsabfall darüber, die GND- und die Stromelektrode schon.
    Das Paper nennt keine Zahlenwerte für Z_c -> R_C und C sind frei wählbare Platzhalter.
  * Parallel-Modus: alle Elektroden senden gleichzeitig mit eigener Frequenz (siehe oben).

Im Fenster lassen sich Spawnrate, Blasengröße, Strömung, Leitfähigkeit,
Kontaktimpedanz u. a. mit Reglern einstellen.

Benötigt: numpy, scipy, matplotlib
"""
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

# ======================= Startwerte (SI-Einheiten) =======================
# Alles mit "Regler" lässt sich im Fenster live ändern.
D = 0.12                   # Rohrdurchmesser (m)
R = D / 2                  # Rohrradius (m)
H = 0.50                   # Rohrlänge (m)
DT = 0.03                  # Zeitschritt (s)
U_MAX = 0.30               # Wassergeschwindigkeit in Rohrmitte (m/s)           [Regler]
SPAWN_RATE = 15.0          # mittlere Zahl neuer Blasen pro Sekunde              [Regler]
R_MEDIAN, R_SIGMA = 0.004, 0.35   # Blasenradius lognormal, Median 4 mm          [Regler: Median]
WOBBLE = 0.012             # seitliches Zittern (m/sqrt(s))                      [Regler]
SLIP = 0.0                 # Auftrieb relativ zum Wasser (m/s), 0 = mitgetragen  [Regler]
N_TRACER = 400             # Wasser-Tracer (nur Visualisierung)

N_PER_RING = 16                          # Elektroden pro Ring
RING_Y = (H / 2 - 0.05, H / 2, H / 2 + 0.05)   # Ringhöhen: 5 cm unter der Mitte, Mitte, 5 cm darüber
N_EL = N_PER_RING * len(RING_Y)          # 48 Elektroden; Index = Ring * 16 + k
EL_WIDTH_FRAC = 0.6                      # Elektrodenbreite als Anteil des Umfangs-Abstands
F0, DF = 20e3, 1e3                       # Frequenz der Elektrode k: F0 + k * DF
V_AMP = 1.0                              # Treiberspannung (V)
RX = 0                                   # Empfangselektrode für die Spektrum-Anzeige [Regler]
SRC = N_PER_RING + 5                     # Anregung im Querschnittsbild: Paper-Modus GND = Elektrode SRC, Strom bei SRC+1; Parallel-Modus Sender SRC [Regler]
MODE_PAPER = True                        # Startmodus: True = Messschema des Papers, False = Parallel (Multifrequenz)
F_EXC, I_EXC = 5e3, 5e-3                 # Paper: 5 kHz, 5 mA
FS_P, M_P = 125e3, 125                   # Paper: 125 kS/s; Auswertefenster hier 1 ms (5 Perioden, rauschfrei) statt 75 ms
AUTO_EVERY = 10                          # Auto-Durchlauf der Anregung: alle n Zeitschritte weiter

SIGMA_W = 0.05             # Leitfähigkeit Wasser (S/m), ca. 500 uS/cm            [Regler]
CONTACT_R = 20.0           # Kontaktwiderstand je Elektrode (Ohm), Platzhalter     [Regler]
CONTACT_C = 500.0          # Kontakt-/Doppelschichtkapazität je Elektrode (nF), Platzhalter [Regler]
HEATMAP_PERCENT = 0.1      # Farbskala der Änderungskarten (+- %)                 [Regler]

GRID_H = 0.005             # Gitterzelle (m)
WIN_HALF = 0.15            # Rechenfenster: Mitte +- 15 cm (muss <= H/2 sein)
FS = 250e3                 # Abtastrate des Empfängers (Hz)
N_SAMP = 250               # Fensterlänge (1 ms -> Frequenzraster 1 kHz, alle f_k orthogonal)

rng = np.random.default_rng(42)

FREQ = F0 + DF * np.arange(N_EL)
EL_ANGLE = 2 * np.pi * (np.arange(N_EL) % N_PER_RING) / N_PER_RING
EL_Y = np.array(RING_Y)[np.arange(N_EL) // N_PER_RING]

# Aktuelle Parameter (werden von den Reglern verändert)
P = dict(u_max=U_MAX, spawn=SPAWN_RATE, r_med=R_MEDIAN, wobble=WOBBLE, slip=SLIP,
         sigma=SIGMA_W, rc=CONTACT_R, cc=CONTACT_C, rx=RX, hm=HEATMAP_PERCENT, src=SRC)


# ======================= Strömung & Blasen =======================
def flow(rho):
    """Aufwärtsgeschwindigkeit des Wassers, parabolisch über den Rohrradius."""
    return P["u_max"] * (0.25 + 0.75 * (1 - (rho / R) ** 2))


def random_in_disk(n, radius):
    """Gleichverteilte Punkte (x, z) in einer Kreisscheibe."""
    rho = radius * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return rho * np.cos(phi), rho * np.sin(phi)


bub_pos = np.empty((0, 3))     # Spalten: x, y (Höhe), z (Tiefe)
bub_r = np.empty(0)
tx, tz = random_in_disk(N_TRACER, R)
tracer = np.column_stack([tx, rng.uniform(0, H, N_TRACER), tz])


def clear_bubbles():
    global bub_pos, bub_r
    bub_pos, bub_r = np.empty((0, 3)), np.empty(0)


def step():
    global bub_pos, bub_r, tracer

    rho_t = np.hypot(tracer[:, 0], tracer[:, 2])
    tracer[:, 1] += flow(rho_t) * DT
    out = tracer[:, 1] > H
    if out.any():
        tracer[out, 1] -= H
        tracer[out, 0], tracer[out, 2] = random_in_disk(out.sum(), R)

    if len(bub_r):
        rho = np.hypot(bub_pos[:, 0], bub_pos[:, 2])
        bub_pos[:, 1] += (flow(rho) + P["slip"]) * DT
        bub_pos[:, 0] += rng.normal(0, P["wobble"] * np.sqrt(DT), len(bub_r))
        bub_pos[:, 2] += rng.normal(0, P["wobble"] * np.sqrt(DT), len(bub_r))

        rho = np.hypot(bub_pos[:, 0], bub_pos[:, 2])          # Blase nicht in die Wand
        rho_max = R - bub_r
        scale = np.where(rho > rho_max, rho_max / np.maximum(rho, 1e-12), 1.0)
        bub_pos[:, 0] *= scale
        bub_pos[:, 2] *= scale

        keep = bub_pos[:, 1] - bub_r < H                      # oben raus -> löschen
        bub_pos, bub_r = bub_pos[keep], bub_r[keep]

    n_new = rng.poisson(P["spawn"] * DT)                      # Poisson-Prozess am unteren Rand
    if n_new:
        r = np.minimum(P["r_med"] * rng.lognormal(0, R_SIGMA, n_new), 0.4 * R)
        xz = np.array([[a[0] for a in random_in_disk(1, R - ri)] for ri in r])
        bub_pos = np.vstack([bub_pos, np.column_stack([xz[:, 0], -r, xz[:, 1]])])
        bub_r = np.concatenate([bub_r, r])


# ======================= Elektrisches Gitter =======================
h = GRID_H
nx = nz = int(round(D / h))
assert WIN_HALF <= H / 2
ny = 2 * int(round(WIN_HALF / h)) + 1
xc = -R + (np.arange(nx) + 0.5) * h
zc = -R + (np.arange(nz) + 0.5) * h
yc = H / 2 + (np.arange(ny) - (ny - 1) / 2) * h      # Ringe liegen exakt auf Zellmitten

rho2d = np.hypot(xc[:, None], zc[None, :])
disk = rho2d <= R                                      # Rohrquerschnitt (Treppenstufen-Rand)
inside = np.broadcast_to(disk[:, None, :], (nx, ny, nz))
cell_id = -np.ones((nx, ny, nz), dtype=int)
cell_id[inside] = np.arange(inside.sum())
n_cells = int(inside.sum())

# Elektroden = Zellen der äußersten Schicht (bündig mit der Wand)
shell = disk & (rho2d > R - h)
ang2d = np.arctan2(zc[None, :], xc[:, None])
half_w = EL_WIDTH_FRAC * np.pi / N_PER_RING
eid3d = -np.ones((nx, ny, nz), dtype=int)
for k in range(N_EL):
    iy = int(np.argmin(np.abs(yc - EL_Y[k])))
    dth = np.angle(np.exp(1j * (ang2d - EL_ANGLE[k])))
    sel = shell & (np.abs(dth) <= half_w)
    if not sel.any():
        sel = shell & (np.abs(dth) == np.abs(dth)[shell].min())
    layer = eid3d[:, iy, :]
    layer[sel] = k
eid_cell = eid3d[inside]
n_el_cells = np.bincount(eid_cell[eid_cell >= 0], minlength=N_EL)

int_of_cell = -np.ones(n_cells, dtype=int)
is_int = eid_cell < 0
n_int = int(is_int.sum())
int_of_cell[is_int] = np.arange(n_int)

_pa, _pb = [], []
for axis in range(3):                                  # alle Zellflächen (Nachbarpaare)
    sa, sb = [slice(None)] * 3, [slice(None)] * 3
    sa[axis], sb[axis] = slice(0, -1), slice(1, None)
    a, b = cell_id[tuple(sa)], cell_id[tuple(sb)]
    m = (a >= 0) & (b >= 0)
    _pa.append(a[m])
    _pb.append(b[m])
_pa, _pb = np.concatenate(_pa), np.concatenate(_pb)
_ea, _eb = eid_cell[_pa], eid_cell[_pb]

m_ii = (_ea < 0) & (_eb < 0)                           # innen - innen
CA, CB = _pa[m_ii], _pb[m_ii]
IA, IB = int_of_cell[CA], int_of_cell[CB]
m_ie = (_ea < 0) != (_eb < 0)                          # innen - Elektrode (Elektrode - Elektrode: kein Pfad)
CI = np.where(_ea[m_ie] < 0, _pa[m_ie], _pb[m_ie])
CE = np.where(_ea[m_ie] < 0, _pb[m_ie], _pa[m_ie])
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
            ok = ((idx[0] >= 0) & (idx[0] < nx) & (idx[1] >= 0) & (idx[1] < ny)
                  & (idx[2] >= 0) & (idx[2] < nz))
            np.add.at(alpha, (idx[0][ok], idx[1][ok], idx[2][ok]), w[ok])
    return np.clip(alpha, 0, 0.6)[inside]


def admittance_matrix(alpha):
    """Gibt (Y, X) zurück: Y = 48x48-Admittanzmatrix des Wassers für Leitfähigkeit 1 S/m (Einheit: S pro S/m = m),
    X = Potentiale aller inneren Zellen, wenn jeweils eine Elektrode auf 1 V und alle anderen auf 0 V liegen.
    Für das echte Wasser mit SIGMA_W multiplizieren (alle Leitwerte skalieren linear).
    alpha = Gasanteil pro Gitterzelle. Strom aus Elektrode j ins Wasser = sum_i Y_ji * u_i."""
    sig = (1 - alpha) / (1 + alpha / 2)                # Maxwell, relativ zum Wasser

    def face(a, b):                                    # harmonisches Mittel zweier Zellen
        return 2 * sig[a] * sig[b] / (sig[a] + sig[b]) * h

    g_ii, g_ie = face(CA, CB), face(CI, CE)
    diag = (np.bincount(IA, g_ii, n_int) + np.bincount(IB, g_ii, n_int)
            + np.bincount(II, g_ie, n_int))
    ar = np.arange(n_int)
    L = coo_matrix((np.concatenate([diag, -g_ii, -g_ii]),
                    (np.concatenate([ar, IA, IB]), np.concatenate([ar, IB, IA]))),
                   shape=(n_int, n_int)).tocsc()
    B = np.zeros((n_int, N_EL))
    np.add.at(B, (II, KE), g_ie)
    X = splu(L).solve(B)                               # Potentiale im Wasser je Elektrode (1 V)
    return np.diag(B.sum(0)) - B.T @ X, X              # Y (Schur-Komplement) und Potentiale X der Zellen


# ======================= Signale =======================
T_AXIS = np.arange(N_SAMP) / FS
_BINS = np.rint(FREQ / (FS / N_SAMP)).astype(int)


def kontaktimpedanz(f):
    """Z_c(f) = R_C + 1/(j 2 pi f C) je Elektrode (komplex, Ohm)."""
    return P["rc"] + 1 / (1j * 2 * np.pi * f * P["cc"] * 1e-9)


def uebertragung(Yf):
    """Komplexe Kopplung H[j, i] (A/V): Strom, der bei Elektrode j ankommt, wenn Elektrode i
    mit ihrer Frequenz f_i und 1 V sendet. Das Netzwerk aus Kontaktimpedanzen und Wasser
    (Y_f) wird bei f_i ausgewertet; das eigene Signal (j = i) wird nicht empfangen."""
    lam, Q = np.linalg.eigh(Yf)                        # Y_f symmetrisch -> Y_f (1 + Z Y_f)^-1 = Q diag(l/(1+Z l)) Q^T
    Hm = np.zeros((N_EL, N_EL), dtype=complex)
    for i in range(N_EL):
        Zc = kontaktimpedanz(FREQ[i])
        Hm[:, i] = -(Q @ (lam / (1 + Zc * lam) * Q[i, :]))   # Strom fließt ins Wasser -> Empfang = -I
        Hm[i, i] = 0.0
    return V_AMP * Hm


def empfangssignale(Hm, t=T_AXIS):
    """Zeitsignale (A) an ALLEN Elektroden gleichzeitig, Form (48, N_SAMP):
    I_j(t) = sum_{i != j} |H_ji| sin(2 pi f_i t + arg H_ji)."""
    ph = np.exp(1j * 2 * np.pi * FREQ[None, :, None] * t[None, None, :])
    return np.imag(Hm[:, :, None] * ph).sum(axis=1)


def demodulieren(sig):
    """FFT je Empfänger: komplexe Amplitude A[j, i] der Linie bei f_i (Betrag = Amplitude, Winkel = Phase)."""
    return 2j * np.fft.rfft(sig, axis=1)[:, _BINS] / N_SAMP


def messung(Yunit):
    """Kompletter Messvorgang für eine (Einheits-)Admittanzmatrix -> demodulierte Kopplungen."""
    return demodulieren(empfangssignale(uebertragung(P["sigma"] * Yunit)))


# ---- Messschema des Papers: Stromanregung, floatende Elektroden, Spannungsmessung ----
_AR = np.arange(N_EL)
CFG_G = _AR.copy()                                                    # GND-Elektrode der Konfiguration n = n
CFG_S = (_AR // N_PER_RING) * N_PER_RING + (_AR % N_PER_RING + 1) % N_PER_RING   # Stromelektrode: Nachbar i+1 im Ring
_E_P = np.exp(1j * 2 * np.pi * F_EXC * np.arange(M_P) / FS_P)


def paper_potentiale(Yf):
    """Elektrodenpotentiale u0 (V, relativ zur GND-Elektrode) für alle Konfigurationen, Form (N_EL, N_EL):
    Strom +I_EXC in Elektrode CFG_S, -I_EXC in CFG_G, alle anderen floatend (Strom 0).
    Löst Y_f u = I (Y_f ist singulär -> Pseudoinverse, Potential nur bis auf Konstante bestimmt)."""
    I = np.zeros((N_EL, N_EL))
    I[_AR, CFG_S] = I_EXC
    I[_AR, CFG_G] = -I_EXC
    u0 = I @ np.linalg.pinv(Yf, hermitian=True)
    return u0 - u0[_AR, CFG_G][:, None], I


def paper_messung(Yf):
    """Messung nach dem Paper. Zeile n = Anregung (GND = Elektrode n, Strom bei CFG_S[n]).
    Klemmenspannung V = u + Z_c * I: Floatende Elektroden (I = 0) messen die Spannung im Wasser;
    die GND-Elektrode liegt an 0 V, d. h. alle Spannungen enthalten zusätzlich +Z_c*I_EXC (Abfall über
    dem GND-Kontakt), die Stromelektrode +2*Z_c*I_EXC. Dann Zeitreihen -> DFT bei 5 kHz (Gl. 3),
    Normierung auf die Stromelektrode (Gl. 4-6). Rückgabe: U~ (N,N), phi (N,N) in rad."""
    Zc = kontaktimpedanz(F_EXC)
    u0, I = paper_potentiale(Yf)
    V = u0 + Zc * I_EXC + Zc * I                 # komplexe Amplitude je Elektrode (Phasor, sin-Bezug)
    V[_AR, CFG_G] = 0.0                          # GND-Elektrode: exakt 0 V
    series = np.imag(V[:, :, None] * _E_P)       # U_ij(t_m), Form (N, N, M_P)
    Xd = (series * np.conj(_E_P)).sum(-1)        # DFT bei F_EXC
    ratio = Xd / Xd[_AR, CFG_S][:, None]
    ph = np.angle(ratio)
    ph = ph - ph[_AR, CFG_S][:, None]
    ph[_AR, CFG_G] = 0.0
    return np.abs(ratio), ph


# ======================= Darstellung =======================
fig = plt.figure(figsize=(18, 9), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[0.7, 1.55, 1.9, 1.5])
ax = fig.add_subplot(gs[:, 0])
ax_x = fig.add_subplot(gs[0, 1])
ax_h = fig.add_subplot(gs[0, 2])
ax_s = fig.add_subplot(gs[1, 1:3])                     # Spektrum (Parallel-Modus)
ax_p = fig.add_subplot(gs[1, 1:3])                     # Phasenkarte (Paper-Modus)
ax_c = fig.add_subplot(gs[:, 3])
ax_c.axis("off")

# --- Rohr ---
ax.set_xlim(-R, R)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.set_facecolor("#1f6fb5")
ax.set_xticks([])
ax.set_yticks([])
for side in ("left", "right"):
    ax.spines[side].set_linewidth(3)
ax.set_xlabel("Ø 12 cm", fontsize=9)
ax.set_ylabel("50 cm", fontsize=9)
tracer_plot = ax.scatter(tracer[:, 0], tracer[:, 1], s=2, c="white", alpha=0.25)

el_x = R * np.cos(EL_ANGLE)
_depth = (np.sin(EL_ANGLE) + 1) / 2


def el_colors():
    c = np.zeros((N_EL, 4))
    c[:, :3] = (1.0, 0.65, 0.1)
    c[:, 3] = 0.25 + 0.75 * _depth
    if not state["paper"]:
        c[int(P["rx"])] = (1, 0.1, 0.1, 1)
    else:                                              # GND rot, Stromelektrode blau (wie Fig. 1 des Papers)
        n = int(P["src"])
        c[CFG_G[n]] = (1, 0.1, 0.1, 1)
        c[CFG_S[n]] = (0.1, 0.3, 1, 1)
    return c


state = {"fut": None, "paused": False, "auto": True, "paper": MODE_PAPER, "n": 0}
el_sc = ax.scatter(el_x, EL_Y, s=55, marker="s", c=el_colors(), edgecolors="k", linewidths=0.6, zorder=5)
for y in RING_Y:
    ax.axhline(y, color="orange", lw=0.6, alpha=0.5)
bubbles = None

# --- Heatmap (Paper: dU, Parallel: Änderung in %) ---
im = ax_h.imshow(np.full((N_EL, N_EL), np.nan), cmap="RdBu_r",
                 vmin=-P["hm"], vmax=P["hm"], origin="lower")
for _r in range(1, len(RING_Y)):
    ax_h.axhline(_r * N_PER_RING - 0.5, color="k", lw=0.8)
    ax_h.axvline(_r * N_PER_RING - 0.5, color="k", lw=0.8)
ax_h.set_xticks(range(0, N_EL, 4))
ax_h.set_yticks(range(0, N_EL, 4))
cb_h = fig.colorbar(im, ax=ax_h)

# --- Phasenkarte (Paper-Modus) ---
im_p = ax_p.imshow(np.full((N_EL, N_EL), np.nan), cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
for _r in range(1, len(RING_Y)):
    ax_p.axhline(_r * N_PER_RING - 0.5, color="k", lw=0.8)
    ax_p.axvline(_r * N_PER_RING - 0.5, color="k", lw=0.8)
ax_p.set_xticks(range(0, N_EL, 4))
ax_p.set_yticks(range(0, N_EL, 4))
cb_p = fig.colorbar(im_p, ax=ax_p, label="Δφ (mrad)")

# --- Spektrum (Parallel-Modus) ---
ring_color = [("#4c78a8", "#54a24b", "#f58518")[k // N_PER_RING] for k in range(N_EL)]
bars = ax_s.bar(FREQ / 1e3, np.ones(N_EL), width=0.6, color=ring_color)
ax_s.set_yscale("log")
ax_s.set_xlabel("Frequenz (kHz) = sendende Elektrode")
ax_s.set_ylabel("Amplitude (µA)")
sec = ax_s.secondary_xaxis("top", functions=(lambda f: f - F0 / 1e3, lambda i: i + F0 / 1e3))
sec.set_xlabel("Elektrode i")

# --- Querschnitt durch den mittleren Ring (entspricht Fig. 1 des Papers) ---
IY_MID = int(np.argmin(np.abs(yc - H / 2)))            # Gitterschicht des mittleren Rings
MID = np.arange(N_PER_RING, 2 * N_PER_RING)            # Elektroden 16-31
SL_IDS = cell_id[:, IY_MID, :]                         # Zell-IDs der Schnittebene
SL_MASK = SL_IDS >= 0
_sl_cells = SL_IDS[SL_MASK]
SL_IS_INT = is_int[_sl_cells]
SL_INT_IDX = int_of_cell[_sl_cells[SL_IS_INT]]
SL_EL_IDX = eid_cell[_sl_cells[~SL_IS_INT]]

ax_x.set_xlim(-1.3 * R, 1.3 * R)
ax_x.set_ylim(-1.3 * R, 1.3 * R)
ax_x.set_aspect("equal")
ax_x.axis("off")
ax_x.add_patch(Circle((0, 0), R, fill=False, lw=2, color="k", zorder=3))
xs_el, zs_el = R * np.cos(EL_ANGLE[MID]), R * np.sin(EL_ANGLE[MID])
x_el_sc = ax_x.scatter(xs_el, zs_el, s=60, marker="s", c="0.55", edgecolors="k", zorder=6)
for k in MID:
    ax_x.text(1.14 * R * np.cos(EL_ANGLE[k]), 1.14 * R * np.sin(EL_ANGLE[k]), str(k),
              fontsize=7, ha="center", va="center")
x_title = ax_x.set_title("", fontsize=9)
ax_x.text(0, -1.30 * R, "farbig: Änderung durch Blasen   durchgezogen: mit Blasen   gestrichelt: Rohr ohne Blasen\n"
          "orange: Blasen in der Schnittebene   rot: GND bzw. Sender   blau: Stromelektrode",
          fontsize=7, ha="center", va="center")
phi_sm = ScalarMappable(norm=Normalize(-P["hm"], P["hm"]), cmap="RdBu_r")
cb_x = fig.colorbar(phi_sm, ax=ax_x, shrink=0.6, label="Δ(Φ / Φ_Stromelektrode) in %  (Feldform, normiert wie Gl. 4)")
feld_artists = []
slice_bubbles = None


def _slice_values(X, u):
    """Potential in der Schnittebene aus den Elektrodenpotentialen u (harmonische Fortsetzung X @ u)."""
    vals = np.empty(len(_sl_cells), dtype=u.dtype)
    vals[SL_IS_INT] = X[SL_INT_IDX] @ u
    vals[~SL_IS_INT] = u[SL_EL_IDX]
    out = np.full(SL_IDS.shape, np.nan, dtype=u.dtype)
    out[SL_MASK] = vals
    return out


def potential_slice(Yunit, X, n):
    """Potentialfeld in der Schnittebene und Potential der Stromelektrode/des Senders.
    Paper-Modus: Strom in Elektrode CFG_S[n], GND = CFG_G[n], übrige floatend; Feld relativ zur GND-Elektrode
    (die Kontaktimpedanz verschiebt nur alle Potentiale um eine Konstante, ändert die Feldform nicht).
    Parallel-Modus: Elektrode n sendet bei ihrer Frequenz, alle anderen auf 0 V (inkl. Kontaktimpedanz), Betrag."""
    Yf = P["sigma"] * Yunit
    if state["paper"]:
        u0, _ = paper_potentiale(Yf)
        return _slice_values(X, u0[n]), float(u0[n, CFG_S[n]])
    e = np.zeros(N_EL, dtype=complex)
    e[n] = V_AMP
    u = np.linalg.solve(np.eye(N_EL) + kontaktimpedanz(FREQ[n]) * Yf, e)
    return np.abs(_slice_values(X, u)), float(np.abs(u[n]))


def draw_feldbild():
    """Potentialfeld im Querschnitt neu zeichnen: Änderung der Feldform gegenüber dem leeren Rohr (Farbe;
    beide Felder auf das Potential an der Stromelektrode normiert, sonst würde die Änderung des Gesamt-
    widerstands alles überdecken) und Äquipotentiallinien (Bruchteile des Stromelektroden-Potentials)."""
    global feld_artists
    for a in feld_artists:
        a.remove()
    n = int(P["src"])
    p1, umax = potential_slice(state["Y"], state["X"], n)
    p0, umax0 = potential_slice(Y0u, X0u, n)
    hm = P["hm"]
    d = 100 * (p1 / umax - p0 / umax0)               # Feldform, normiert auf das Potential der Stromelektrode (wie Gl. 4)
    phi_sm.set_clim(-hm, hm)
    cf = ax_x.contourf(xc, zc, d.T, levels=np.linspace(-hm, hm, 21), cmap="RdBu_r", extend="both",
                       alpha=0.9, zorder=1)
    fr = (np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]) if state["paper"]
          else np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]))
    c0 = ax_x.contour(xc, zc, p0.T, levels=fr * umax0, colors="0.35", linestyles="dashed", linewidths=0.8, zorder=2)
    c1 = ax_x.contour(xc, zc, p1.T, levels=fr * umax, colors="k", linewidths=1.2, zorder=2)
    feld_artists = [cf, c0, c1]
    colors = np.tile((0.55, 0.55, 0.55, 1.0), (N_PER_RING, 1))
    if state["paper"]:
        g, s_ = CFG_G[n], CFG_S[n]
        for k, col in ((g, (1, 0.1, 0.1, 1)), (s_, (0.1, 0.3, 1, 1))):
            if k in MID:
                colors[k - N_PER_RING] = col
        where = f"Anregung {n}: GND = Elektrode {g}, Strom bei {s_}"
        if g not in MID:
            where += " (anderer Ring)"
    else:
        if n in MID:
            colors[n - N_PER_RING] = (1, 0.1, 0.1, 1)
        where = f"Sender: Elektrode {n}, alle anderen 0 V" + ("" if n in MID else " (anderer Ring)")
    x_el_sc.set_facecolors(colors)
    el_sc.set_facecolors(el_colors())
    x_title.set_text(f"Querschnitt Mittelring (y = {100 * H / 2:.0f} cm)\n{where}")


# --- Messung / Hintergrund-Feldlösung ---
Y0u, X0u = admittance_matrix(np.zeros(n_cells))        # Basislinie: Rohr ohne Blasen
_pool = ThreadPoolExecutor(max_workers=1)
state.update({"Y": Y0u, "X": X0u, "gas": 0.0, "t_field": time.time()})


def feldloesung(alpha):
    """Hintergrund-Thread: Feldmodell für die aktuelle Blasenverteilung lösen."""
    Y, X = admittance_matrix(alpha)
    return Y, X, alpha.sum() * h**3 / (np.pi * R**2 * ny * h)


def apply_mode():
    """Beschriftung und Sichtbarkeit je nach Messmodus."""
    paper = state["paper"]
    ax_s.set_visible(not paper)
    sec.set_visible(not paper)
    ax_p.set_visible(paper)
    cb_p.ax.set_visible(paper)
    if paper:
        ax_h.set_xlabel("Elektrode j (gemessene Spannung)")
        ax_h.set_ylabel("Anregung i (GND = i, Strom bei i+1)")
        cb_h.set_label("ΔŨ = Ũ − Ũ_ref  (% der Spannung an der Stromelektrode)")
        ax_p.set_xlabel("Elektrode j (gemessene Spannung)")
        ax_p.set_ylabel("Anregung i")
    else:
        ax_h.set_xlabel("sendende Elektrode i  (0-15 unterer, 16-31 mittlerer, 32-47 oberer Ring)")
        ax_h.set_ylabel("empfangende Elektrode j")
        cb_h.set_label("Änderung der Amplitude (%)")
    el_sc.set_facecolors(el_colors())
    b_mode.label.set_text("Modus: Paper" if paper else "Modus: Parallel")


def refresh():
    """Messung mit aktuellem Feld und aktuellen Reglerwerten neu auswerten und anzeigen."""
    off = ~np.eye(N_EL, dtype=bool)
    if state["paper"]:
        Yf, Yf0 = P["sigma"] * state["Y"], P["sigma"] * Y0u
        U, ph = paper_messung(Yf)
        U0, ph0 = paper_messung(Yf0)                   # Referenzmessung (Gl. 7-8): Rohr ohne Blasen
        im.set_data(100 * (U - U0))
        im.set_clim(-P["hm"], P["hm"])
        dph = 1e3 * (ph - ph0)
        m = max(float(np.percentile(np.abs(dph), 99)), 1e-3)
        im_p.set_data(dph)
        im_p.set_clim(-m, m)
        ax_h.set_title(f"Referenzdifferenz ΔŨ (Gl. 7), Gasanteil {100 * state['gas']:.3f} %", fontsize=10)
        ax_p.set_title("Phasendifferenz Δφ (Gl. 8)", fontsize=10)
    else:
        A, A0 = np.abs(messung(state["Y"])), np.abs(messung(Y0u))
        im.set_data(np.where(off, 100 * (A - A0) / np.where(off, A0, 1), np.nan))
        im.set_clim(-P["hm"], P["hm"])
        rx = int(P["rx"])
        amps = A[rx] * 1e6
        others = np.delete(amps, rx)
        for k, (b, a) in enumerate(zip(bars, amps)):
            b.set_height(1e-12 if k == rx else a)
        ax_s.set_ylim(others.min() / 1.5, others.max() * 1.5)
        ax_s.set_title(f"Bei Elektrode {rx} ankommende Signale (Frequenz = Sender, eigenes Signal fehlt)", fontsize=10)
        ax_h.set_title(f"Änderung der Empfangsamplitude durch Blasen (Gasanteil {100 * state['gas']:.3f} %)", fontsize=10)
    draw_feldbild()
    fig.canvas.draw_idle()


def update(_frame):
    global bubbles, slice_bubbles
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

    # Blasen in der Schnittebene des Mittelrings: in JEDEM Zeitschritt neu
    if slice_bubbles is not None:
        slice_bubbles.remove()
        slice_bubbles = None
    dy = bub_pos[:, 1] - H / 2
    cut = np.abs(dy) < bub_r
    if cut.any():
        rs = np.sqrt(bub_r[cut] ** 2 - dy[cut] ** 2)
        slice_bubbles = EllipseCollection(2 * rs, 2 * rs, np.zeros_like(rs), units="x",
                                          offsets=bub_pos[cut][:, [0, 2]], offset_transform=ax_x.transData,
                                          facecolors=(1.0, 0.75, 0.1, 0.9), edgecolors="k", linewidths=0.6,
                                          zorder=4)
        ax_x.add_collection(slice_bubbles)

    # Anregung durchlaufen (wie der Multiplexer im Paper): nächste Konfiguration im Mittelring
    if state["auto"] and state["n"] % AUTO_EVERY == 0:
        nxt = N_PER_RING + (int(P["src"]) - N_PER_RING + 1) % N_PER_RING
        slider_by_key["src"].set_val(nxt)

    fut = state["fut"]                                 # fertige Feldlösung anzeigen, nächste einreichen
    if fut is not None and fut.done():
        state["Y"], state["X"], state["gas"] = fut.result()
        state["t_field"] = time.time()
        state["fut"] = None
        refresh()
    if state["fut"] is None:
        state["fut"] = _pool.submit(feldloesung, gas_alpha(bub_pos, bub_r))
    lines = x_title.get_text().split("\n")
    x_title.set_text(lines[0] + "\n" + lines[1].split("  |")[0]
                     + f"  |  Feldstand {time.time() - state['t_field']:.1f} s alt")
    return tracer_plot,


# --- Regler ---
# (Beschriftung, Schlüssel in P, min, max, Startwert (Anzeigeeinheit), Format, Schritt, Faktor Anzeige->SI)
SLIDERS = [
    ("Spawnrate (Blasen / s)", "spawn", 0, 150, SPAWN_RATE, "%.0f", 1, 1),
    ("Blasenradius, Median (mm)", "r_med", 1, 12, R_MEDIAN * 1e3, "%.1f", 0.5, 1e-3),
    ("Wassergeschwindigkeit Rohrmitte (m/s)", "u_max", 0.05, 1.0, U_MAX, "%.2f", 0.05, 1),
    ("Seitliches Zittern (mm/√s)", "wobble", 0, 40, WOBBLE * 1e3, "%.0f", 1, 1e-3),
    ("Auftrieb relativ zum Wasser (m/s)", "slip", 0, 0.5, SLIP, "%.2f", 0.05, 1),
    ("Leitfähigkeit Wasser (mS/m)", "sigma", 10, 500, SIGMA_W * 1e3, "%.0f", 10, 1e-3),
    ("Kontaktwiderstand R_C (Ω)", "rc", 0, 500, CONTACT_R, "%.0f", 10, 1),
    ("Kontaktkapazität C (nF)", "cc", 5, 2000, CONTACT_C, "%.0f", 5, 1),
    ("Empfangselektrode (nur Parallel-Modus)", "rx", 0, N_EL - 1, RX, "%d", 1, 1),
    ("Anregung / Sender (Querschnittsbild)", "src", 0, N_EL - 1, SRC, "%d", 1, 1),
    ("Farbskala Änderungskarten (± %)", "hm", 0.05, 10, HEATMAP_PERCENT, "%.2f", 0.05, 1),
]
LIVE_REFRESH = {"sigma", "rc", "cc", "rx", "hm"}       # wirken sofort auf die Auswertung
widgets = []
slider_by_key = {}
y_top, dy = 0.955, 0.066
for n_, (label, key, lo, hi, init, fmt, stp, fac) in enumerate(SLIDERS):
    sax = ax_c.inset_axes([0.05, y_top - n_ * dy, 0.80, 0.025])
    sl = Slider(sax, "", lo, hi, valinit=init, valstep=stp, valfmt=fmt)
    sax.text(0.0, 1.9, label, transform=sax.transAxes, fontsize=9, va="bottom")

    def on_change(val, key=key, fac=fac):
        P[key] = val * fac
        if key == "rx":
            el_sc.set_facecolors(el_colors())
        if key == "src":
            draw_feldbild()
            fig.canvas.draw_idle()
        elif key in LIVE_REFRESH:
            refresh()

    sl.on_changed(on_change)
    widgets.append(sl)
    slider_by_key[key] = sl

b_clear = Button(ax_c.inset_axes([0.05, 0.12, 0.38, 0.045]), "Blasen leeren")
b_pause = Button(ax_c.inset_axes([0.47, 0.12, 0.38, 0.045]), "Pause")
b_mode = Button(ax_c.inset_axes([0.05, 0.06, 0.38, 0.045]), "Modus")
b_auto = Button(ax_c.inset_axes([0.47, 0.06, 0.38, 0.045]), "Anregung läuft")
b_clear.on_clicked(lambda _e: clear_bubbles())


def toggle_pause(_e):
    if state["paused"]:
        anim.resume()
        b_pause.label.set_text("Pause")
    else:
        anim.pause()
        b_pause.label.set_text("Weiter")
    state["paused"] = not state["paused"]
    fig.canvas.draw_idle()


def toggle_mode(_e):
    state["paper"] = not state["paper"]
    apply_mode()
    refresh()


def toggle_auto(_e):
    state["auto"] = not state["auto"]
    b_auto.label.set_text("Anregung läuft" if state["auto"] else "Anregung fest")
    fig.canvas.draw_idle()


b_pause.on_clicked(toggle_pause)
b_mode.on_clicked(toggle_mode)
b_auto.on_clicked(toggle_auto)
widgets += [b_clear, b_pause, b_mode, b_auto]

# Vorlauf, damit das Rohr beim Start nicht leer ist
for _ in range(int(H / (U_MAX * DT))):
    step()
apply_mode()
refresh()

anim = FuncAnimation(fig, update, interval=int(DT * 1000), blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()
    # Als GIF speichern (statt plt.show()):
    # anim.save("blasen.gif", writer="pillow", fps=30, frames=300)