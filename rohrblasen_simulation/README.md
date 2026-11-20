# 3D-Rohrstroemung mit Luftblasen und 48 Elektroden

## Inhalt
- 3D-zylindrisches Rohr: Durchmesser 20 cm, Hoehe 1 m.
- Drei Elektrodenringe bei z = 0.45, 0.50 und 0.55 m.
- 16 Elektroden pro Ring, insgesamt 48.
- Jede Elektrode sendet gleichzeitig eine eigene Frequenz.
- Stochastisch erzeugte, kugelfoermige Luftblasen steigen im Rohr auf.
- Elektrisches Feld: komplexe quasi-statische Finite-Volumen-Loesung
  `div((sigma + j*omega*epsilon) grad(phi)) = 0`.
- Luftblasen veraendern die lokale komplexe Admittivitaet.
- Ausgabe der komplexen Uebertragungsstroeme, ihres Betrags und ihrer Phase.
- Die Matrixdiagonale (eigene Senderelektrode) wird ausgeblendet.

## Installation

Python 3.10 oder neuer wird empfohlen.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Unter Windows:
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Einstellungen

Alle Parameter stehen am Anfang von `main.py` in der Klasse `Config`.

Wichtige Parameter:
- `water_conductivity`: Wasserleitfaehigkeit in S/m.
- `air_conductivity`: Luftleitfaehigkeit in S/m.
- `water_relative_permittivity`: relative Permittivitaet des Wassers.
- `frequency_start`, `frequency_step`: Frequenzzuordnung der Elektroden.
- `grid_spacing`: Gitterweite. Kleinere Werte verbessern die Aufloesung,
  erhoehen aber Speicherbedarf und Rechenzeit stark.
- `electrical_every_n_steps`: wie oft das elektrische Problem neu geloest wird.
- `water_velocity`, `bubble_rate`, `bubble_radius_min/max`: Blasenmodell.

Beispiel: Bei `frequency_start=1000` Hz und `frequency_step=1000` Hz
senden die Elektroden Frequenzen von 1 kHz bis 48 kHz.

## Ausgabedateien

Im Ordner `output/`:
- `transfer_current_step_XXXXX.csv`: komplexe 48x48-Strom-Uebertragungsmatrix.
  Zeile i bedeutet Sender i, Spalte j Empfaenger j.
- `current_amplitude_A.csv`: Betrag der zuletzt berechneten Matrix.
- `current_phase_deg.csv`: Phase in Grad.
- `frequencies_hz.csv`: Frequenz jeder Elektrode.
- `latest_result.png`: letzte Visualisierung.
- `run_config.json`: verwendete Parameter.

Leere/NaN-Diagonaleintraege sind absichtlich ausgelassen.

## Modellannahmen und Grenzen

1. Die Blasen werden kinematisch bewegt; es wird keine Navier-Stokes- oder SPH-
   Stroemung geloest. Die Blasengeschwindigkeit ist an die vorgegebene
   Wasseraufwaertsgeschwindigkeit gekoppelt.
2. Das elektrische Problem ist eine 3D-Finite-Volumen-Naeherung auf einem
   kartesischen Gitter innerhalb eines Zylinders. Die Rohrwand ist ausserhalb
   der Elektroden elektrisch isolierend (Null-Normalstrom).
3. Elektroden sind im diskreten Modell jeweils eine wandnahe Gitterzelle.
   Ein realistischeres Modell benoetigt Flaechenelektroden, Kontaktimpedanzen
   und ein genaueres Netz (z.B. Complete Electrode Model in FEM).
4. Fuer eine Sendefrequenz wird die zugehoerige Elektrode auf die Sendespannung
   gesetzt; die uebrigen Elektroden sind bei genau dieser Frequenz auf 0 V.
   Die Stroeme an diesen Elektroden sind die Empfangssignale. Das entspricht
   simultaner Mehrfrequenz-Spannungsanregung mit Strommessung.
5. Die Ausgabe ist ein komplexer Strom (Amplitude und Phase), kein direkt
   kalibriertes Sensorsignal. Die Diagonale wird nicht ausgewertet.
6. Die aktuelle Standardgitterweite von 1 cm ist ein Kompromiss. Fuer
   quantitative Ergebnisse muss eine Gitterkonvergenzstudie erfolgen.

## Hinweise zur Laufzeit

Es werden 48 komplexe lineare Gleichungssysteme je elektrischem Update geloest.
Zum schnellen Test:
- `grid_spacing = 0.02`
- `total_time = 0.2`
- `electrical_every_n_steps = 10`
- `live_visualization = False`

Danach schrittweise feinere Gitter und laengere Laufzeiten verwenden.
