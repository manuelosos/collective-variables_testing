Notes on some results:

CNVM_ab2_n500_r100-100_rt003-003_l100_a1000_s150
weird 3. coordinate. May be swapped with some other coordinate.

Runs with lag time 1 behave weird.

Runs with differing r rates (in the magnitude of at least double) seem to have
an interpretable asymmetry in the coordinates plot. 
This asymmetry could be predictable

Color correspondance: Guess from interpreting results: yellow = a, blue = b

Observations:
* Dense points => Points lead to the same dynamic
* If the curve has a thin end: Then the corresponding state handles small outbreaks well
* If end of curve is thick: The corresponding state is easily overthrown by small outbreaks
  * If corresponding state is a: A state with a randomly infected node of high degree can have high impact if the r_ba rate is high. Such a state can be observed as an outlier.


Besprechung
* Holme-Kim Model umsetzen? Man kann Cluster Coeff. nach dem run ausrechnen und in der Datentabelle dann danach sortieren.
* Wegen Dimensions Schätzung Lidl Paper ansprechen
* Wegen Code für Simulation von MFE nachfragen


Haken bei Wurst enden gehen nach hinten. Bisher noch keine Erklärung 
EIn experiment wo es wegen Eigenwerten fehlgeschlagen ist schicken.
Experimente wo Raten komplett auf 0 gesetzt sind machen.
Anschauen wie sich die Lag time auswirkt (Im Detail nicht wirklich interessant)
Dimension estimate ist nicht wirklich vertrauenswürdig. Mal andere schätzer gegentesten.

Konkret Frage für Paper: Für welche Parameter des CNVM kann eine 1-dim Reduktion gemacht werden.

Untersuchung von Transition manifold theoretisch einbinden und als Begründung für Reduktion.

3 verschiedene Parametersets raussuchen und Detailliert herausfinden warum die sich unterscheiden. 
Unterschiede in der Hinsicht welche Koordinaten was beschreiben. i.e erste Dim weighted shares.

Dann reduziertes Model vorstellen.

Interessantes Ergebnis CNVM2_hk2-025_n500_r100-100_rt001-001_l200_a1000_s150

Nächste Schritte:

* Experimente machen wo Raten auf 0 gesetzt wird
* Fehlschlagen von Eigenwertsolver herausfinden! Wichtig!
* 2-3 Parametersets raussuchen die sehr unterschiedliche Transition manifolds erzeugen. Dann werden diese 3 Parametersätze weiter durchtesten.
  * Was ist diese zweite Dim die relevant wird?
  * Dann irgendwann reduziertes Model bauen welches gut funktioniert.
* Lidl Paper schicken
* Ergebnisse schicken wo Eigenwertsolver fehlgeschlagen hat.


#### Notes on Dim Estimates 

* Dim estimate mit Maximierung der Steigung von "Kernel Summe" bezieht sich auf untransformierte Daten "euklidischer Abstand"
* In visualisierungen werden transformierte Daten gezeigt. Die relativen Abstände der Punkte werden hier nur näherungsweise wiedergegeben


#### New notes
16.05.24

CNVM2_ab2_n500_r101-100_rt001-004_l500_a1000_s150_r02 Hat sehr schöne CVs. Struktur der xixi plots zeigt ähnliche "higher order harmonics auf"

### Fragen 27.05.2024

Warum darf smpling per Alias table nur bei Zahlen kleiner als 10^10 gemacht werden? was ist der large number bias?
Was macht alpha mit den bei den degrees? alpha relaxiert Einfluss von high degree nodes