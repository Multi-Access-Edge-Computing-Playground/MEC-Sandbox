# MEC-Demonstrator
A Playground to learn coding within a Multi-Access Edge Computing Application
### Im Folgenden wird die Roadmap für die MEC-Entwicklungsumgebung beschrieben:

„Entwicklungsumgebung“ soll verdeutlichen dass Studenten und Mitarbeiter ihre Programmierkenntnisse dauerhaft vertiefen können und das Produkt nie fertig sein wird (Continuous Improvement).

Folgendes ist für die MEC-Entwicklungsumgebung vorgesehen:

1. Schritt: Aufsetzen eines MVP (Minimum Viable Product): Alle Raspberry Pi’s und Kameras sind für Benutzer zugänglich und programmierbar, die Kameras werden auf Aluprofil-Stativen montiert. die jeweiligen Scripte auf den Pi’s werden monolithisch programmiert (klassische große Code-Blöcke). Es werden drei Scripte programmiert (KI-basierte Objekt/Konturerkennung, Kommunikation mit Roboter und Qualitätskontrolle mit Tiefenbildkamera)
2. Schritt: ein Projekt-Wiki einführen. Neuankömmlinge sollen in der Lage sein, sich selbstständig in die gut Dokumentierte Umgebung einzuarbeiten.
Schritt: Code-Blöcke in Docker-Container überführen, demonstrieren wie schnell die Programme auf neuer Hardware deployed werden können ohne zig Bibliotheken installieren zu müssen.
3. Schritt: ein simples GUI auf dem die Videostreams der RPi’s dargestellt werden
4. Schritt: Ein Master-Raspberry Pi, ausgestattet mit einem Dashboard (basierend auf dem vorangegangenen Schritt) und der Kubernetes Container-Kompositionssoftware verwaltet und überwacht die Prozesse der einzelnen Raspberry’s. Ab diesem Punkt befinden wir uns tatsächlich im Multi-access Edge Computing.
5. Schritt: Monolithische Codeblöcke in Microservices überführen (große Code-Blöcke werden in unabhängige Prozesse unterteilt, welche untereinander mit sprachunabhängigen Programmierschnittstellen kommunizieren. Die Dienste sind weitgehend entkoppelt und erledigen eine kleine Aufgabe)
6. Schritt: Versionskontrolle einführen, um zwischen verschiedenen „Branches“ wechseln zu können. Anfänger können so mit einem Klick in das verständlichere klassische monolithische Codegefüge aus Schritt 1 wechseln, Fortgeschrittene können die Microservice Architektur weiter ausbauen und verstehen. Die Versionskontrolle erfolgt mittels GitHub, welches zulässt das mehrere Personen gleichzeitig an einem „Repository“ (Projekt-Ordner mit Code) arbeiten können und jede Codeerweiterung dokumentiert wird und rückgängig gemacht werden kann.
7. Schritt: Aus der Entwicklungsumgebung wird ein (gut aussehender) Demonstrator generiert. Ähnlich wie beim KUKA-Innovation Award wird ein Endeffektor mit integriertem Raspberry Pi (beinhaltet alle Module), Accelerator, Kamera und Greifer entwickelt, welcher per Schnellverschluss an einen beliebigen UR gekoppelt werden kann. So können beide Stative wegfallen und die Entwicklungsumgebung wird zu einer mobilen Anwendung.
Warum wird der letzte Schritt nicht zuerst durchgeführt? Voll integrierte Systeme sind schwer zu programmieren, da für das debuggen stets ein Zugang zu der Hardware möglich sein muss und die Komplexität gering gehalten werden sollte (Lieber 3 differenzierbare Raspberry Pi’s statt einem einzelnen)
