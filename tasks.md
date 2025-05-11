##   Step 1: Single-Agent Setup (Completed)

###   Environment & Core Functionality

   * [X] Set up the PyBullet environment with a single agent.
        * [X] Generate a 10x10 room with border walls.
        * [X] Add a sphere agent with a camera.
        * [X] Implement basic agent movement (forward/backward/left/right and rotation).
        * [X] Add a red cube and a blue cylinder as interactable objects.
   * [X] Implement the Schmidhuber VAE/RNN-based world model in Python.
        * [X] Code the VAE (Vision Model).
        * [X] Code the RNN (Memory Model - predicting next latent state).
        * [X] Code the Self Model (predicting world model error).
        * [X] Integrate the VAE, RNN, and Self Model.
      * [X] Implement the agent with action selection based on Self Model predictions.
   * [ ] Implement curiosity-driven reward calculation (based on the actual Self-Model prediction error). (Der Reward ist hoch, wenn die Vorhersage des Self-Models über den RNN-Fehler schlecht war).
   * [X] Implement logging of relevant metrics (position, orientation, losses, interaction frequency).
   * [X] Implement visualization of VAE reconstruction.
   * [X] Implement video recording with metric annotations.
   * [ ] Test the world model by training the agent and observing its behavior (mit der neuen Reward-Struktur).

###   Loss Logging & Visualization Refinement

   * [X] **Separate VAE Loss Components:**
        * [X] Modify `MetricLogger` (`metric_logger.py`) to explicitly log VAE Reconstruction Loss (`vae_recon_loss`) and VAE KL Divergence (`vae_kld_loss`) as separate columns in the CSV, in addition to the total `vae_loss`. (Already logged, ensure plotting uses them).
        * [X] Modify `plot_metrics` in `MetricLogger`:
            * [X] Create a separate subplot for VAE losses (Total, Reconstruction, KLD) due to potentially different scales compared to RNN and Self losses.
            * [X] Alternatively, consider plotting VAE losses on a secondary y-axis if keeping them on the main loss plot.
        * [X] Rerun the single-agent simulation to generate updated logs and plots with separated VAE loss components visible.
        * Implementation Details:
            * Created a figure with two subplots: one for Self/World losses and one for VAE losses
            * Added explicit calculation and visualization of VAE reconstruction loss
            * Improved plot formatting and readability with proper labels and legends
            * Used consistent color coding for different loss types
            * Maintained rolling averages for all metrics

---

##   Step 2: Multi-Agent Extension (Approach 2: Shared Core, Separate Heads) - **IN PROGRESS**

###   A. Environment Modification (`environment.py`)

   * [X] **Agenten Instanziierung:** Füge eine zweite Agenteninstanz hinzu (z.B. `agent_id_1`, `agent_id_2`) mit eindeutiger ID.
   * [X] **Startkonfiguration:** Definiere Startpositionen/Orientierungen für den zweiten Agenten.
   * [X] **Zustandsabfrage (`get_state`):** Erweitere die Funktion, sodass sie die Zustände (Position, Orientierung, etc.) für *beide* Agenten zurückgibt (z.B. unter Schlüsseln wie `"agent_1"` und `"agent_2"`).
   * [X] **Aktionsanwendung (`apply_action`):** Modifiziere die Funktion, sodass sie eine Agenten-ID als Argument akzeptiert und die Aktion nur auf den entsprechenden Agenten anwendet.
   * [X] **Kamerabilder:** Stelle sicher, dass separate Kamerabilder für jeden Agenten abgerufen werden können (ggf. durch eine neue Funktion oder Anpassung von `get_camera_image` mit Agenten-ID). **Entscheidung:** Soll jeder Agent seine eigene Kamera haben oder gibt es eine globale Kamera? (Aktuell hat jeder Agent eine eigene Perspektive, das beibehalten?) -> Ja, beibehalten für individuelle Beobachtungen. Funktion `get_camera_image` anpassen oder duplizieren (`get_camera_image(agent_id)`).
   * [X] **Reset (`reset`):** Aktualisiere die Funktion, um beide Agenten und ggf. Objekte korrekt zurückzusetzen.

###   B. Model Adaptation (`models.py` - now primarily within `multi_agent_controller.py`)

   * [X] **VAE:**
        * [X] **Entscheidung & Implementierung:** Verwende *ein gemeinsames* VAE, das die Bilder beider Agenten verarbeitet (erfordert Batching der Bilder von beiden Agenten für den VAE-Input). Passe die `forward`-Methode an, um einen Batch von Bildern (von beiden Agenten) zu verarbeiten. (Implemented within `MultiAgentController`)
   * [X] **RNNModel (Shared Core, Separate Heads):**
        * [X] **Input Anpassung:** Modifiziere die `__init__`-Methode, um die korrekte `input_size` für den LSTM-Kern zu definieren. Diese muss die konkatenierten latenten Zustände ($z_t^1, z_t^2$) und Aktionen ($a_t^1, a_t^2$) beider Agenten berücksichtigen. (Implemented within `MultiAgentController`)
        * [X] **LSTM Kern:** Behalte die Kern-LSTM-Schicht(en) bei, die diese kombinierte Eingabe verarbeiten und den gemeinsamen Hidden State $h_t$ erzeugen. (Implemented within `MultiAgentController`)
        * [X] **Separate Ausgabe-Köpfe:** Füge *zwei* separate lineare Schichten (`nn.Linear`) hinzu:
            * `fc_out_1`: Nimmt $h_t$ als Input und sagt $z_{t+1}^1$ vorher.
            * `fc_out_2`: Nimmt $h_t$ als Input und sagt $z_{t+1}^2$ vorher. (Implemented within `MultiAgentController`)
        * [X] **Forward Methode:** Passe die `forward`-Methode an:
            * Sie muss die kombinierten $z_t$- und $a_t$-Vektoren beider Agenten als Input nehmen.
            * Sie muss den gemeinsamen Kern durchlaufen, um $h_{t+1}$ zu erhalten.
            * Sie muss *beide* Vorhersagen $(predicted\_z_{t+1}^1, predicted\_z_{t+1}^2)$ mithilfe der separaten Köpfe zurückgeben, zusammen mit dem nächsten gemeinsamen Hidden State $h_{t+1}$. (Implemented within `MultiAgentController`)
   * [X] **SelfModel:**
        * [X] Keine strukturellen Änderungen nötig, aber es werden *zwei separate Instanzen* benötigt ($\Lambda^1, \Lambda^2$). (Implemented within `MultiAgentController`)

###   C. Simulation Logic Adaptation (`simulation.py`)

   * [X] **Initialisierung:**
        * [X] Erstelle die modifizierte `Environment`-Instanz.
        * [X] Erstelle die `MultiAgentController`-Instanz (welche VAE, RNNModel, SelfModels, Controller intern erstellt).
   * [X] **Zentrale Simulationsschleife:**
        * [X] **Beobachtungen holen:** Hole $o_t^1$ und $o_t^2$ von der Umgebung.
        * [X] **Encoding:** (Handled by `MultiAgentController`) Kodiere $o_t^1, o_t^2$ zu $z_t^1, z_t^2$.
        * [X] **Aktionsauswahl:** (Handled by `MultiAgentController`)
            * Für Agent 1: Wähle $a_t^1$ mit $C^1$ basierend auf Vorhersagen von $\Lambda^1$ (Input: $z_t^1, h_t$).
            * Für Agent 2: Wähle $a_t^2$ mit $C^2$ basierend auf Vorhersagen von $\Lambda^2$ (Input: $z_t^2, h_t$).
        * [X] **Aktionen anwenden:** Übergebe $a_t^1$ und $a_t^2$ an `env.apply_action`.
        * [X] **Simulation Schritt:** Führe `env.step_simulation()` aus.
        * [X] **Nächste Beobachtungen:** Hole $o_{t+1}^1$ und $o_{t+1}^2$.
        * [X] **Nächste Zustände berechnen:** (Handled by `MultiAgentController`)
            * Kodiere $o_{t+1}^1, o_{t+1}^2$ zu $z_{t+1}^1, z_{t+1}^2$ (werden für das Training benötigt).
            * Berechne den nächsten gemeinsamen Hidden State $h_{t+1}$ mit dem RNN (Input: $z_t^1, z_t^2, a_t^1, a_t^2, h_t$).
        * [X] **Erfahrung speichern:** Speichere die *gemeinsame* Erfahrung $(o_t^1, o_t^2, a_t^1, a_t^2, ..., o_{t+1}^1, o_{t+1}^2, done)$ in einem Replay Buffer. (Handled by `MultiAgentController`)
        * [X] **Agenten-Zustand aktualisieren:** Setze $h_t \leftarrow h_{t+1}$ für den nächsten Schritt. (Handled by `MultiAgentController`)
        * [X] **Modell-Updates:** Rufe periodisch die `update_models`-Funktion des `MultiAgentController` auf.

###   D. Agent & Update Logic Adaptation (Mainly `multi_agent_controller.py`)

   * [X] **Multi-Agent Management:** `MultiAgentController` Klasse erstellt, die die Controller, Self-Models und die gemeinsamen Modelle (VAE, RNN) verwaltet.
   * [X] **`update_models` Funktion (in `MultiAgentController`):**
        * [X] Sample einen Batch gemeinsamer Erfahrungen aus dem Replay Buffer.
        * [X] **VAE Update:** Trainiere das gemeinsame VAE mit den Beobachtungen $o^1, o^2$ aus dem Batch.
        * [X] **RNN Update:**
            * [X] Führe einen Forward Pass durch das RNN mit den Batch-Daten ($z^1_t, z^2_t, a^1_t, a^2_t, h_t$) durch, um $(pred\_z^1_{t+1}, pred\_z^2_{t+1})$ und $h_{t+1}$ zu erhalten.
            * [X] Berechne Loss $L_1$ zwischen $pred\_z^1_{t+1}$ und dem tatsächlichen $z^1_{t+1}$ (aus dem Batch).
            * [X] Berechne Loss $L_2$ zwischen $pred\_z^2_{t+1}$ und dem tatsächlichen $z^2_{t+1}$.
            * [X] **Kombiniere die Losses:** $L_{RNN} = L_1 + L_2$.
            * [X] Führe Backpropagation mit $L_{RNN}$ durch, um die Gewichte der Köpfe *und* des gemeinsamen Kerns zu aktualisieren.
        * [X] **Self-Model Updates:**
            * [X] Trainiere $\Lambda^1$: Ziel ist der tatsächliche Fehler $L_1$ (Curiosity-Belohnung). Input für $\Lambda^1$ sind $z^1_t, h_t, a^1_t$.
            * [X] Trainiere $\Lambda^2$: Ziel ist der tatsächliche Fehler $L_2$ (Curiosity-Belohnung). Input für $\Lambda^2$ sind $z^2_t, h_t, a^2_t$.

###   E. Logging & Video Adaptation

   * [X] **`metric_logger.py`:**
        * [X] Erweitere `log_metrics`, um Daten für beide Agenten (Position, Aktion, Losses etc.) zu akzeptieren und zu speichern. Füge Spalten für Agenten-ID hinzu.
        * [X] Füge Metriken für Agenten-Interaktionen hinzu (Distanz zwischen Agenten, Interaktion mit Objekten).
        * [X] Passe `plot_metrics` an, um sinnvolle Vergleiche oder separate Plots für beide Agenten sowie Interaktionsmetriken zu erstellen.
            * [X] Plot interaction with objects for both agents combined.
            * [X] Plot interaction with objects for each agent separately.
            * [X] Plot how often agents interact together.
            * [X] Plot World Loss (RNN Loss).
            * [X] Plot Self Model Loss for both agents separately.
   * [X] **`video_recorder.py`:**
        * [X] Passe die `annotate_frame`-Methode an, um relevante Infos (Curiosity, Self-Loss) anzuzeigen. (Bereits vorhanden, wird nun pro Agent im Sim-Loop gefüttert).
        * [X] Stelle sicher, dass in `simulation.py` zwei Recorder instanziiert und mit den jeweiligen Kamerabildern und Daten gefüttert werden, um separate Videos zu erzeugen.

###   F. Analyse & Vergleich

   * [ ] Analysiere emergente Verhaltensweisen: Beobachte, ob Kooperation, Konkurrenz oder spezialisierte Rollen auftreten.
   * [ ] Vergleiche Curiosity-Verlauf (Single- vs. Multi-Agent).
   * [ ] Analysiere Interaktionsmetriken.

###   G. Dokumentation & Maintenance

   * [ ] Dokumentiere den Multi-Agenten-Code und das Setup.
   * [ ] Aktualisiere README.md.
   * [ ] Halte PLANNING.md und TASK.md aktuell.

---

##   Step 3: Further Steps (Optional/Future)

###   Policy Search Methods Comparison

   * [ ] Implement alternative action selection/policy search methods for the multi-agent setting.
   * [ ] Compare the performance of different action selection methods.

###   Testing & Deployment

   * [ ] Generate unit tests for key multi-agent modules/interactions.
   * [ ] Create a Dockerfile for containerizing the multi-agent application.
   * [ ] Define steps to deploy the application on the UniCluster.

###   Hyperparameter Optimization (wandb sweep)

   * [ ] Integrate PyTorch Lightning for cleaner multi-agent training.
   * [ ] Define a `wandb` sweep configuration file for multi-agent hyperparameters.
   * [ ] Modify the training script for `wandb` integration.
   * [ ] Launch hyperparameter search using `wandb sweep`.

###   Documentation & Maintenance

   * [ ] Document the multi-agent code and setup.
   * [ ] Update README.md with multi-agent setup and usage instructions.
   * [ ] Keep PLANNING.md and TASK.md updated.
