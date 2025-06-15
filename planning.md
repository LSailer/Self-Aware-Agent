##   Project: Multi-Agent Extension of Intrinsically Motivated Self-Aware Agents

###   1.  Vision

   * Extend the "Learning to Play with Intrinsically-Motivated, Self-Aware Agents" project to a multi-agent setting.
   * Replace the original world model with a Schmidhuber-style VAE/RNN-based world model.
   * Introduce a comparative analysis between the original and the Schmidhuber world model.
   * Implement object and agent frequency measurements as an evaluation metric.
   * Design agents that are rewarded based on the prediction error of their Self-Model. (Der Reward basiert darauf, wie überraschend der Fehler des World-Models für das Self-Model war - hoher Self-Model-Fehler = hoher Reward).

###   2.  Architecture

   * Two self-model agents in a shared environment.
   * **Implementation Strategy:** Follow **Approach 2: Shared VAE/RNN-Core, Separate "Heads" & Self Models**. This involves:
        * A shared core RNN learning common environmental dynamics based on combined inputs.
        * Separate prediction heads on the RNN, each dedicated to predicting the next latent state for one specific agent ($z_{t+1}^1$ bzw. $z_{t+1}^2$) based on the shared core's output ($h_t$). **(Klarstellung: Dies ermöglicht eine klare Aufgabentrennung für die Vorhersage.)**
        * Separate Self-Models ($\Lambda^1, \Lambda^2$), each predicting the error of its *corresponding* prediction head.
        * Separate Controllers ($C^1, C^2$), each using its own Self-Model for curiosity-driven action selection.
        * **(Klarstellung: Updates des gemeinsamen RNN-Kerns erfolgen koordiniert, basierend auf den kombinierten Fehlern beider Vorhersage-Köpfe.)**
   * Agents can implicitly cooperate or hinder each other through environment interactions and the shared model's influence.
   * PyBullet environment for simulation.

###   3.  Constraints

   * Focus initially on implementing the chosen multi-agent architecture (Approach 2).
   * Ensure the shared VAE/RNN world model functions correctly (predicts individual next states based on joint input and shared hidden state).
   * The environment will include a plane, two agents, a cube, and a cylinder. **(Hinweis: Aktuell sind es Kugeln, ggf. später auf Würfel ändern, falls gewünscht)**.
   * Agents will be simple spheres (oder Würfel), and actions will be applied as forces/torques.

###   4.  Tech Stack
   * Python
   * PyBullet
   * PyTorch
   * PyTest
