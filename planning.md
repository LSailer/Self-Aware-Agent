##   Project: Multi-Agent Extension of Intrinsically Motivated Self-Aware Agents

###   1.  Vision

   * Extend the "Learning to Play with Intrinsically-Motivated, Self-Aware Agents" project to a multi-agent setting.
   * Replace the original world model with a Schmidhuber-style VAE/RNN-based world model.
   * Introduce a comparative analysis between the original and the Schmidhuber world model.
   * Implement object and agent frequency measurements as an evaluation metric.
   * Design agents that are rewarded based on Self-Model predictions of the World Model's prediction error.

###   2.  Architecture

   * Two self-model agents in a shared environment.
   * Agents can either cooperate or hinder each other.
   * PyBullet environment for simulation.

###   3.  Constraints

   * Focus on the first step: setting up the single-agent environment and replacing the world model.
   * Ensure the new VAE/RNN world model functions correctly (predicts next state).
   * The environment will include a plane, agent, cube, and cylinder.
   * The agent will be a simple sphere, and actions will be applied as forces/torques.

###   4.  Tech Stack
   * Python
   * PyBullet
   * PyTorch
   * PyTest
   * 