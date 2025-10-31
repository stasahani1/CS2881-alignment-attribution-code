# Safety-Critical Neuron Analysis and Fine-tuning Extension

## Overview

This extension implements experiments to understand **why safety alignment is brittle** in language models. Building on "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications," we investigate the following question about safety degradation during fine-tuning:

1. **Unfrozen Fine-Tuning (Safety Neuron Drift)**: How much do safety-critical neurons change compared to other neurons during normal fine-tuning?

## Research Questions

### Experiment 1: Unfrozen Fine-Tuning (Safety Neuron Drift)

**Setup**: Fine-tune normally (no freezing) and measure how much each safety-critical neuron changes.

**Goal**: Determine if safety-critical neurons are particularly fragile (high drift) or if safety degradation comes from new harmful circuits activating elsewhere.

**Hypotheses**:
- **Hypothesis A (Fragile Safety Neurons)**: Safety-critical neurons move more than average → safety is inherently "fragile" and easily disrupted.
- **Hypothesis B (Recontextualization)**: They move less but are recontextualized—fine-tuning activates new harmful circuits instead of destroying old safety ones.

**Key Metrics**:
- Cosine similarity between pre/post fine-tune weight vectors per neuron
- L2 distance of weight changes
- Comparison: safety-critical vs. random vs. utility-critical neurons