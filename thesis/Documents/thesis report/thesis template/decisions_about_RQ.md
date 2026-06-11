# Thesis Report Update — Decisions & Discussion Points

This file captures the outcome of a planning conversation on research question (RQ) formulation, experimental design, and figure planning. It is intended as a reference for updating the LaTeX thesis report. LaTeX-specific formatting and file structure decisions are deferred — this file only documents *what* needs to change and *why*.

---

## 1. Finalised Research Questions

These replace the earlier RQ1/RQ2/RQ3 in the report.

**RQ1.** Which methods and metrics are suitable for evaluating the utility (statistical resemblance and predictive performance) and privacy risk of synthetic tabular medical data, and what distinct dimension of data quality does each capture?

**RQ2.** How do GAN-based (HealthGAN, PATE-GAN) and LLM-based (Pythia 70M, Pythia 1B) synthesisers compare on the privacy–utility trade-off when applied to tabular medical data, and which achieves a more effective balance?

**RQ3.** At a matched privacy budget (ε=5.0), how does enforcing differential privacy using mechanisms appropriate to each family affect utility, and how do the resulting privacy–utility profiles differ between GAN-based (PATE-GAN vs HealthGAN) and LLM-based (Pythia 1B-DP vs Pythia 1B) synthesisers of tabular medical data?

### Why these RQs (rationale for the change)

- The original RQ2 ("trade-offs in GAN- and LLM-based generation") was descriptive and overlapped with the original RQ3 ("which achieves a better balance"). They have been merged into a single comparative RQ2.
- The original RQ3 has been sharpened to focus specifically on the *DP enforcement effect* at a matched privacy budget, using concrete model pairs. This creates a clean breadth-then-depth progression: RQ2 establishes the non-private landscape; RQ3 measures what DP enforcement costs each family.
- The mechanisms (PATE for GANs, DP-SGD for LLMs) are kept inside the RQ via the phrase "mechanisms appropriate to each family" — specific mechanism names are deferred to methods, not stated in the RQ itself.
- "Matched privacy budget" is preferred over "fixed privacy cost" because it emphasises across-family comparability.

---

## 2. Experimental Design — Model Lineup per RQ

### RQ2 (non-private landscape, plus PATE-GAN as DP-GAN benchmark)

Five synthesisers, plotted on the privacy–utility plane:
- HealthGAN (non-private GAN)
- PATE-GAN (DP-GAN — already includes DP by construction)
- Pythia 70M (non-private LLM)
- Pythia 1B (non-private LLM)
- Pythia 1B-DP (DP LLM)

Note: PATE-GAN appears in RQ2 because it is the standard DP-GAN benchmark in the literature. Its position on the plane reflects its built-in DP, which should be acknowledged in the methodology when discussing the RQ2 figure.

### RQ3 (DP enforcement effect, within-family)

Four synthesisers — a subset of the RQ2 set — viewed through a different analytical lens:
- HealthGAN → PATE-GAN (GAN-family DP toggle)
- Pythia 1B → Pythia 1B-DP (LLM-family DP toggle)

Pythia 70M is excluded from RQ3's main comparison because DP-SGD failed at that scale (see Section 4).

### Note on shared data

RQ2 and RQ3 share the underlying experimental data. They are not separate experiments — they are two analytical lenses on the same five model outputs. This must be made explicit in the report so a reader understands RQ3 is not a re-run but a focused re-analysis.

---

## 3. Visualisation Plan

### RQ2 main figure — landscape view

- **Type:** scatter plot on privacy–utility plane
- **Points:** all five synthesisers
- **Encoding:** colour by family (GAN vs LLM); marker shape can distinguish private vs non-private variants
- **Visual question:** "Where does each family sit, and does one family dominate?"
- **Optional overlay:** Pareto frontier or shaded family regions

### RQ3 main figure — within-family DP effect view (combined 4-point arrow plot)

**Decision: combined chart with arrows, not two separate within-family charts.**

- **Type:** privacy–utility plane with arrows showing within-family DP transitions
- **Points:** four (HealthGAN, PATE-GAN, Pythia 1B, Pythia 1B-DP)
- **Arrows:**
  - HealthGAN → PATE-GAN (GAN-family DP arrow)
  - Pythia 1B → Pythia 1B-DP (LLM-family DP arrow)
- **Visual question:** "How far does each arrow move, and in what direction? Which family degrades more gracefully under DP?"
- **Annotations to consider:** arrow length labels, utility drop / privacy gain values

**Rationale:** RQ3 is explicitly a *cross-family comparison of DP effects*. The figure should make this comparison visually immediate. Two separate within-family charts force the reader to manually compare across panels.

### RQ3 supplementary panels — within-family detail (optional, recommended)

Per-family panels useful when reporting *multiple metrics* (e.g., predictive performance, distributional similarity, MIA success separately):
- Panel A: HealthGAN vs PATE-GAN across all metrics
- Panel B: Pythia 1B vs Pythia 1B-DP across all metrics

Can be appendix or sub-figures.

### One-figure-per-metric vs. composite

If multiple metrics are being reported, consider one arrow plot per metric rather than collapsing to a single composite score. This enables findings like "GANs degrade more on predictive utility but less on distributional similarity under DP," which is richer than a single composite arrow.

---

## 4. The Pythia 70M DP-SGD Failure

This is to be reported as a finding, not hidden as a methodological inconsistency.

### In the results section (RQ3) — drop-in paragraph

> An initial attempt to apply DP-SGD to Pythia 70M at ε=5.0 failed to train usefully — the injected noise exceeded gradient magnitudes, and the model did not converge. We therefore scaled to Pythia 1B, which trained successfully under DP-SGD at the same privacy budget. This itself constitutes a finding: DP-SGD on small LLMs is impractical at moderate privacy budgets, consistent with prior work showing that larger LLMs tolerate DP noise more gracefully.

### In the methodology section

Brief note explaining why the LLM-side DP comparison uses Pythia 1B specifically (rather than 70M), referencing the failed attempt at 70M.

### Suggested supporting citation

Li et al., *"Large Language Models Can Be Strong Differentially Private Learners"*, ICLR 2022 — for the claim that larger LLMs tolerate DP noise more gracefully.

---

## 5. Known Caveats & Limitations to Acknowledge

These should appear in either the methodology or limitations sections.

### 5.1 HealthGAN ↔ PATE-GAN is not a clean DP toggle

HealthGAN and PATE-GAN are *architecturally different* models — one of which happens to be DP. This is asymmetric with the LLM side, where Pythia 1B vs Pythia 1B-DP is the same base model with DP-SGD applied. The GAN-side comparison therefore conflates architectural and mechanism differences. Suggested framing:

> The GAN-family DP comparison conflates architectural and mechanism differences (HealthGAN and PATE-GAN are not the same architecture with DP toggled), whereas the LLM-family comparison isolates the DP-SGD effect. This asymmetry is unavoidable given the absence of a standardised non-private analogue of PATE-GAN, but should be considered when interpreting cross-family differences in DP cost.

### 5.2 PATE-GAN's dual role in RQ2

PATE-GAN appears alongside non-private models in RQ2's landscape comparison even though it incorporates DP by construction. This is not a flaw but should be made explicit:

> PATE-GAN is included in RQ2 because it is the de facto DP-GAN baseline in the literature; we note that it incorporates differential privacy by design (unlike HealthGAN and non-private Pythia variants), and interpret its position on the privacy–utility plane accordingly.

### 5.3 Shared data across RQ2 and RQ3

The report should make clear that RQ3 reuses RQ2's experimental outputs under a different analytical lens (within-family DP toggle), rather than being a separate experiment.

---

## 6. Pending Code Changes (Do Not Implement Yet — Just Note)

These are deferred until after the report-text updates are stable. Listed here so the figure-generation pipeline can be updated in sync with the new RQ framing.

### RQ3 figure — arrow chart implementation needed

Current plotting code likely shows independent points or per-family panels. Needs modification to:
- Plot all four points (HealthGAN, PATE-GAN, Pythia 1B, Pythia 1B-DP) on a single privacy–utility plane
- Draw arrows from non-private to DP variant within each family
- Family-coded colours; private/non-private encoding via marker style
- Optional: arrow length annotations showing utility drop and privacy gain

### Possible additional changes

- If multiple metrics are reported, replicate the arrow chart per metric (one per privacy or utility dimension).
- Supplementary within-family panels (Section 3, "Supplementary panels") may need separate plotting scripts.
- RQ2's main figure may need restyling to make family membership visually obvious (colour-coding) if not already done.

---

## 7. Open Questions / TBD

To revisit later when updating the report:

- **"More effective balance" operationalisation in RQ2.** The exact criterion (Pareto dominance, composite score, qualitative discussion at matched privacy levels) — confirm this is already explicit in the existing methodology/metrics section, or add it.
- **Supplementary panels (Section 3) — in or out?** Decide whether to include per-family detailed panels in the appendix or skip them.
- **Limitations section update.** Add the two caveats from Section 5 if not already present.
- **RQ1 §4.1 metric-count mismatch (revisit later).** The RQ1 answer (§4.1.2 Table 4.1 + §4.1.4) claims the *eleven* single-number metrics constitute "the set deemed suitable", but the methodology evaluation framework (§3.3) defines more instruments than these eleven — per-column KDE / distributional match, summary statistics (mean, std), PCA manifold overlap, ROC-curve overlay, feature-importance Spearman rank, NNAA + PrivacyLoss, and the per-attribute AIA breakdown. The eleven are only the headline-scalar subset shown in the comprehensive summary table. Fix: reword §4.1.4 (and Table 4.1 framing) so the eleven are described as the *headline single-number subset*, with the visual/auxiliary diagnostics in §3.3 acknowledged as supporting instruments — rather than implying the framework has only eleven metrics. Keeps the RQ1 answer honest against §3.3.
- **RQ1 "methods" half thin (revisit later).** RQ1 asks for "methods *and* metrics". §4.1 is metric-centric; the evaluation *methods* (TSTR/TRTR regime, MIA and AIA simulation protocols, Cox regression, distance-based attacks) are named only in passing. Consider a sentence or short list distinguishing evaluation methods from the metrics they produce.
- **RQ1 §4.1.3 minimal suite still empty.** §4.1.4 forward-references the recommended minimal evaluation suite (§4.1.3), but that subsection is an empty TODO. Either write it or remove the forward reference. (Minimal suite is not in RQ1's literal wording, so optional — but the answer currently promises it.)

---

## 8. What This File Is For

This file documents the *intent* behind the report updates. When using Claude Code in VS Code to apply changes to the LaTeX source:

- Reference this file for the finalised RQ wording (Section 1).
- Reference Section 4 for the exact paragraph to drop into RQ3 results.
- Reference Section 5 for caveats that need to appear in methodology or limitations.
- Defer figure code changes (Section 6) until report text updates are stable.
