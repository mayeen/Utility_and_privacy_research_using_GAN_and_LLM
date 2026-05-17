# Section 2.4.4 — DP-Augmented LLM Synthesis

## Angle / Framing

This subsection establishes the **research category** that the thesis occupies: combining LLM-based tabular synthesis with differential privacy. The angle is **landscape first, then positioning** — not a deep-dive into any single paper.

Two distinct paths to DP + LLM synthesis exist:

1. **DP at training time** — apply DP-SGD during LM fine-tuning (the Pythia-DP path used in this thesis).
2. **DP at generation time** — generator trains normally, noise is added during synthesis (the SafeSynthDP path).

The subsection should make this split explicit so that the thesis contribution (DP-SGD on Pythia + LoRA, evaluated on healthcare benchmark) has a clear place on the map.

## What to Write

### Opening (1 paragraph)
- DP-augmented LLM synthesis is an emerging frontier.
- Most existing DP synthesis work is GAN-based (PATE-GAN, DP-CTGAN).
- LLM + DP for tabular data has had minimal exploration.
- Two architectural paths exist (training-time vs generation-time DP).

### SafeSynthDP as Precedent (1 paragraph)
- Uses Laplace and Gaussian mechanisms applied during generation.
- Generator is trained normally; DP knob is turned at synthesis.
- Empirically traces (ε, utility) frontier.
- Reports competitive downstream accuracy + membership inference resistance.
- Cite: `\cite{Nahid2024}`.

### Thesis Positioning (1 short paragraph)
- This thesis occupies the **DP-SGD path** (training-time DP) rather than SafeSynthDP's generation-time path.
- DP-SGD couples privacy to optimisation; epsilon is fixed at training time.
- LoRA reduces the trainable parameter count so DP noise has less surface to corrupt.
- Explicit gap: no prior head-to-head DP-LLM vs DP-GAN comparison on a healthcare benchmark with formal (ε, δ) accounting.

## Length

Single subsection, **~0.5 page**. Do not split into further subsections — literature is too thin to support 2.4.4.1, 2.4.4.2, etc. Method details for Pythia-DP belong in Chapter 4 (Methods), not here.

## Results to Mention

- **SafeSynthDP:** can mention that they report competitive TSTR accuracy across multiple epsilons and resist standard membership inference attacks. No need for specific numbers; the qualitative claim is enough for a background section.
- **Do not** report Pythia-DP results here. Those belong in Chapter 5 / 6 (Results / Discussion).

## Additional Papers Worth Citing

- `\cite{Abadi2016DPSGD}` — DP-SGD foundational paper. Justifies the training-time DP path. (Already cited elsewhere in your file; reference it here too.)
- `\cite{Yousefpour2021Opacus}` — Opacus, the implementation used for your Pythia-DP runs. Worth one mention here so the reader knows the engineering substrate.
- `\cite{Hu2022LoRA}` — LoRA. Brief reference: why parameter-efficient adapters reduce DP-SGD's noise footprint.
- Optional: any recent DP fine-tuning paper for LLMs (e.g., Yu et al. 2022 on DP fine-tuning of language models) if you can find one specifically targeting tabular or structured text. This would strengthen the "frontier" claim.
- Optional: `\cite{Miletic2024}` cross-reference — Tabula established LLM-tabular utility but skipped privacy, so it sets up the DP gap that 2.4.4 then names.

## Cross-References to Add

- Forward-reference to Chapter 4 where Pythia-DP method is described.
- Backward-reference to 2.3.3 (PATE-GAN) so the reader can see the GAN-DP counterpart of what 2.4.4 introduces on the LLM side.
- Backward-reference to 2.4.1 (Tabula) — the non-DP LLM baseline that motivates adding DP.

## Suggested Sentence Skeleton

> "DP-augmented LLM synthesis sits at the intersection of two literatures. The first applies DP at training time (DP-SGD \cite{Abadi2016DPSGD}); the second injects noise at generation time. SafeSynthDP \cite{Nahid2024} exemplifies the latter, pairing LLM-driven row synthesis with Laplace and Gaussian mechanisms to trace explicit (ε, utility) frontiers while resisting membership inference. This thesis occupies the former regime — DP-SGD applied to Pythia via LoRA adapters \cite{Hu2022LoRA, Yousefpour2021Opacus} — and provides the head-to-head DP-LLM vs DP-GAN benchmark on healthcare data that the literature currently lacks (cf. §2.3.3 for PATE-GAN, §2.4.1 for Tabula's non-DP precedent)."

---

## 2.4 LLM-based Approaches for Tabular Data

### 2.4.1 Tabula and the Row-as-Text Paradigm

The Tabula framework \cite{Miletic2024} approaches tabular synthesis by serialising each data row as a natural-language text sequence, exploiting the sequential decoding behaviour of autoregressive language models to reconstruct column values in order. Empirical evaluation demonstrates that Pythia-based generators can match or surpass CTGAN \cite{Xu2019CTGAN} on Train-on-Synthetic Test-on-Real (TSTR) utility, with performance gains scaling consistently as model capacity increases from 70M to 410M to 1B parameters. The investigation is, however, confined to utility metrics and leaves the privacy properties of the generated data unexamined. This thesis directly addresses that gap by subjecting the same row-as-text paradigm to formal differential-privacy analysis.

**Rationale:** The row-as-text serialisation strategy is adopted in this thesis because the sequential, left-to-right decoding of a decoder-only language model maps naturally onto the ordered generation of column values within a tabular row, without requiring architectural modifications to the underlying model.

### 2.4.2 The Pythia Model Family

Pythia \cite{Biderman2023Pythia} is a suite of decoder-only transformer language models spanning parameter counts from 70M to 12B, released with open weights and full intermediate checkpoint lineage to facilitate interpretability research. The consistent training protocol and architectural transparency across model sizes make Pythia well suited to controlled scaling studies, where a single variable — model capacity — can be isolated and its effect on downstream task performance measured cleanly. Pythia-70M is selected as the primary baseline in this thesis on the grounds that it fits on consumer-grade GPU hardware when combined with LoRA adapters, offers stable pretrained weights, and provides sufficient representational capacity for the diabetes readmission tabular task. The 410M and 1B variants are employed as model-size scaling references to examine whether the utility and privacy trade-offs observed at 70M generalise across the Pythia family.

**Rationale:** Pythia-70M was selected over larger variants as the default training configuration because the combination of manageable parameter count and LoRA-based fine-tuning renders differentially-private optimisation via DP-SGD computationally feasible within the hardware constraints of this project, while still enabling meaningful performance comparisons at scale.

### 2.4.3 LoRA Fine-tuning for Parameter-Efficient Adaptation

Low-Rank Adaptation (LoRA) \cite{Hu2022LoRA} introduces trainable low-rank decomposition matrices into each targeted weight matrix of a pretrained model while keeping the original weights frozen, thereby dramatically reducing the number of parameters updated during fine-tuning. In the context of differentially-private training, this reduction in trainable parameter count is critical: DP-SGD computes per-sample gradients exclusively over the LoRA adapter parameters, thereby confining the surface across which Gaussian noise is injected and mitigating the utility degradation that full-model DP-SGD would otherwise incur. The final configuration adopted in this thesis employs rank 8, scaling factor alpha 16, and dropout 0.0, a setting chosen to balance representational expressiveness against the noise sensitivity imposed by the privacy budget.

**Rationale:** LoRA was selected as the adaptation strategy because applying DP-SGD to the full parameter space of even a 70M-parameter model would require prohibitive per-sample gradient computation and would inject noise at a scale that renders the model's output statistically uninformative; restricting adaptation to low-rank matrices concentrates the privacy cost on a tractable parameter subset.

### 2.4.4 DP-Augmented LLM Synthesis

DP-augmented LLM synthesis constitutes an emerging frontier in which language model-based tabular generation is coupled with formal differential-privacy guarantees, a regime largely unexplored relative to the more established GAN-based DP literature (cf. §2.3.3 for PATE-GAN as the GAN-DP counterpart). Two architecturally distinct paths to this combination exist: privacy may be enforced at training time by applying DP-SGD \cite{Abadi2016DPSGD} during language model fine-tuning, or it may be enforced at generation time by leaving the generator's training unconstrained and injecting calibrated noise into the synthesis procedure. SafeSynthDP \cite{Nahid2024} exemplifies the second path, pairing LLM-driven row synthesis with Laplace and Gaussian mechanisms to trace explicit $(\varepsilon, \text{utility})$ frontiers while reporting competitive downstream accuracy and resistance to membership inference attacks. This thesis occupies the first path, applying DP-SGD to Pythia via Opacus \cite{Yousefpour2021Opacus} and LoRA adapters \cite{Hu2022LoRA} — the latter reducing the trainable parameter count so that injected noise corrupts a smaller portion of the learned representation (cf. §2.4.1 for Tabula as the non-DP LLM precedent motivating this privacy extension). The specific method configuration is described in Chapter 4; the present subsection establishes the research category and positions the thesis contribution against it. To the best of the author's knowledge, no prior work provides a head-to-head comparison of a DP-LLM generator against a DP-GAN generator on a healthcare benchmark with formal $(\varepsilon, \delta)$ accounting, which constitutes the central empirical gap this thesis addresses.

**Rationale:** This subsection is maintained as a single flat entry rather than subdivided into further nested subsections because the DP-augmented LLM literature is presently too sparse to sustain meaningful thematic subdivision; the two-path framing (training-time versus generation-time DP) provides sufficient taxonomic structure for a background section, while implementation details of the Pythia-DP configuration are properly deferred to Chapter 4.

---

## 2.5 Differential Privacy

### 2.5.1 (ε, δ)-Differential Privacy and Composition

Differential privacy provides a formal mathematical framework for bounding the influence any single individual's record can exert on the output of a randomised algorithm \cite{Dwork2006DP,Dwork2014Foundations}. A mechanism satisfies (ε, δ)-differential privacy if, for every pair of neighbouring datasets differing in exactly one record and for every measurable output subset, the probability of any given output under one dataset is bounded above by the exponential of ε times the probability under the other dataset, plus a small additive slack term δ. The privacy budget ε controls the tightness of this multiplicative bound — smaller values enforce stricter indistinguishability — while δ is conventionally set to be substantially smaller than the inverse of the dataset size, with 10⁻⁵ representing a standard choice for datasets of order 10⁵ records. The global sensitivity of a query, defined as the maximum change in its output over all neighbouring dataset pairs, determines the scale of noise that must be added to satisfy the guarantee, with the Laplace mechanism covering pure ε-DP and the Gaussian mechanism covering the approximate (ε, δ) variant.

Iterative procedures that query the private data repeatedly accumulate privacy cost, requiring composition theorems to characterise the aggregate budget expenditure. Basic sequential composition bounds the cost of k mechanisms, each ε-DP, by the linear quantity k·ε, but this bound is loose for the thousands of gradient steps typical in deep learning. The advanced composition theorem of \cite{Dwork2014Foundations} tightens this to a bound growing approximately as the square root of k times log(1/δ) times ε, yet even this improvement proves insufficient for the repeated Gaussian-noise injections of DP-SGD, motivating the adoption of Rényi-divergence-based accountants (cf. §2.5.4).

**Rationale:** The inadequacy of both basic and advanced composition for practical deep learning training schedules is the primary motivation for the tighter accounting framework introduced in §2.5.4; establishing this limitation here ensures the later technical development is properly contextualised.

### 2.5.2 DP-SGD and Opacus

The Differentially-Private Stochastic Gradient Descent algorithm of \cite{Abadi2016DPSGD} extends standard mini-batch optimisation with two privacy-preserving operations executed at every training step: per-example gradient clipping to a fixed L2 norm threshold C, followed by the addition of isotropic Gaussian noise with standard deviation proportional to σC to the sum of clipped gradients, where σ is the noise multiplier. Training examples are drawn via Poisson subsampling, in which each record is included in a mini-batch independently with a fixed probability, a choice that enables the amplification-by-subsampling property exploited in privacy accounting. The total privacy expenditure across T training steps is tracked by a tight accountant operating over Rényi divergences (cf. §2.5.4) rather than by naïve sequential composition, which would substantially overestimate the true budget consumption.

**Rationale:** Per-example gradient clipping is essential to the DP-SGD construction because it bounds the L2 sensitivity of the gradient sum to at most C regardless of the individual record's loss landscape, thereby ensuring that the Gaussian noise of scale σC is sufficient to provide a formal privacy guarantee; without this sensitivity cap, no fixed noise level could satisfy the definition for adversarially chosen inputs.

The Opacus library \cite{Yousefpour2021Opacus} implements DP-SGD for the PyTorch ecosystem, providing per-example gradient computation, automated clipping, noise injection, and privacy accounting as drop-in extensions to standard training pipelines. Opacus is adopted in this thesis to train the Pythia-DP generator; method-specific configuration details, including the chosen noise multiplier and target (ε, δ), are deferred to Chapter 4.

### 2.5.3 The PATE Framework

The Private Aggregation of Teacher Ensembles (PATE) framework of \cite{Papernot2017PATE} provides an alternative route to differentially-private learning in which privacy is achieved through data partitioning and noisy aggregation rather than through direct perturbation of gradients. The private training data is partitioned into N disjoint subsets, and an independent teacher model is trained on each partition without any privacy mechanism applied; a student model is then trained exclusively on unlabelled public data, using labels generated by a noisy voting procedure over the teacher ensemble in which Laplace noise is added to each class vote count before the argmax is taken. The disjointness of the partitions is the structural property that makes formal privacy guarantees possible: because any individual record influences exactly one teacher, the sensitivity of the aggregate vote count to the substitution of a single record is bounded by one, allowing calibrated noise to confer an (ε, δ)-DP guarantee on each released label.

**Rationale:** Disjoint data partitioning is not a heuristic design choice but a necessary condition for bounding sensitivity to one, which is the value upon which the formal privacy analysis depends; overlapping partitions would raise the sensitivity and require proportionally more noise, undermining the utility of the released labels.

Once trained, the student model never accesses the private dataset, so the post-processing property of differential privacy ensures the guarantee transfers from the noisy labels to the student's parameters. A notable efficiency property of the framework is that when teachers reach strong consensus on a query, the privacy budget consumed is smaller than when they disagree, allowing the system to exploit high-confidence queries without disproportionate privacy cost. The general PATE framework described here underpins the PATE-GAN adaptation for generative modelling reviewed in §2.3.3.

### 2.5.4 Rényi DP and the Moments Accountant

The repeated application of the Gaussian mechanism in DP-SGD and the iterative noisy label aggregation in PATE both demand composition tools more refined than those provided by the standard (ε, δ)-DP framework. Rényi Differential Privacy (RDP), proposed by \cite{Mironov2017RDP}, expresses the privacy guarantee in terms of the Rényi divergence of order α greater than one between the mechanism's output distributions on neighbouring datasets; a mechanism is (α, ε̄)-RDP if this divergence does not exceed ε̄ for all neighbouring pairs. The central computational advantage of RDP is that sequential composition is linear at any fixed order α: k mechanisms each satisfying (α, ε̄)-RDP compose to a single (α, k·ε̄)-RDP mechanism, with no square-root inflation of the kind incurred by the advanced composition theorem of standard differential privacy.

**Rationale:** The linear composition property of RDP under a fixed Rényi order is the primary reason RDP supplants (ε, δ)-DP as the accounting framework for DP-SGD and PATE; because DP-SGD may execute thousands of noisy gradient steps, even the tighter advanced composition bound grows too quickly to yield practically useful privacy budgets, whereas RDP accumulation remains proportional to the number of steps.

An (α, ε̄)-RDP guarantee is converted back to a standard (ε, δ)-DP statement by adding the term log(1/δ)/(α−1) to the Rényi bound, a formula that holds for any chosen δ in (0, 1) \cite{Mironov2017RDP}. In practice, an accountant tracks the accumulated RDP cost at multiple orders α simultaneously throughout training, and at the point of reporting selects the order that minimises the resulting ε at the desired δ. The Gaussian mechanism, the subsampled-Gaussian mechanism employed by DP-SGD with Poisson batching, and the Laplace mechanism employed by PATE all admit closed-form RDP bounds at every order α, making RDP the standard accounting tool for both algorithms. RDP generalises the earlier moments accountant of \cite{Abadi2016DPSGD}, which tracked higher moments of the privacy-loss random variable to obtain tighter bounds for repeated Gaussian queries than basic composition would yield.

---

## 2.6 Evaluation Frameworks for Synthetic Data — Angle Guidance

### Closing Commitment Paragraph for §2.6.3

The background chapter, as currently drafted, surveys the full landscape of statistical similarity, predictive performance, and privacy metrics across §2.6.1–§2.6.3 without declaring which specific measures will be reported in the empirical chapters. A concise closing paragraph appended at the end of §2.6.3 should remedy this by committing the thesis to a concrete, named subset spanning all three branches, thereby allowing the reader to form precise expectations before reaching Chapters 5 and 6. The paragraph should follow the structure: "From the metrics surveyed in §2.6.1–§2.6.3, this thesis reports [statistical similarity metrics — to be confirmed by the user from §2.6.1], [predictive performance metrics under TSTR — accuracy, F1, and/or AUC, to be confirmed by the user from §2.6.2], and [privacy metrics]." On the privacy axis, the metric selection is already settled: nearest-neighbour-distance-based evaluation is fully implemented for HealthGAN, PATE-GAN, and Pythia, with artefacts available at `thesis/privacy_evaluation/results/`, and Nearest-Neighbour Adversarial Accuracy of \cite{yale2020synthesizing} is therefore a confirmed reporting choice. The user must still confirm which statistical similarity metrics from §2.6.1 will be carried forward; the paragraph should close with a forward reference to Chapter 5 where the rationale governing all metric selections is elaborated.

**Rationale:** A background chapter that surveys but never commits to a metric subset leaves the reader unable to anticipate the empirical chapters and creates the risk that the author appears to have retained post-hoc flexibility in metric selection; an explicit commitment paragraph at the close of §2.6.3 binds the literature review to the experimental protocol, signals methodological transparency, and prevents the impression that metrics were selected after results were known.

### Anchor Equations for the Three Evaluation Pillars

Across §2.5, each subsection introduces its principal formal construct as a numbered equation — the (ε, δ)-DP definition, the DP-SGD gradient update, the PATE noisy-voting procedure, and the RDP divergence bound — establishing a stylistic convention in which theoretical machinery is grounded in explicit mathematical notation before being discussed in prose. Section 2.6 currently maintains a prose-only register, creating an asymmetry that may convey that evaluation methodology is treated as less rigorous than privacy theory. To restore stylistic consistency, one anchor equation or compact formal definition per branch should be introduced. For §2.6.1, the Kolmogorov–Smirnov statistic serves as the canonical univariate similarity measure, defined as $D_{KS} = \sup_x |F_{real}(x) - F_{synth}(x)|$, where $F_{real}$ and $F_{synth}$ denote the empirical cumulative distribution functions of the real and synthetic samples respectively; this single equation suffices because the remaining univariate metrics cited in §2.6.1 (KL divergence, moment comparisons) are standard, and multivariate metrics such as MMD and Wasserstein distance may remain in prose unless the user wishes to extend the formalism further. For §2.6.2, a compact definitional sentence in inline mathematics is appropriate rather than a numbered equation: given a generator $G$ trained on real data $D_{\text{train}}$, a classifier $h$ is fitted on a synthetic sample $S = G(z)$ matched in size to $D_{\text{train}}$ and evaluated on the held-out real test set $D_{\text{test}}$; the reported metric is then a function of $h$ evaluated on $D_{\text{test}}$. For §2.6.3, the Nearest-Neighbour Adversarial Accuracy of \cite{yale2020synthesizing} provides the most load-bearing formal definition and should be presented as a numbered equation drawn directly from the HealthGAN paper: AA combines the fraction of real records whose nearest neighbour in the pooled real-and-synthetic set is another real record with the analogous fraction for synthetic records, with values near 0.5 indicating statistical indistinguishability and values diverging from 0.5 signalling either memorisation or insufficient coverage; this equation is the most critical of the three anchor definitions because §2.6.3 cites AA repeatedly and the empirical chapters will report it directly.

**Rationale:** Adding one anchor equation per evaluation branch ensures that each pillar is grounded in at least one canonical formal definition to which later chapters can refer back, matching the convention established in §2.5 and avoiding the asymmetric impression that differential-privacy theory warrants mathematical rigour while evaluation methodology does not; the AA equation in §2.6.3 is particularly load-bearing given its repeated citation throughout §2.6.3 and its direct role in the empirical reporting of Chapters 5 and 6.

---

## Methodology

### Scope Decision: Exclusion of a Differentially-Private HealthGAN Variant

The experimental design confines the differentially-private synthesis path to the LLM-based Pythia pipeline and does not introduce a DP-augmented variant of HealthGAN, despite the existence of prior DP-GAN literature including DPGAN \cite{Xie2018DPGAN}, PATE-GAN \cite{Jordon2019PATEGAN}, DP-WGAN, and DP-CTGAN. This decision rests on both technical incompatibilities specific to the WGAN-GP architecture and scientific-scope considerations particular to this thesis.

On the technical side, four compounding obstacles render a well-controlled DP-WGAN-GP variant impractical. First, the gradient penalty that distinguishes WGAN-GP from weight-clipping WGAN \cite{Arjovsky2017WGAN} is itself a function of real data: it is computed over interpolated samples drawn between real and generated points, meaning its gradient cannot be cleanly clipped on a per-example basis as DP-SGD requires. Accommodating the DP-SGD mechanism would therefore necessitate dropping the gradient penalty and reverting to weight-clipping Wasserstein training, sacrificing the architectural property that motivated the WGAN-GP variant in the first place. Second, the standard WGAN-GP training protocol applies five discriminator updates per generator update; because the discriminator is the sole component that directly processes real training records, this ratio implies that the privacy budget is consumed at five times the rate of a comparable setup in which one private gradient computation is performed per logical batch, as is the case in the Pythia-DP configuration. Third, the injection of Gaussian noise into discriminator gradients — the core operation of DP-SGD — is well documented to exacerbate the mode collapse and non-convergence that already characterise GAN training on tabular health data, with empirical results in the DP-GAN literature \cite{Xie2018DPGAN, Jordon2019PATEGAN} demonstrating sharp utility cliffs at privacy budgets of ε ≤ 10. Fourth, the HealthGAN implementation relies on TensorFlow 1 compatibility mode via `tf.compat.v1` with eager execution disabled; the per-example gradient computation required by `tensorflow_privacy` is reported to be five to ten times slower in graph execution mode, making the 100,000-epoch training schedule computationally prohibitive within the resource envelope of this project.

On the scientific side, the thesis already establishes a DP comparison on the LLM arm — Pythia-DP versus non-DP Pythia — which constitutes the central privacy narrative. Introducing a HealthGAN-DP condition would triple the experimental matrix without proportional gain in insight, since DP-GAN performance on tabular healthcare data is already characterised as poor in the established literature \cite{Xie2018DPGAN, Jordon2019PATEGAN}; replicating that outcome would not yield a novel empirical contribution. The more informative comparisons enabled by the chosen design are: (a) LLM-DP versus LLM-non-DP, which isolates the utility cost of the privacy guarantee on the LLM synthesis path; (b) GAN-non-DP versus LLM-non-DP, which provides a clean architecture comparison free of DP confounds; and (c) LLM-DP versus GAN-non-DP, which tests whether a differentially-private LLM generator can match or exceed the utility of a non-private GAN generator — a positive result here constituting a substantive thesis claim. HealthGAN is therefore retained as the non-private utility ceiling, and the differentially-private synthesis path is studied exclusively via the Pythia pipeline.

**Rationale:** The scope restriction was adopted because the WGAN-GP gradient penalty creates a fundamental incompatibility with DP-SGD's per-example clipping that cannot be resolved without abandoning the architectural advantage of the WGAN-GP formulation, and because the scientific question of whether DP-LLM synthesis outperforms non-private GAN synthesis is more novel and more tractable than replicating the well-documented utility degradation of DP-GANs on tabular health data.
