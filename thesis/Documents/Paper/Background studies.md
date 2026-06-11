# Background Studies

Per-paper key findings for every PDF under `thesis/Documents/Paper/`, written to feed
directly into the thesis background chapter and to position the work of this thesis.

**Thesis context (what each paper is measured against).** This thesis is a unified
empirical comparison of **five generators** on the **UCI Diabetes 130-US Hospitals**
dataset — HealthGAN (WGAN-GP, no DP), PATE-GAN ((ε,δ)-DP GAN), Pythia-70M-LoRA and
Pythia-1B-LoRA (LLMs, no DP), and Pythia-1B-DP-LoRA (DP-SGD via Opacus) — evaluated under
two pillars: **utility** (statistical similarity: per-column distributions, correlation
difference, Kolmogorov–Smirnov, PCA manifold overlap; predictive performance: TSTR
classifier, feature-importance preservation, Cox proportional-hazards concordance) and
**privacy** (nearest-neighbour distance, NN adversarial accuracy, membership inference,
attribute inference, outlier linkability). Matched privacy budget **ε = 5.0**.
Research questions: **RQ1** (which metrics suit utility/privacy evaluation), **RQ2**
(GAN vs LLM trade-off), **RQ3** (marginal cost of DP within each family).

> Relevance legend: ⭐⭐⭐ core (directly used / replicated / extended) · ⭐⭐ strong ·
> ⭐ supporting / contextual · ○ tangential.

---

## 0. Quick map: paper → thesis use

| Paper (folder name)                                                   | BibTeX key                  | Role in this thesis                                                       | Rel.   |
| --------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------- | ------ |
| generationa and evaluation of privacy preserving = health-gan         | `healthgan`               | Source of**HealthGAN** generator + NN-distance/NNAA privacy metrics | ⭐⭐⭐ |
| healthgan 1 (Synthesizing Quality Open Data Assets)                   | `yale2020synthesizing`    | HealthGAN export-generator governance model                               | ⭐⭐   |
| healthgan 2 (Medical Time-Series Data Generation)                     | `dash2020medical`         | HealthGAN lineage (time-series)                                           | ⭐     |
| health gan 3 (Investigating time-series resemblance)                  | `bhanot2022investigating` | HealthGAN lineage; resemblance metrics                                    | ⭐     |
| pategan                                                               | `pategan`                 | Source of**PATE-GAN** generator (DP-GAN arm)                        | ⭐⭐⭐ |
| LLM for synthetic health data (Miletic benchmark)                     | `Miletic2024`             | **Template for the entire LLM arm** (Pythia + Tabula + TSTR-RF)     | ⭐⭐⭐ |
| LLM for privacy preservation (SafeSynthDP)                            | `Nahid2024`               | Motivates LLM+DP; contrasted with proper DP-SGD                           | ⭐⭐   |
| Differentially Private Fine-tuning of Language Models                 | `Yu2022DPFT`              | Justifies**DP-LoRA**; "bigger models survive DP better"             | ⭐⭐⭐ |
| Synthetic Data – Anonymisation Groundhog Day                         | `stadler2022synthetic`    | Foundation of the**privacy evaluation** (MIA/AI/linkability)        | ⭐⭐⭐ |
| generation of syn healthcare (Nik)                                    | `nik2023generation`       | GAN-on-tabular-health methodology + TSTR/DCR                              | ⭐⭐   |
| data synthesis based on GAN (table-GAN)                               | `park2018data`            | "model compatibility" privacy–utility framing                            | ⭐     |
| Anonymization…ADS-GAN                                                | `ADSGAN2020`              | Identifiability-constrained GAN; JSD/Wasserstein utility                  | ⭐⭐   |
| Grouped Correlational GAN                                             | `yang2019grouped`         | Discrete-EHR GAN variant                                                  | ○/⭐  |
| continuous patient centric sequence (SC-GAN)                          | `wang2019continuous`      | Coupled sequential EHR GAN                                                | ○/⭐  |
| SenseGen                                                              | —                          | Early LSTM+MDN generative + discriminator test                            | ○     |
| Generation and evaluation of synthetic patient data (Goncalves)       | `goncalves2020generation` | 3-model-class taxonomy + disclosure metrics                               | ⭐     |
| Generating…cross-sectional EHR / UK Primary Care (Wang/Myles/Tucker) | `wang2019generating`      | First combined utility+privacy framework (CPRD)                           | ⭐⭐   |
| privacy preserving Deep NN…clinical (SPRINT_gan)                     | `beaulieu2019privacy`     | Early**DP-GAN** for clinical sharing                                | ⭐⭐   |
| PrivSyn                                                               | `privsyn2021`             | Marginal-based DP synthesis baseline + metrics                            | ⭐⭐   |
| A Scalable and General Approach to DP Synthetic Data                  | (lit. review)               | Query/marginal DP synthesis                                               | ⭐     |
| Optimizing synthesis…sequential trees (El Emam)                      | `emam2021optimizing`      | Non-DL baseline; distinguishability metric                                | ⭐     |
| State of the art in health care domain (Murtaza)                      | `Murtaza2023`             | Organizing taxonomy of the background chapter                             | ⭐⭐⭐ |
| syn tabular health data (Hernandez systematic review)                 | `hernandez2022synthetic`  | Metric catalogue (utility + privacy)                                      | ⭐⭐   |
| A comprehensive survey of synthetic tabular data generation (Shi)     | `shi2025comprehensive`    | Recent survey incl. diffusion + LLM                                       | ⭐⭐   |
| a scoping review of privacy and utility metrics                       | (lit. review)               | Critique of conflicting similarity metrics                                | ⭐⭐   |
| Privacy-preserving healthcare informatics (Chong review)              | —                          | Anonymization vs DP; disclosure taxonomy                                  | ⭐     |
| A Comparative Study of Data Anonymization Techniques                  | `anonymization`           | Why classical anonymization is insufficient                               | ⭐     |
| evaluation of synthetic data (Kaabachi)                               | `syninhospital`           | Closest predecessor: unified utility+privacy framework                    | ⭐⭐   |
| paper on privacy-utility combined chart (Thees)                       | —                          | Multivariate risk-utility visualization (PCA biplots)                     | ⭐     |
| language-models (GPT-2, Radford)                                      | —                          | Foundation of autoregressive LM generation                                | ⭐     |
| Synthetic Data for Text Localisation (Gupta)                          | —                          | Out of scope (vision)                                                     | ○     |

---

## 1. Surveys, reviews, and the privacy-evaluation critique

### 1.1 Synthetic data generation: State of the art in the health care domain — Murtaza et al. (2023) ⭐⭐⭐

`Murtaza2023` · *Computer Science Review* 48:100546.

- **Methodology.** A structured review that organizes the field along two axes: (i) **data
  granularity** — snapshot/cross-sectional, aggregate, time-series, longitudinal, and
  clinical text; and (ii) **model class** — Knowledge-Driven (Synthea, rule documents),
  Data-Driven classical (Bayesian networks, Dynamic BNs, decision-tree synthesizers,
  Prophet/NeuralProphet), Data-Driven deep (GANs/NNs, LSTM/GPT-2 for text), and Hybrids.
- **Achievements / framing.** Matches models to data shapes (e.g. KD ⇒ longitudinal, DD-deep
  ⇒ mixed snapshot/aggregate). Splits evaluation into **realism** (univariate:
  dimension-wise statistics, KS/KL/Hellinger; multivariate: dimension-wise prediction,
  correlation preservation, indistinguishability classifier ≈0.5, latent/distance metrics)
  and **privacy** (membership inference, attribute disclosure, "meaningful identity"
  disclosure) plus combo metrics (adversarial accuracy, identifiability loss).
- **Remarks.** Stresses that **good univariate scores do not guarantee multivariate
  fidelity** (e.g. impossible co-morbidities), so both must be reported.
- **Alignment.** This is the **backbone taxonomy** of the thesis background chapter. The
  thesis's split of utility into statistical similarity + predictive performance, and its
  privacy threat list (membership/attribute/identity), come straight from this framing. The
  univariate-vs-multivariate caveat motivates including correlation-difference and PCA
  manifold overlap alongside per-column KS.

### 1.2 Synthetic data generation for tabular health records: a systematic review — Hernandez et al. (2022) ⭐⭐

`hernandez2022synthetic` · *Neurocomputing*.

- **Methodology.** Systematic review focused specifically on **tabular** health records.
- **Utility metrics catalogued.** TSTR / TRTR / TRTS / TSTS paradigms; accuracy, F1, AUROC.
- **Privacy metrics catalogued.** Simulated identity disclosure (linkage on
  quasi-identifiers), **Distance-to-Closest-Record (DCR)** (smaller ⇒ higher risk),
  attribute-disclosure inference, **membership inference** (e.g. SynTEG), DBSCAN
  proximity/near-duplicate flags, classical SDC re-identification risk, reported DP cost
  (ε/δ), and divergences (JSD/Wasserstein) used as indirect privacy proxies.
- **Alignment.** The thesis's **metric menu for RQ1** is essentially a curated subset of
  this catalogue: TSTR for utility, DCR / NN-distance and membership/attribute inference for
  privacy. Also flags the same hazard the scoping review raises — distances double as utility
  *and* privacy measures.

### 1.3 A Comprehensive Survey of Synthetic Tabular Data Generation — Shi et al. (2024/25) ⭐⭐

`shi2025comprehensive`.

- **Methodology.** Three-part survey: (1) background/pipeline (problem definition,
  generation, post-processing, evaluation), (2) generation methods categorized into
  **traditional**, **diffusion-model**, and **LLM-based**, (3) applications and challenges.
- **Achievements.** Unlike older GAN-only surveys, integrates **diffusion models and LLMs**
  in one view; compares architecture, generation quality, applicability.
- **Remarks.** Open challenges: heterogeneity, fidelity, privacy protection.
- **Alignment.** Justifies treating GANs and LLMs as the two contemporary paradigms (exactly
  the thesis's RQ2 axis) and provides the up-to-date positioning that LLM-based tabular
  synthesis is now a first-class method, not a curiosity.

### 1.4 A scoping review of privacy and utility metrics in medical synthetic data ⭐⭐

(*from `literature review.docx`*)

- **Key argument.** Synthetic data is being treated as a "silver bullet," but malicious
  adversaries can still infer presence/absence of records. The field over-weights utility
  relative to privacy, and synthetic data has not been adequately scrutinized.
- **Findings.** Most common privacy evaluation = **membership inference**, then attribute
  inference, then classification/regression inference, holdout distinguishing, distance to
  real data, record matching, ML-model inference. Privacy is mostly measured with
  **similarity-based metrics that are also used for utility**, producing **conflicting
  results**. Two reasons similarity is inadequate: (i) similarity ≠ privacy guarantee;
  (ii) successful inference attacks can exist even when synthetic data is dissimilar from
  the original. **Differential privacy is the most useful** as it gives strong resilience to
  inference attacks. Distinguishes **broad utility** (univariate/bivariate/multivariate/
  longitudinal similarity) from **narrow utility** (task-specific performance).
- **Remarks.** Highlights gaps: no consensus on metrics, conflicting metrics, no
  standardized privacy guarantees; cites IEEE / Horizon Europe calls for reliable frameworks.
- **Alignment.** This is the **central methodological warning** the thesis answers: it keeps
  statistical-similarity metrics and privacy-attack metrics **separate** (utility pillar vs
  privacy pillar) precisely to avoid the conflation this review condemns, and it includes
  formally DP generators because this review names DP as the strongest defense.

### 1.5 Privacy-preserving healthcare informatics: a review — Chong (2021) ⭐

(no bib key) · *ITM Web of Conferences* 36:04005.

- **Methodology.** Survey of privacy-enhancing methods for EHR sharing, focused on **data
  anonymization** and **differential privacy**.
- **Achievements.** Defines the canonical **privacy-disclosure taxonomy** used throughout the
  literature: (1) **identity disclosure / re-identification**, (2) **attribute disclosure**,
  (3) **membership disclosure**. Contrasts strengths/limits of anonymization vs DP.
- **Alignment.** Source of the three-way disclosure vocabulary the thesis uses to frame its
  privacy attacks; supports the claim that anonymization alone is insufficient.

### 1.6 A Comparative Study of Data Anonymization Techniques — Murthy et al. ⭐

`anonymization` · IEEE.

- **Methodology.** Empirically compares **five anonymization techniques** (suppression,
  generalization, swapping, perturbation, etc.) on the same dataset, on efficiency and
  resource cost.
- **Results.** Suppression most efficient; swapping the slowest and most resource-consuming.
- **Alignment.** Provides the concrete "classical anonymization is brittle / lossy" evidence
  the thesis introduction uses to motivate moving from anonymization to synthesis.

### 1.7 Beyond the Trade-off Curve: Multivariate Risk-Utility Maps — Thees, Müller, Templ ⭐

(*"paper on privacy-utility combined chart"*; no bib key)

- **Methodology.** Compares **six visualization approaches** for simultaneously evaluating
  multiple risk and utility indicators: heatmaps, dot plots, composite scatterplots,
  parallel-coordinate plots, radial profile charts, and **PCA-based biplots**. Introduces
  blockwise PCA for composite scatterplots and joint PCA for biplots; applies systematic
  **Pareto-optimal** method identification.
- **Results.** No single view dominates: **PCA biplots** best reveal multivariate structure;
  composite scatterplots are most intuitive. Recommends combining complementary views.
- **Alignment.** Directly relevant to **how the thesis presents the privacy–utility
  trade-off** across five generators and many metrics. Supports using PCA (the thesis already
  uses PCA manifold overlap for utility) and Pareto framing rather than a single 2-D R-U
  curve when ranking generators for RQ2/RQ3.

---

## 2. GAN-based health-data synthesis

### 2.1 Generation and evaluation of privacy-preserving synthetic health data (HealthGAN) — Yale et al. (2020) ⭐⭐⭐

`healthgan` · *Neurocomputing* 416:244–255.

- **Methodology.** Defines **HealthGAN** and a four-dimensional quality scheme:
  **resemblance, privacy, utility, footprint**. HealthGAN is a **WGAN-GP**–based generator
  (consistent with the thesis's "HealthGAN = WGAN-GP, no formal DP"). Critical workflow:
  (1) train HealthGAN **inside** a secure environment; (2) **export the trained generator
  model** for external users to sample from — so no patient-level data leaves, and
  de-identification (costly, fidelity-lossy, training-heavy) is avoided.
- **Achievements / results.** Compared against **five baseline methods**; HealthGAN matches
  resemblance and utility while delivering the **best privacy and footprint**. Two case
  studies: a classroom data-analysis challenge, and reproduction of **three** medical papers
  using synthetic data. Data, code, and the challenge were released.
- **Privacy metrics.** Nearest-neighbour-based privacy assessment — **nearest-neighbour
  adversarial accuracy (NNAA)** and **distance to closest record** — quantify how close
  synthetic records sit to real ones.
- **Alignment (extensive).** HealthGAN is the thesis's **non-DP GAN generator**, the
  "utility-first, privacy-by-distance" anchor against which formal-DP methods are judged. The
  thesis's privacy pillar (**nearest-neighbour distance, NN adversarial accuracy**) is taken
  directly from this lineage. The export-generator governance pattern is exactly the
  deployment story the thesis cites when arguing GAN synthesis can satisfy data-steward
  constraints — and it sets up RQ3: HealthGAN gives good NN-distance privacy *empirically*
  but offers **no formal (ε,δ) guarantee**, which is the gap PATE-GAN fills.

### 2.2 Synthesizing Quality Open Data Assets from Private Health Research Studies — Yale et al. (2020) ⭐⭐

`yale2020synthesizing` · BIS 2020.

- **Methodology.** Uses HealthGAN to **reproduce the outcomes of two previously published
  studies** (Autism Spectrum Disorder comorbidity analyses) entirely from synthetic data.
  Data live in **OptumLabs Data Warehouse (OLDW)**, a secure environment that forbids
  exporting any patient-level data — so HealthGAN **exports a privacy-preserving generator
  model** instead of data.
- **Achievements / results.** Synthetic data reproduces the real-study findings while
  preserving privacy; evaluated on resemblance / privacy / utility / **efficiency**. Privacy
  includes **membership inference**.
- **Remarks (from `literature review.docx`).** Notes MIMIC-III bias concerns (single Boston
  ICU ⇒ limited generalization). Introduces a novel categorical/ordinal **encoding** for
  generation. Headline governance point: **you can export the generator without the training
  data**, so the real data is never needed again.
- **Alignment.** Reinforces the deployment argument behind choosing HealthGAN, and supplies
  the **membership-inference** and **efficiency** angles the thesis evaluates.

### 2.3 Medical Time-Series Data Generation using GANs — Dash, Yale, Guyon, Bennett (2020) ⭐

`dash2020medical` · AIME 2020.

- **Methodology.** Workflow that adapts existing time-series generative models to medical
  longitudinal data where **static covariates** (age, gender, comorbidities) influence
  temporal values.
- **Results.** Higher resemblance and utility than a state-of-the-art benchmark baseline.
- **Alignment.** Same research group as HealthGAN; shows the lineage's extension to
  longitudinal data. Tangential to the thesis (which is cross-sectional tabular) but supports
  the resemblance-metric methodology.

### 2.4 Investigating synthetic medical time-series resemblance — Bhanot et al. (2022) ⭐

`bhanot2022investigating` · *Neurocomputing* 494:368–378.

- **Methodology.** Proposes **four time-series resemblance metrics** to quantitatively
  evaluate real-vs-synthetic similarity, replacing subjective covariate-plot inspection.
- **Results.** Metrics effectively capture resemblance; reveal varying resemblance across
  covariate subgroups and multivariate series.
- **Alignment.** Contributes to the resemblance-metric toolbox (HealthGAN lineage). Marginal
  for cross-sectional tabular but cited for completeness of the evaluation discussion.

### 2.5 Generation of Synthetic Tabular Healthcare Data Using GANs — Nik et al. (2023) ⭐⭐

`nik2023generation` · SimulaMet / Univ. of Stavanger.

- **Methodology.** Benchmarks **TGAN, CTGAN, CTABGAN, and WGAN-GP** across four datasets of
  differing shape: **Epileptic Seizure Recognition (EEG→tabular, 11,500×178 integers)**,
  **Diabetes (Health Facts, ~89K×29, mixed/multimodal)**, **Thyroid (UCI, imbalanced
  categoricals)**, and **MIMIC-III (derived tabular, ~41K×14, long-tailed)**.
- **Evaluation.** Utility = **indistinguishability** (LR/SVM real-vs-synthetic classifier,
  reported as normalized AUROC = 1 − AUROC) and **cross-testing TSTR** (DT/RF/LR/MLP trained
  on real vs synthetic, evaluated on held-out real, |Δ Macro-F1|). Privacy = **Distance to
  Closest Record (DCR)** (prefer larger mean, small std). Treats privacy–utility as a
  trade-off; DP discussed as a direction but not used.
- **Results.** All models can produce synthetic data preserving statistical characteristics,
  model compatibility, and privacy; no architecture is uniformly best.
- **Alignment (extensive).** This is the **closest GAN-side methodological sibling** of the
  thesis: same domain (incl. a diabetes dataset), same TSTR + indistinguishability + DCR
  evaluation logic, and the explicit privacy–utility-trade-off framing. The thesis differs
  by (a) standardizing on **Diabetes 130-US Hospitals**, (b) actually adding **formal DP**
  (PATE-GAN, DP-SGD) instead of leaving it as future work, and (c) adding LLM generators.

### 2.6 Data Synthesis based on GANs (table-GAN) — Park et al. (2018) ⭐

`park2018data` · *PVLDB* 11(10):1071–1083.

- **Methodology.** **table-GAN** synthesizes whole relational tables; introduces **model
  compatibility** — ML models trained on synthetic must perform like models trained on real
  for unseen test cases — and argues anonymization/perturbation without model compatibility
  is "of little value." Offers low-privacy and high-privacy settings.
- **Results.** Across four datasets (incl. a Health set), only table-GAN consistently
  balances privacy level and model compatibility, beating anonymization, perturbation, and
  DCGAN/condensation.
- **Alignment.** Early articulation of the **TSTR / model-compatibility** idea the thesis
  operationalizes, and an early explicit privacy-level knob (precursor to formal DP budgets).

### 2.7 Anonymization Through Data Synthesis using GANs (ADS-GAN) — Yoon et al. (2020) ⭐⭐

`ADSGAN2020` · *IEEE J. Biomed. Health Inform.* 24(8):2378–2388.

- **Methodology.** **Conditional WGAN-GP + identifiability loss.** Privacy is enforced by an
  **ε-identifiability constraint**: every synthetic record must be "different enough" from
  any real record, measured by a **weighted Euclidean distance** that up-weights
  rare/identifying features (weights = inverse feature entropies; continuous variables
  quantized).
- **Datasets.** MAGGIC (heart failure, 30,389×29), UNOS transplant registries (1-year
  mortality labels).
- **Evaluation.** Feature-wise distribution matching (Student's t for continuous, χ² for
  binary); joint-distribution similarity via **Jensen–Shannon Divergence and Wasserstein
  distance** (estimated with small NNs, lower better); downstream **AUROC** of a classifier
  trained on synthetic, tested on real, at fixed identifiability (e.g. 0.1).
- **Remarks.** Notes that de-identification (e.g. MIMIC-III) is insufficient against linkage,
  and that **DP noise can hurt fidelity** in practice (DP-GAN / PATE-GAN).
- **Alignment.** A second formal privacy mechanism (identifiability) alongside DP — useful
  contrast for RQ3, and the JSD/Wasserstein/feature-test utility metrics overlap with the
  thesis's statistical-similarity pillar. The "DP hurts fidelity" remark is exactly the
  marginal-cost-of-DP question the thesis quantifies.

### 2.8 Grouped Correlational GAN for Discrete EHR (GcGAN) — Yang et al. (2019) ⭐/○

`yang2019grouped` · IEEE BIBM 2019.

- **Methodology.** Embeds **treatment efficacy** into disease diagnosis, then learns
  **inter-group correlations** with grouped variables; adds dense connections to strengthen
  the generator.
- **Results.** Matches real-data distribution statistics; boosts **multi-label treatment
  recommendation** via augmentation, beating SOTA; automatically separates disease-specific
  from adjuvant drugs (interpretability).
- **Alignment.** Discrete-EHR variant; mainly evidence that GANs capture cross-variable
  correlation structure — the property the thesis's **correlation-difference** metric tests.

### 2.9 Continuous Patient-Centric Sequence Generation (SC-GAN) — Wang, Zhang, He (2019) ⭐/○

`wang2019continuous` · DASFAA 2019, LNCS 11447:36–52.

- **Methodology.** **Sequentially Coupled GAN**: two coupled generators that jointly produce
  **patient state** and **medication dosage** time series, modeling their mutual influence
  (dosage depends on current state; next state depends on previous state + dosage). Uses
  feature-matching loss; discriminator classifies real/synthetic per time step.
- **Datasets / baselines.** MIMIC-III sepsis and diabetes cohorts; compared vs SeqGAN,
  C-RNN-GAN, RCGAN, and an Imitation-RNN.
- **Results.** Synthetic data from SC-GAN yields better downstream task performance than the
  comparison generators.
- **Alignment.** Longitudinal/coupled approach; out of the thesis's cross-sectional scope but
  useful background on EHR-GAN diversity and on TSTR-style downstream evaluation.

### 2.10 Generation and Evaluation of Synthetic Patient Data — Goncalves et al. (2020) ⭐

`goncalves2020generation` · *BMC Medical Research Methodology* 20:108.

- **Methodology.** Evaluates **three classes** of generators: probabilistic models,
  **classification-based imputation** models, and **GANs** (medGAN-style). Demonstrated on
  **SEER** cancer registry data (breast/respiratory/non-solid, >360K cases, 2010–2015).
- **Achievements.** Presents and discusses utility + **information-disclosure** metrics, with
  guidance on method/metric trade-offs for medical synthetic data.
- **Alignment.** Reinforces that GANs are one option among several, and supplies the
  disclosure-risk framing; a useful comparator for the thesis's "which family balances best."

### 2.11 Generating and Evaluating (Cross-sectional / UK Primary Care) Synthetic Healthcare Data — Wang, Myles, Tucker (2019/2021) ⭐⭐

`wang2019generating` · IEEE conference + *Computational Intelligence* 37:819–851 (journal).

- **Methodology.** Proposes a framework (from **CPRD / MHRA**, the UK regulator) to generate
  and evaluate synthetic healthcare data that **simultaneously preserves data complexity and
  privacy** — described as the first framework with that joint aim. Uses Bayesian-network–
  style generation; privacy via identity-disclosure / DBSCAN proximity checks.
- **Datasets.** Indian Liver Patient dataset and **UK primary care (CPRD)** data (15% of the
  UK population, 17,400+ event types).
- **Achievements.** Concrete requirements list for synthetic data used to **benchmark ML
  algorithms and reveal real-world data bias**; demonstrated across scenarios.
- **Alignment.** A regulator-grade precedent for the thesis's central premise — a single
  framework reporting **both** utility and privacy — and a model for stating data-steward
  requirements in the discussion/recommendations.

### 2.12 Privacy-Preserving Generative Deep Neural Networks Support Clinical Data Sharing (SPRINT_gan) — Beaulieu-Jones et al. (2019) ⭐⭐

`beaulieu2019privacy` · *Circ. Cardiovasc. Qual. Outcomes* 12:e005122.

- **Methodology.** Pairs of deep networks (an AC-GAN) generate synthetic trial participants
  resembling the **SPRINT** (Systolic Blood Pressure) trial; crucially, the networks are
  trained with **differential privacy (DP-SGD)** to bound the chance that querying synthetic
  participants identifies a real one.
- **Results.** ML predictors built on the synthetic population **generalize to the real
  data**, showing synthetic data can support hypothesis-generating reanalysis. Code released
  (`greenelab/SPRINT_gan`).
- **Alignment (extensive).** The **canonical early DP-GAN for clinical data** — direct
  precedent for the thesis's DP-GAN arm (PATE-GAN) and for the methodological choice of
  training generators under formal DP. It establishes the very claim RQ3 stress-tests: that
  DP-trained generators can retain enough utility for downstream ML.

---

## 3. Differential-privacy-based synthesis (GAN and non-GAN)

### 3.1 PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees — Jordon, Yoon, van der Schaar (2019) ⭐⭐⭐

`pategan` · ICLR 2019.

- **Methodology.** Adapts the **PATE** (Private Aggregation of Teacher Ensembles) framework
  to GANs. The data is split into disjoint partitions, one per **teacher-discriminator**.
  A **student-discriminator** is trained only on generated samples labelled by a **noisy
  aggregation of the teachers' votes** — the noise on the aggregation is what provides the
  **(ε,δ)-DP** guarantee. The generator is trained against the student. Because only the
  student (trained on already-private labels) touches the generator, the PATE moments
  accountant gives **tight** DP bounds.
- **New evaluation idea.** Proposes that for synthetic data to be useful, the **relative
  ranking of two algorithms** trained/tested on synthetic should match their ranking on real
  data (later called **Synthetic Ranking Agreement**) — beyond raw accuracy.
- **Baselines / results.** Compared against **DPGAN** (Xie et al. 2018, which adds noise to
  discriminator gradients) at the **same** DP level, and an (∞,∞) non-private GAN as an upper
  bound. At (ε,δ)=(1, 10⁻⁵), PATE-GAN **consistently outperforms DPGAN** in
  train-on-synthetic/test-on-real AUROC/AUPRC across 12 predictive models and multiple
  (including medical) datasets.
- **Alignment (extensive).** PATE-GAN is the thesis's **DP-GAN generator** and the formal-DP
  counterpart to HealthGAN. It supplies the **family-appropriate DP mechanism for GANs**
  (teacher/student aggregation), contrasted in RQ3 with **DP-SGD** for the LLM arm. The
  thesis runs it at **ε = 5.0** and uses TSTR-style downstream evaluation in the same spirit
  as PATE-GAN's ranking-agreement argument. The HealthGAN-vs-PATE-GAN pair *is* the GAN-side
  "marginal cost of formal DP" experiment.

### 3.2 PrivSyn: Differentially Private Data Synthesis ⭐⭐

`privsyn2021`.

- **Methodology.** Marginal-based DP synthesis using **PrivBayes** (DP Bayesian networks) and
  **PGM** (probabilistic graphical models / Markov random fields) — **not** GAN/LLM. Privacy
  via the **Gaussian mechanism** with **zCDP** composition; δ = 1/n², ε swept.
- **Evaluation.** **Pairwise-marginal release** (average L1 error over all attribute pairs),
  **3-attribute range queries** (mean absolute error over 1000 random queries — where PrivSyn
  gains most over graphical-model baselines), and **downstream classification** (SVM trained
  on synthetic, misclassification on real; bounded by Majority and NonPriv references).
- **Results.** PrivSyn matches or beats marginal-based baselines and gets closest to NonPriv
  on downstream tasks; larger ε ⇒ less noise ⇒ lower L1 / better classification.
- **Alignment.** A **non-deep DP baseline family** and a source of marginal/range-query
  utility metrics; useful contrast to deep generators and a concrete example of the
  ε-vs-utility curve the thesis characterizes at ε = 5.0.

### 3.3 A Scalable and General Approach to Differentially Private Synthetic Data ⭐

(*from `literature review.docx`*)

- **Methodology.** Shifts focus from learning the whole joint distribution to **query-based**
  synthesis in a 3-step workflow — **query selection → query measurement → synthetic data
  generation** — so the output preserves measured statistics and **overcomes high
  dimensionality**.
- **Evaluation.** (i) 3-way marginal accuracy, (ii) high-order conjunction accuracy, (iii)
  domain-specific (income-inequality) statistics.
- **Alignment.** Background for the "measure-then-generate" school of DP synthesis;
  contextualizes why deep generators (thesis's choice) are attractive for mixed-type tabular
  health data without manual query design.

### 3.4 Optimizing the Synthesis of Clinical Trial Data Using Sequential Trees — El Emam, Mosquera, Zheng (2021) ⭐

`emam2021optimizing` · *JAMIA* 28(1):3–13.

- **Methodology.** Synthesizes data with **sequential decision trees** and studies how
  **variable order** affects utility. Implements **particle-swarm optimization** with a
  **distinguishability hinge loss** to find a good order (hinge threshold chosen to avoid
  over-fitting, which would create a privacy problem); compared to a curriculum-learning
  ordering. Six oncology clinical-trial datasets.
- **Results.** Utility variability with order grows as the number of variables grows;
  particle swarm + hinge loss gives adequate utility across all six datasets and beats
  curriculum learning.
- **Alignment.** A strong **non-GAN, non-LLM baseline paradigm** (sequential trees, the
  industry-standard from Replica Analytics) and an explicit **distinguishability** privacy/
  utility metric the thesis can reference when justifying its own metric choices.

---

## 4. LLM-based synthesis

### 4.1 Large Language Models for Synthetic Tabular Health Data: A Benchmark Study — Miletic & Sariyar (2024) ⭐⭐⭐

`Miletic2024` · *Studies in Health Technology and Informatics* (MIE 2024).

- **Methodology.** Benchmarks the **Pythia scaling suite (14M, 31M, 70M, 160M, 1B)** for
  tabular synthesis via the **Tabula framework** (a development of **GReaT**), which
  serializes each table **row as text** and models the joint distribution autoregressively;
  uses **token-sequence compression and left padding**; sampling is conditioned on the binary
  target as a **start token** ("Class 0"/"Class 1"). GAN competitor = **CTGAN**; also a
  random forest trained on the **Original** data as reference. Each Pythia model is run both
  **randomly initialized** (pretraining from scratch) and **pretrained-then-fine-tuned**.
- **Datasets.** CDC Diabetes Health Indicators (253,680×22), Adult (32,561×15), Smoking &
  Drinking (991,346×19). Train sizes **500/1000/2500/5000**, matched test sizes.
- **Evaluation (utility only).** **Train-on-synthetic, test-on-real (TSTR)**: a **random
  forest** trained on synthetic, evaluated on real, **mean accuracy ± SD over 100 runs** per
  (model × dataset × size). **No privacy evaluation at all.**
- **Results.** As parameters increase, LLMs **surpass CTGAN** on all three datasets; even the
  14M model is comparable to CTGAN; some LLM variants slightly beat the Original. Positive
  correlation between **training-set size** and utility. For **categorical-heavy** datasets
  (CDI, SDD) extra parameters help little ⇒ small LLMs can suffice. The literature claim that
  **random-init beats pretrained could not be confirmed**. LLMs do better than CTGAN with
  **limited** data. Costs: 1B needed ~40 GB VRAM across two GPUs; even 14M needed 22 GB.
  Target-as-start-token makes models task-specific (would need feature permutation to be
  general). Future work: **diffusion models**.
- **Alignment (very extensive).** This paper is the **direct blueprint for the thesis's LLM
  arm**: same Pythia family, same row-as-text (Tabula/GReaT) serialization, same TSTR-with-
  random-forest utility protocol. The thesis **extends Miletic on four axes that he explicitly
  leaves open**: (1) standardizes on **Diabetes 130-US Hospitals**; (2) adds **LoRA**
  parameter-efficient fine-tuning (Pythia-70M-LoRA, Pythia-1B-LoRA); (3) adds **formal
  differential privacy** (Pythia-1B-DP-LoRA via DP-SGD/Opacus) — Miletic has none; (4) most
  importantly, **adds a full privacy evaluation** (membership/attribute inference, NN
  distance, outlier linkability), filling Miletic's biggest gap. Miletic's "bigger Pythia is
  better, but gains saturate, and big models are hardware-hungry" directly informs the
  thesis's choice of the **70M and 1B** scale anchors and its observation about DP-SGD failing
  on Pythia-70M but working on Pythia-1B.

### 4.2 SafeSynthDP: LLMs for Privacy-Preserving Synthetic Data Generation Using Differential Privacy — Nahid & Hasan (2024) ⭐⭐

`Nahid2024` · Univ. of Alberta.

- **Methodology.** Uses LLMs (**GPT-4o-mini, Gemini 1.5 Flash**, plus MNB/GRU/LSTM
  downstream) to generate synthetic data with **DP noise injection** — Laplace/Gaussian noise
  described as **perturbing token/word frequencies** — and calibrates **ε** to trade privacy
  for utility. Dataset: **AGNews** (news text classification, not tabular health).
- **Evaluation.** Utility = downstream classification **accuracy**; privacy = resilience to
  **membership-inference** attacks, plus the ε hyperparameter itself.
- **Results / remarks.** Reports a viable privacy–utility balance. **Critique (per
  `summaries.docx`):** the privacy "guarantee" is essentially *add more noise, measure
  accuracy* — ε governs noise magnitude but the scheme is **not formal DP-SGD**, so the
  privacy claim is weaker than it appears.
- **Alignment (extensive).** The clearest motivation for the **LLM + DP** combination the
  thesis investigates — but also a **cautionary contrast**: the thesis implements **formal
  (ε,δ)-DP via DP-SGD (Opacus)** during fine-tuning rather than post-hoc frequency
  perturbation, and runs **multiple attack types** rather than accuracy-as-privacy. SafeSynthDP
  is therefore "what to improve on" for the thesis's Pythia-1B-DP design.

### 4.3 Differentially Private Fine-tuning of Language Models — Yu et al. (2022) ⭐⭐⭐

`Yu2022DPFT` · Microsoft / ICLR 2022.

- **Methodology.** A **meta-framework for DP fine-tuning** built on **parameter-efficient**
  adapters — including **LoRA** and Adapters — combined with **DP-SGD**, giving simpler,
  sparser, faster private training than full-model DP fine-tuning.
- **Results.** State-of-the-art privacy–utility on NLP: e.g. **RoBERTa-Large 87.8% on MNLI at
  ε=6.7** (vs 90.2% non-private); GPT-2 NLG BLEU within a few points of non-private at
  ε≈6.8. DP-LoRA's accuracy spread across hyperparameters is only ~2%.
- **Headline finding.** **Larger models are better suited for private fine-tuning** — they not
  only reach higher accuracy non-privately but **better retain accuracy when DP is
  introduced**.
- **Alignment (very extensive).** This paper is the **methodological and empirical
  justification for the thesis's Pythia-1B-DP-LoRA design**: it validates **combining LoRA
  with DP-SGD** as the right way to fine-tune LMs privately, and its central result —
  *bigger models survive DP better* — is exactly why the thesis adds the **Pythia-1B** scale
  anchor and the **Pythia-1B-DP** variant after observing that **DP-SGD on Pythia-70M fails to
  converge usefully at ε = 5.0**. In short, Yu et al. predict the thesis's RQ3 outcome for the
  LLM family.

### 4.4 Language Models are Unsupervised Multitask Learners (GPT-2) — Radford et al. (2019) ⭐

(*"language-models"*; no bib key — foundational)

- **Methodology.** A 1.5B-parameter Transformer trained on WebText; demonstrates that
  large autoregressive LMs perform many tasks **zero-shot**, with capability scaling
  log-linearly with model size.
- **Alignment.** The **architectural foundation** for LLM-based tabular synthesis: Pythia and
  the row-as-text paradigm rest on exactly this decoder-only autoregressive-Transformer
  design. Background for §"The Pythia Model Family" and "Row-as-Text."

---

## 5. The privacy critique and unified evaluation frameworks

### 5.1 Synthetic Data – Anonymisation Groundhog Day — Stadler, Oprisanu, Troncoso (2022) ⭐⭐⭐

`stadler2022synthetic` · USENIX Security 2022 · `github.com/spring-epfl/synthetic_data_release`.

- **Methodology.** A **critical** study that builds an open-source framework to **quantify
  privacy gain**, implementing **membership-inference (MIA)** and **attribute-inference (AI)**
  attacks with model adapters. Tests non-DP and DP synthesizers (DataSynthesizer BayNet/
  PrivBay, PATE-GAN) at ε ∈ {10, 1, 0.1}.
- **Key findings.** (1) **Non-DP synthetic data does not protect outlier records** from
  linkage/inference. (2) It is **impossible to predict** which characteristics a generator
  will preserve vs suppress. (3) **DP helps — but only if implemented correctly, and at a
  utility cost** (DP can worsen summary-statistic error by orders of magnitude). (4)
  Attribute-inference results vary by target attribute. Defines a per-record, **outlier-
  focused utility "advantage"** showing synthetic data (esp. DP) suppresses the very
  outlier signals needed for anomaly/fraud/rare-phenotype analysis.
- **Alignment (extensive).** This is the **foundation of the thesis's privacy pillar.** The
  thesis's **membership inference, attribute inference, and outlier linkability** metrics come
  straight from this framework, and its core thesis — *synthetic ≠ private unless DP is done
  right, and DP costs utility* — **is precisely what RQ3 measures** for both families. It also
  justifies *why* DP variants must be in the comparison at all.

### 5.2 Generation and Evaluation of Synthetic Data in a University Hospital Setting — Kaabachi et al. (2022) ⭐⭐

`syninhospital` · CHUV / EPFL / Charité · *Stud. Health Technol. Inform.* (MIE 2022).

- **Methodology.** Proposes a **unified, modular framework jointly measuring utility and
  privacy** for EHR synthetic data (tabular and longitudinal, discrete + numeric). Improves
  **SynTEG** (SOTA longitudinal GAN) and uses **CTGAN** for tabular; **adapts Stadler et al.'s
  privacy-gain** evaluation to model attackers with varying partial background knowledge,
  measuring **membership- and attribute-inference risk per patient record**. Organizes utility
  metrics into six categories (correlations, distributions, temporal patterns). Demonstrated on
  Texas hospital-discharge data with an interactive dashboard.
- **Alignment (extensive).** This is the **closest direct predecessor** to the thesis's
  contribution: a single framework reporting **both** pillars, built on the same Stadler
  privacy foundation. The thesis distinguishes itself by (a) a **head-to-head five-generator
  benchmark across the GAN↔LLM and DP↔non-DP axes** (Kaabachi compares two GANs), (b) the
  **Diabetes 130-US** standardization, and (c) adding **Cox concordance / feature-importance**
  predictive-utility metrics. Strong "we extend X" citation for the methodology chapter.

---

## 6. Tangential / out of scope

### 6.1 SenseGen: A Deep Learning Architecture for Synthetic Sensor Data Generation — Alzantot, Chakraborty, Srivastava (2017) ○

(no bib key) · PerCom Workshops 2017.

- **Methodology.** Generator = stacked **LSTM + Mixture Density Network**; a separate **LSTM
  discriminator** distinguishes real vs synthetic — an early "pass the deep-learning
  discriminator test" idea on **accelerometer sensor traces**.
- **Relevance.** Pre-GAN-era precedent for **discriminator-based indistinguishability** as a
  quality test (mirrors the thesis's NN-adversarial / distinguishability logic), but the
  domain (mobile sensor time series) is outside medical tabular synthesis.

### 6.2 Synthetic Data for Text Localisation in Natural Images — Gupta, Vedaldi, Zisserman (2016) ○

(no bib key) · CVPR 2016.

- **Methodology.** A rendering engine overlays synthetic text onto natural images
  (respecting 3-D scene geometry) to train an FCRN text detector; 84.2% F-measure on ICDAR
  2013.
- **Relevance.** **Out of scope** (computer vision, not tabular/medical). Only general
  evidence that training on synthetic data can match or beat real-data training — a one-line
  motivational citation at most.

---

## 7. Relevant papers NOT in the downloaded folder

These are works that matter to this thesis but have **no PDF in `thesis/Documents/Paper/`**.
Two buckets: (7.1) already cited in the thesis `.bib` (foundational methods, attacks, and the
dataset) and (7.2) cross-referenced by the folder papers and worth adding.

**Status flags:** ❌ = not in folder · ✅cite = already in thesis `.bib` · ➕sug = suggested
addition (not yet in `.bib`).

### 7.1 Cited by the thesis but not downloaded (foundational + dataset) — ❌ / ✅cite

These define the building blocks of the five generators and the evaluation, so each deserves a
short background paragraph even though no PDF is present. Sourcing them is recommended.

| Citation                                                                                    | Key                       | What it is                                | Why it matters here                                            | Flag      |
| ------------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------- | -------------------------------------------------------------- | --------- |
| Goodfellow et al.,*Generative Adversarial Nets*, NeurIPS 2014                             | `Goodfellow2014GAN`     | The original GAN                          | Base of HealthGAN / PATE-GAN / all GAN baselines               | ❌ ✅cite |
| Arjovsky et al.,*Wasserstein GAN*, ICML 2017                                              | `Arjovsky2017WGAN`      | WGAN (Earth-Mover loss)                   | Stabilizes GAN training; precursor to WGAN-GP                  | ❌ ✅cite |
| Gulrajani et al.,*Improved Training of WGANs*, NeurIPS 2017                               | `Gulrajani2017WGANGP`   | **WGAN-GP** (gradient penalty)      | **HealthGAN is WGAN-GP** — this is its core objective   | ❌ ✅cite |
| Choi et al.,*medGAN*, MLHC 2017                                                           | `Choi2017MedGAN`        | First discrete-EHR GAN                    | Canonical EHR-GAN baseline cited by most folder GAN papers     | ❌ ✅cite |
| Xu et al.,*Modeling Tabular Data using Conditional GAN (CTGAN)*, NeurIPS 2019             | `Xu2019CTGAN`           | **CTGAN**                           | The GAN baseline in Miletic, Nik, Kaabachi; mode-specific norm | ❌ ✅cite |
| Xie et al.,*Differentially Private GAN (DPGAN)*, arXiv 2018                               | `Xie2018DPGAN`          | DP via noisy discriminator gradients      | The baseline**PATE-GAN beats** at equal ε               | ❌ ✅cite |
| Esteban et al.,*Real-valued (Medical) Time Series with RCGAN*, 2017                       | `Esteban2017TSTR`       | RCGAN + coined**TSTR**              | Origin of the train-on-synthetic/test-on-real protocol         | ❌ ✅cite |
| Dwork et al.,*Calibrating Noise to Sensitivity*, TCC 2006                                 | `Dwork2006DP`           | Definition of differential privacy        | Formal basis of (ε,δ)-DP used throughout RQ3                 | ❌ ✅cite |
| Dwork & Roth,*Algorithmic Foundations of DP*, 2014                                        | `Dwork2014Foundations`  | DP textbook                               | Composition theorems, mechanisms                               | ❌ ✅cite |
| Abadi et al.,*Deep Learning with Differential Privacy (DP-SGD)*, CCS 2016                 | `Abadi2016DPSGD`        | **DP-SGD + moments accountant**     | The mechanism behind**Pythia-1B-DP** (via Opacus)        | ❌ ✅cite |
| Mironov,*Rényi Differential Privacy*, CSF 2017                                           | `Mironov2017RDP`        | RDP accounting                            | Tighter ε composition used by Opacus                          | ❌ ✅cite |
| Papernot et al.,*Semi-supervised Knowledge Transfer (PATE)*, ICLR 2017                    | `Papernot2017PATE`      | **PATE** teacher/student            | The framework**PATE-GAN** adapts                         | ❌ ✅cite |
| Yousefpour et al.,*Opacus*, 2021                                                          | `Yousefpour2021Opacus`  | PyTorch DP-SGD library                    | The**tool** used to train Pythia-1B-DP-LoRA              | ❌ ✅cite |
| Biderman et al.,*Pythia*, ICML 2023                                                       | `Biderman2023Pythia`    | **Pythia model suite (14M–1B)**    | The exact LLMs the thesis fine-tunes                           | ❌ ✅cite |
| Hu et al.,*LoRA*, ICLR 2022                                                               | `Hu2022LoRA`            | **Low-Rank Adaptation**             | The PEFT method for all Pythia-LoRA variants                   | ❌ ✅cite |
| Borisov et al.,*Language Models are Realistic Tabular Data Generators (GReaT)*, ICLR 2023 | `Borisov2023GReaT`      | **Row-as-text** serialization       | The paradigm (Tabula extends it) for the LLM arm               | ❌ ✅cite |
| Shokri et al.,*Membership Inference Attacks*, IEEE S&P 2017                               | `Shokri2017MIA`         | **MIA**                             | The membership-inference attack in the privacy pillar          | ❌ ✅cite |
| Carlini et al.,*Extracting Training Data from LLMs*, USENIX 2021                          | `Carlini2021Extracting` | LLM memorization/leakage                  | Motivates**why LLM synthesis needs DP**                  | ❌ ✅cite |
| Li et al.,*LLMs Can Be Strong Differentially Private Learners*, ICLR 2022                 | `Li2022LargeDP`         | Large-batch DP fine-tuning                | Companion to Yu et al.; "scale helps DP" — supports RQ3       | ❌ ✅cite |
| Cox,*Regression Models and Life-Tables*, JRSS-B 1972                                      | `Cox1972PH`             | **Cox proportional hazards**        | Basis of the Cox concordance utility metric                    | ❌ ✅cite |
| Strack et al.,*Impact of HbA1c Measurement…70,000 records*, BioMed Res. Int. 2014        | `strack2014impact`      | The**Diabetes 130-US** source study | Defines the readmission task / target the thesis uses          | ❌ ✅cite |
| Strack et al.,*Diabetes 130-US Hospitals 1999–2008*, UCI 2014                            | `diabetes_db`           | The**dataset itself**               | The single benchmark dataset of the thesis                     | ❌ ✅cite |
| Johnson et al.,*MIMIC-III*, Sci. Data 2016                                                | `mimic`                 | ICU EHR database                          | Referenced for longitudinal/EHR context (not the main dataset) | ❌ ✅cite |

### 7.2 Suggested additions — cross-referenced by folder papers, not yet in `.bib` — ❌ / ➕sug

Worth sourcing and citing; each is referenced by a paper already in the folder.

| Citation                                                                                                             | What it is                                                                      | Referenced by (in folder)               | Why add it                                                                                                               | Flag     |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------- |
| Zhao, Birke, Chen,*TabuLa: Harnessing Language Models for Tabular Data Synthesis*, arXiv:2310.12746, 2023          | **Tabula** framework (extends GReaT; token-seq compression, left padding) | Miletic2024                             | **The exact LLM serialization framework** the thesis's pipeline follows — currently uncited despite being central | ❌ ➕sug |
| Kotelnikov et al.,*TabDDPM: Modelling Tabular Data with Diffusion Models*, ICML 2023                               | Diffusion model for tabular data                                                | Miletic2024, Shi                        | The third paradigm (diffusion) named as future work by both                                                              | ❌ ➕sug |
| Lu, Shen, Wang, van Rechem, Wei,*Machine Learning for Synthetic Data Generation: A Review*, arXiv:2302.04062, 2023 | Broad ML-synthesis review                                                       | Miletic2024                             | Up-to-date cross-paradigm survey complementing Murtaza/Hernandez                                                         | ❌ ➕sug |
| Yoon, Jarrett, van der Schaar,*Time-series GAN (TimeGAN)*, NeurIPS 2019                                            | SOTA time-series GAN                                                            | dash2020medical, healthgan-2            | The benchmark baseline for the HealthGAN time-series lineage                                                             | ❌ ➕sug |
| Zhang, Yan, Lasko, Sun, Malin,*SynTEG*, JAMIA 2020                                                                 | Temporal structured EHR GAN                                                     | Kaabachi (`syninhospital`), Hernandez | The longitudinal generator Kaabachi's framework is built on                                                              | ❌ ➕sug |
| Baowaly, Lin, Liu, Chen,*Synthesizing EHRs using improved GANs (medBGAN/medWGAN)*, JAMIA 2019                      | Improved discrete-EHR GANs                                                      | Shi, Murtaza                            | Standard EHR-GAN baselines beyond medGAN                                                                                 | ❌ ➕sug |
| Zhao et al.,*CTAB-GAN* (ACML 2021) / *CTAB-GAN+* (2022)                                                          | Mixed-type tabular GAN                                                          | nik2023generation                       | A GAN baseline Nik benchmarks; mixed-type handling                                                                       | ❌ ➕sug |
| Yuan, Zhou, Yu,*EHRDiff*, arXiv:2303.05656, 2023 (and He et al., *MedDiff*, 2023)                                | Diffusion EHR synthesis                                                         | Shi                                     | Concrete diffusion-for-EHR references for the future-work section                                                        | ❌ ➕sug |
| Tramèr & Boneh,*Differentially Private Learning Needs Better Features (or Much More Data)*, ICLR 2021             | Why DP costs accuracy                                                           | Yu2022DPFT                              | Explains the utility cost RQ3 measures; the "DP needs more data" thesis                                                  | ❌ ➕sug |

> **Action items.** (a) Add `Borisov2023GReaT` is present but **Tabula is missing** — add it, it is
> the framework actually used. (b) The DP-vs-utility narrative would be tighter with
> Tramèr & Boneh. (c) If diffusion is mentioned in future work, cite TabDDPM + EHRDiff.

---

## 8. How the corpus maps onto the thesis chapters

- **§ GAN-based approaches** ← `healthgan`, `yale2020synthesizing`, `dash2020medical`,
  `bhanot2022investigating`, `nik2023generation`, `park2018data`, `ADSGAN2020`,
  `yang2019grouped`, `wang2019continuous`, `goncalves2020generation`, `wang2019generating`,
  `beaulieu2019privacy`.
- **§ LLM-based approaches** ← `Miletic2024` (template), GPT-2 (foundation), `Borisov2023GReaT`/
  Tabula (row-as-text), `Biderman2023Pythia`, `Hu2022LoRA`.
- **§ Differential privacy** ← `pategan` (+ `Papernot2017PATE`), `beaulieu2019privacy`,
  `privsyn2021`, "A Scalable and General Approach…", `Nahid2024`, `Yu2022DPFT`,
  `Abadi2016DPSGD`/`Yousefpour2021Opacus`, `Mironov2017RDP`.
- **§ Evaluation frameworks** ← `stadler2022synthetic`, `syninhospital` (Kaabachi, unified),
  Thees (R-U maps), scoping review, `hernandez2022synthetic`, `Murtaza2023`,
  `shi2025comprehensive`.
- **§ Why not classical anonymization** ← `anonymization`, Chong review, `stadler2022synthetic`.

**Net positioning of the thesis.** No prior work runs a **single, reproducible, two-pillar
benchmark** that crosses **GAN vs LLM** *and* **DP vs non-DP** on one medical tabular dataset.
Miletic (LLM) and Nik (GAN) cover one family each with **utility-only** or limited privacy;
Stadler and Kaabachi build the privacy/eval machinery but on GANs/graphical models; PATE-GAN,
SPRINT_gan, and Yu et al. supply the DP mechanisms for each family. This thesis stitches these
threads into one comparison and reports the **marginal cost of formal DP within each
generative family** at a matched ε = 5.0.
