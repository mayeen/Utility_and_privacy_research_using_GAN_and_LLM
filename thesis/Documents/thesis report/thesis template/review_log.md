# Thesis Review Log

Lean tracker: status · action backlog · decisions · citations. One line each, no prose.
**Status:** ☐ not reviewed · 🔄 in progress · ✅ reviewed · ✏️ edited (not full review)

Chapter map: 1 Intro · 2 Background · 3 Methodology (3.1 Dataset, 3.2 Methods, 3.3 Eval) · 4 Results · 5 Discussion · 6 Conclusion · App A–F

---

## 1. Status

| § | Title | Status |
|---|-------|--------|
| — | Abstract | ☐ |
| 1 | Introduction | ☐ |
| 2.1 | Medical Tabular Data & Privacy Regs | ✅ |
| 2.2 | SDG Taxonomy | ✅ |
| 2.3 + 2.3.1 + 2.3.2 | GAN foundations, HealthGAN | ✅ |
| 2.3.3 | PATE-GAN | ✅ |
| 2.4 + 2.4.1–4 | LLM, Tabula, Pythia, LoRA, DP-aug | ✅ |
| 2.5 + 2.5.1–4 | DP, DP-SGD, PATE, RDP | ✅ |
| 2.6.1 | Eval — Statistical Similarity | ✅ |
| 2.6.2 | Eval — Predictive | ✅ |
| 2.6.3 | Eval — Privacy | ✏️ |
| 2.6.4 | Over-reliance / similarity≠privacy (NEW) | ✏️ |
| 2.7 | Research Gap | ✅ |
| 3.1 | Dataset & Preprocessing | ☐ |
| 3.2 | Generation Methods | ☐ |
| 3.3 | Evaluation Framework | ✏️ |
| 4 | Results (RQ1–3) | ☐ |
| 5 | Discussion | ✏️ |
| 6 | Conclusion | ✏️ |
| A–F | Appendices | ☐ |

---

## 2. Action backlog

**Mechanical fixes**
- ✅ Stale comment numbering §6.x → §4.x
- ✅ "four" → "five" generators (background, methodology — both done)
- ☐ "seven Cox" → thirteen: evaluation.tex
- ✅ Hardcoded body refs §4.2/§2.5 → \ref{sec:methods:healthgan}; dropped false "DP extension" clause
- ✅ Hyphen `differentially-private`→`differentially private` (3 spots: 2.3.3, 2.5.1, 2.5.3)
- ☐ Decide global \textcite vs cite-at-end standardisation (~15 \textcite in 2.2/2.4.1–2/2.6)
- ☐ Stale comment "four generators rather than three": conclusion.tex:13

**Empty TODOs**
- ☐ Abstract headline findings
- ☐ Conclusion: RQ1–3 answers
- ☐ Results 4.1.3 minimal-suite
- ☐ Appendix: hyperparam tables, reproducibility, per-attribute

**Citation gaps to chase**
- ☐ "Seven ways to evaluate the utility of synthetic data" (El Emam 2020) — not in folder
- ☐ Privacy-preserving healthcare informatics review (uncited, fits 2.1)

**Cleanup**
- ☐ Delete redundant `data_analysis.ipynb.bak` (notebook reverted by user)

---

## 3. Decisions

- 2.1 expanded (option B): +med-data characteristics +reg pointer; dropped t-closeness; l-diversity→park2013perturbed; +PeGS "classical metrics don't apply to model-based synthesisers"
- 2.2: citations corrected — LLM family +shi+Borisov, sequential trees→emam, failure modes→shi; hybrid claim fixed (blend, not "predominantly KD"); axes sentence de-slashed; all 5 cites verified; grammar "which is a challenge" kept per user
- 2.3 + 2.3.1: Goodfellow / WGAN(Arjovsky) / WGAN-GP(Gulrajani) / shi verified verbatim; broken §4.2/§2.5 refs fixed → \ref{sec:methods:healthgan}; "first documented in image domain" (soft, shi) kept per user
- 2.3.2 HealthGAN: claims verified (batch/bottleneck/export/six-method near-verbatim); +healthgan cite on architecture sentence; writing fixes — binary softened, leveraging→using, AA→NNAA, paragraph split
- 2.3.3 PATE-GAN: verified vs pategan(Jordon ICLR19)+Papernot — disjoint partitions, k teacher-discriminators, noisy aggregation, student-discriminator, formal (ε,δ)-DP all confirmed; "avoids gradient exposure" = fair paraphrase; survey/stadler/privsyn cites prior-verified
- 2.4.1 Tabula/row-as-text: NLL-min algebra (∏p ⟺ ∑log ⟺ −∑log) = standard MLE; left-padding explained; Miletic2024 verified
- 2.4.2 Pythia: model family + scale axis (70M vs 1B); last sentence reworded to avoid variable name "Pythia-1B-DP"
- 2.4.3 LoRA: h=W0x+BAx, A~Gaussian/B=0, scale by α/r (thesis α=16); Yu = adapters competitive/superior, Li kept but flagged as CONTRADICTING (full FT remains strong) — honesty per user
- 2.4.4 DP-aug LLM: Carlini/Nahid/Yu/Li verified verbatim; \textcite→author+cite; "two routes" = author synthesis
- 2.6.1: +goncalves joint-fidelity line; dim-wise probability (medGAN), Hellinger + distinguishability/pMSE (emam); MI named as non-linear option
- 2.6.2: DWP mis-attributed to emam → fixed to Choi2017MedGAN (medGAN, verified); emam = "all-models" mult. prediction accuracy
- 2.6.3: +hitting rate + SynthEval privacy suite (ε-identifiability already present via Murtaza)
- 2.6.4 NEW: over-reliance + similarity≠privacy (stadler/TAPAS/Shokri/Carlini/kaabachi); +SynthEval DP-not-sufficient finding
- 2.7 rewritten to 3 paras, academic tone, em-dash/slash removed, four→five
- 3.3: +SynthEval validation of metric selection (complementary metrics, per-metric ranking, no single composite)
- 5: +standardisation SynthEval cite; +RQ2 corroboration (high utility ≠ strong privacy); +MI limitation (PCD linear-only)
- 6: +future-work subsection "Non-Linear Dependency Metrics"
- MI-difference: tested via SynthEval's own def → REJECTED (PATE-GAN scores best = independence-mimicry, contradicts every other metric). Written as limitation only, NOT as discarded experiment. Notebook reverted.
- 2.5 DP: 18/18 claims ✅ (2 DP-SGD eqs + PATE eq + RDP conversion = verbatim/exact); δ wording finalised "no larger than inverse dataset size, and preferably smaller still" + danger reason (Dwork2014 Def 2.4, p.18: δ≈1/N "very dangerous", permits releasing complete records); +section intro para (signposts 4 subsecs; Dwork2014 auxiliary-info cite); 5×\textcite→author+cite; +Dwork2014 cite on post-processing sentence (Prop 2.1, §2.5.3); sources re-verified vs PDFs (Dwork06/14, Abadi, Papernot, Mironov); advanced-composition O(√(k log1/δ)·ε) big-O shorthand → exact Theorem 3.20 (Dwork2014 §3.5.2 p.49) as display eq:bg:dp:advcomp, ε'=√(2k ln(1/δ'))ε+kε(eᵉ−1) (user wanted verbatim eq, no simplification)
- 2.5.2 eqs reverted to Abadi Algorithm 1 exact notation (p.310): eq2.8 ḡₜ(xᵢ)←gₜ(xᵢ)/max(1,‖gₜ(xᵢ)‖₂/C); eq2.9 g̃ₜ←(1/L)Σᵢ(ḡₜ(xᵢ)+N(0,σ²C²I)), θₜ₊₁←θₜ−ηₜg̃ₜ. Noise-in-sum kept per paper (user: exact-to-paper, despite known imprecision = clean form adds noise once); +ηₜ to symbol list (was undefined); |B|→L, mini-batch→lot; italic kept not bold (user: consistent w/ eqs 2.1–2.7)
- 2.5.3 eq 2.10 (PATE = Papernot eq (1), p.4): +vector arrow on nⱼ(x⃗); LHS renamed ŷ(x)→f(x) to match paper exactly; formal label-count def + Laplace expansion tried then reverted per user (too much — kept original prose). Sensitivity-1→DP = Papernot Theorem 2 p.6
- 2.5.4 accountant HONESTY fix (match thesis→code): code never sets accountant='rdp'. DP-LLM trained via Opacus PrivacyEngine() default = PRV accountant (Opacus≥1.2; pin opacus>=1.4.0 in pythia/requirements.txt) NOT RDP; PATE-GAN uses moments accountant (pate_gan.py:219). Reworded §2.5.4 closing "RDP standard framework for both algorithms" → RDP = theoretical foundation; concrete accountants = moments (PATE-GAN) + Opacus numerical/PRV (LLM, ~cite Opacus). Softened §2.5 intro "accountant both rely" → "framework underpins". Side note: root requirements.txt opacus==0.15.0 is stale (code uses 1.x make_private_with_epsilon API). RE-VERIFY caught a citation-claim mismatch in the new closing: Opacus 2021 paper documents RDP (not PRV/numerical), so ~cite Opacus didn't support "numerically rather than Rényi". FIXED by adding gopi2021numerical (Gopi/Lee/Wutschitz NeurIPS21 = PLD/PRV accountant, verified) on the numerical clause + Opacus cite moved to "trained with Opacus". §2.5.4 now 6/6 claims sourced
- 2.5.2 vs code: mechanism justified (max_grad_norm=C=1.0, Gaussian noise, Opacus default poisson_sampling=True, σ derived from target_ε=5). 2 honesty clauses added near Opacus para: (a) optimizer is AdamW=DP-Adam not plain SGD — eq 2.9 descent is canonical/Abadi, clip+noise unchanged; (b) DP applied only to LoRA adapter params, base frozen (xref §2.4.3). Heading kept "DP-SGD" (canonical umbrella term; Opacus+Li2022 call it DP-SGD w/ Adam). TODO methods.tex: "DP-SGD with AdamW (DP-Adam)" for precision
- 2.5 optional/deferred: Papernot2018 (Gaussian-PATE→RDP) + Mironov2019 (sampled-Gaussian RDP) cites — sentences uncited + field-true, not hallucinations; add only if PDFs/bib sourced. [gopi2021numerical PRV cite — DONE, added to §2.5.4]
- 5 generators confirmed correct (70M vs 1B separate — scale axis, load-bearing for RQ2/RQ3)
- Style: no em-dash; no slash; keep `privacy--utility` en-dash; `differentially private` (no -ly hyphen)
- Cite placement MIXED across doc: 2.4.3–4 + 2.5 = author+cite-at-end; 2.2 + 2.4.1–2 + 2.6 = \textcite. Global standardisation DEFERRED — user picks direction; both academically valid (\textcite OK when author is sentence subject)
- Verify citations by direct read — pypdf mangles two-column PDFs

---

## 4. Citation health

- Now cited (were unused/newly added): Choi2017MedGAN, kaabachi2025scoping, houssiau2022tapas, park2013perturbed, Borisov2023GReaT, lautrup2025syntheval, emam2021optimizing
- Fixed bib title artifact: lautrup2025syntheval ("...: AD Lautrup et al." → clean title)
- PDFs added to folder for verification: Goodfellow GAN, WGAN, WGAN-GP, medGAN, GReaT, SynthEval, TAPAS, scoping review, Perturbed Gibbs, Tabula, Pythia, LoRA(Hu), Li2022, Carlini, Dwork2006(Calibrating noise), Dwork2014(Algorithmic Foundations), Abadi2016(Deep Learning w/ DP), Mironov2017(RDP), Papernot2017(Semi-supervised KT), Opacus, Xie2018DPGAN, pategan(Jordon)
- Still uncited folder PDFs: DP-synthetic-data (scalable/general), cross-sectional synthetic EHR, privacy-utility combined chart
- _TODO: list bib keys never \cite'd._
