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
| 3.1 | Dataset & Preprocessing | ☐ (3.1.1–2 peer-critiqued, unedited) |
| 3.2.1 | Selection Rationale | ✅ |
| 3.2.2 | HealthGAN (.1 SDV, .2 arch, .3 loss, .4 hp, .5 sample-sel, .6 DP-excl) | ✅ verified vs code + edited |
| 3.2.3 | PATE-GAN (intro,.1,.2,.4 ✅ verified; .3,.5 user-checked) | 🔄 (intro/.1/.2/.4 flags pending edit) |
| 3.2.4 | Pythia-70M (intro,.1–.6) | ✅ code-verified; ⚠️ HP batch/epochs/lr = CLI defaults ≠ run metadata |
| 3.2.5 | Pythia-1B (scale anchor, cross-ref) | ✅ verified; ⚠️ "identical config" false (batch/lr differ per run) |
| 3.2.6 | Pythia-1B-DP (.1–.3 after 6→3 merge) | ✅ verified + ARS panel → 6→3 merge; ⚠️ DP batch 32/16 swapped (run=16/32) |
| 3.2.7 | Implementation Summary | ✅ verified + edited (cut Software col, deleted Property 3) |
| 3.3 | Evaluation Framework (evaluation.tex) | ✏️ style pass only; content unverified |
| 4 | Results (RQ1–3) | ☐ |
| 5 | Discussion | ✏️ |
| 6 | Conclusion | ✏️ |
| A–F | Appendices | ☐ |

---

## 2. Action backlog

**Mechanical fixes**
- ✅ Stale comment numbering §6.x → §4.x
- ✅ "four" → "five" generators (background, methodology — both done)
- ✅ "seven Cox" → thirteen: evaluation.tex (verified 13 entries in code `cox_covariates`)
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
- 3.2+3.3 STYLE PASS (from deep-research peer review): em-dashes removed entirely both files (eval 19→0, methods 12→0; en-dashes privacy--utility/1--14/train--validation kept); eval "captures" template 17→0 (varied verbs / lead-with-measurement, \emph noun-phrases kept); methods throat-clearing fixed (load-bearing→essential; "worth re-stating explicitly"→"Three properties...bear on the comparison"); "this thesis"/"here" thinned (eval 6, methods 2); table N/A em-dash cells→"n/a" (judgment call — confirm convention); Cox seven→thirteen (verified). Skipped reviewer's blind-to antithesis (only 3×, fine). Reviewer-praised parts left untouched (selection rationale, HealthGAN DP-exclusion 4-point, "Differences from Non-DP Pipeline"). ⚠️ methods CONTENT + citations still need USER verification (not done)
- 3.1.1/3.1.2 peer-critique given (NOT edited, user reviews tomorrow): major flags = (a) A1Cresult/max_glu_serum "None" counted as missingness contradicts "?"-only def + later ordinal encoding; (b) 47-features vs 50-columns unreconciled; (c) field-name format inconsistency (spaces/hyphens vs underscores); (d) dedup shifts readmission 46%→34% unflagged
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

**3.2 methods CODE-verification (this session) — vs healthgan/ + Pategan/ source**
- 3.2.2.1 SDV (sdv_converter.py): truncnorm mean=mid std=(b−a)/6, ordinal add-1 smoothing α=1 + zero-count reinsert, binary cat/Beta, numeric minmax, missing cat→MISSING/ord-bin→mode/num→median, decode clip[0,1]+snap≤50+ID clip-int — ALL exact. Flag (unedited): rename "Laplace smoothing"→add-one (clashes w/ DP Laplace mech)
- 3.2.2.2 arch (wgan_for_mac.py): gen 100→2F(90)→round(1.5F)(68)→F(45)sigmoid, critic F→B→2B→4B→1 (64/128/256), B=64 F=45 — exact
- 3.2.2.3 loss EDITED: collapsed L_D/L_G+L_GP → single Gulrajani eq(3) w/ \underbrace (user req); sign fix (terms=negated W₁, minimise→maximise); "enforces"→"regularises"; DP-incompat scoped "DP-SGD pipeline in this thesis". Src = Gulrajani p.4 eq(3)+Alg1 (λ=10, ε~U[0,1], x̂=εx+(1−ε)x̃)
- 3.2.2.4 hp EDITED: n_crit=5, Adam(1e-4,0.5,0.9), batch ⌊57214/5⌋→11400, 100k. FIXES: test-loss on shuffled 11400-batch (not full 14304); files "10 gen/3 eval" (was "10×14304 match test" — disk = 3×57214 vs TRAIN); β1=0.5≠Gulrajani0 noted; "epoch"→"iteration" (1 gen step, not data pass). large-batch→rare-values verbatim healthgan p.5
- 3.2.2.5 sample-sel: 10 drawn/3 kept; labels 0/3/6 KEPT (user) ← files synthetic_0/1/2_decoded (57214 rows); eval vs TRAIN (real_hg=df_train); draw0 carried fwd (cell30 HealthGAN=df_healthgan=synthetic_0); KS 0.0477–0.0484/corr 0.0336–0.0342 ranks consistent. (net edit 0: 0/3/6→0/1/2 then reverted)
- 3.2.2.6 DP-excl REPLACED (user-supplied text + cite fix): grounded in wgan_for_mac_dp.py (Opacus PrivacyEngine, GP removed→weight-clip, docstring verbatim). ε≤10-cliff OVERCLAIM removed; param-grad(DP-SGD) vs input-grad(WGAN-GP) distinction; cite fix utility→pategan / DPGAN→Xie2018 / framework→Papernot; +lee2021scaling (He&Kifer GPU). All claims source-verbatim (Abadi p2+p8, Opacus, healthgan p5)
- 3.2.3 intro (pate_gan.py+generate): teacher-student-generator isolation, Laplace voting, post-processing, D=45(incl encounter_id) min-max[0,1], inversion exact. Flags (unedited): inversion ORDER (code clips[0,1]→rescale→round; thesis reversed); "integer endpoints"→"min&max both integers" (endpoint heuristic, not discrete-support)
- 3.2.3.1 teacher-student: k=10 disjoint equal, refit-from-scratch LINEAR LR teachers (not NN), balanced task, student isolation — exact. ARS 5-reviewer panel → Minor Revision. Flags (unedited): (P1) DP misattributed to "teacher training" (actually noisy label-release; disjoint=bounded SENSITIVITY); (P2) weight-clip on STUDENT not generator; (P3) "exactly one teacher"→"at most one" (4 records dropped, 10×5721=57210)
- 3.2.3.2 Laplace voting: n0/n1 counts, η~Lap(0,λ=1), ŷ=1[(n1+η)/(n0+n1)>½] — exact; noise on POSITIVE-count only (not noisy-argmax) honestly disclosed. Flag (unedited): "privacy analysis applies directly"→"accounting of original impl applies unchanged"
- 3.2.3.3 moments accountant + 3.2.3.5 privacy budget/hp: user-checked (not by me)
- 3.2.3.4 arch: gen z~U[-1,1]^45→180tanh→180tanh→45sigmoid, student 45→45ReLU→1, teachers LR — exact. Flag (unedited): code xavier_init = stddev sqrt(2/fan_in) = HE init, not Xavier (naming misnomer, thesis copies code label)

**3.2.4 Pythia-70M (pythia_tabular.py + generate_pythia_synthetic.py)**
- 3.2.4 intro + .1 Base Model: model EleutherAI/pythia-70m (gen script L58), decoder-only, EOS→pad fallback (L538-9), padding_side=left (L540), Miletic "outperform reference GAN" — exact. Flag (unedited): §3.2.1 says "comparable to CTGAN" vs §3.2.4.1 "match/surpass strong GAN baselines" — harmonise (Miletic = single reference GAN)
- 3.2.4.2 serialisation: `Class_{L} | col=val | ...` (CLASS_PREFIX_TEMPLATE L20, serialize_row L198-202), int→str/float→.15g/bool→0,1/NA — exact. Flags (unedited): (a) eq shows `c = v` w/ spaces, code = `col=val` no spaces; (b) body includes target col (df.columns.tolist) → label double-encoded, unstated
- 3.2.4.3 schema (derive_table_schema L147-195): dtype/is_numeric/min-max/integer_coded(isclose-round)/discrete≤25/categories/mode — 7 fields exact, threshold 25 exact. CLEAN, no fix
- 3.2.4.4 LoRA (L549-554 hardcoded): r=8,α=16,dropout=0.05,bias=none,CAUSAL_LM; collator mlm=False pad_to_8 (L613-4); fp16=use_cuda; best-loss snapshot/restore (508-529,623-4); eval() L631. Training knobs (epochs10/batch8/lr2e-5/maxlen512) = gen-script CLI DEFAULTS (L67-70) not hardcoded — holds if run used defaults. CLEAN
- 3.2.4.5 generation: prompt `Class_ℓ | {first_non_target}=` (L905-909); do_sample/temp0.8/top_p0.95/max_new=max_length/min_new=64/use_cache (L1006-13); label enforced post-parse (L342). temp/top_p = CLI defaults. Flag (unedited): "512 max new tokens" = reused max_length, not independent
- 3.2.4.6 parse/coerce/retry: regex L232, first-occ dedup L242-3, coerce clip/snap/round + cat→mode (251-289), default mode→discrete[0]→midpoint (296-318), target overwrite L342, retry 8·n_rows (--max-retries-per-row=8), resample-replace L1078, validator L406-428. FIXED: formula `⌈0.1·K⌉`→`⌊0.1·K⌋` (code int()=floor; value 4 ✓, ceil gave 5). User judged over-detailed → drafted compressed 2-para replacement (approved, not yet swapped in)
- 3.2.5 Pythia-1B: same code, model_name=pythia-1b override (1B params, Biderman); identical serial/schema/LoRA/gen/parse — cross-ref correct, right altitude. CLEAN. Minor: training knobs = same defaults
- 3.2.6 Pythia-1B-DP (pythia_tabular_dp.py): .1 rationale (70M DP-SGD failed at ε=5 → 1B, Li2022), .2 Opacus/Yu2022 DP-on-adapters post-processing, .3 clip C=1.0 + dropout=0, .4 B_eff=B_phys32×G16=512, .5 target(5,1e-5) PRV accountant realised-ε, .6 differences. ARS 5-reviewer panel (length focus) → Minor Revision = OVER-REDUNDANT not over-detailed. EDITED P1: clip eq removed → back-ref Eqs eq:bg:dp:clip/update; P2: .6 itemize→table tab:pythia-dp-diff; P3: Opacus 3-step→1 sentence. ~20 redundant lines cut, all unique DP facts kept
- 3.2.6 RESTRUCTURE (user "do earlier version" = ext-assessment 6→3): merged to 3 subsubsecs = Rationale / Private LoRA Training [.2+.3] / Privacy Budget & Pipeline Differences [.4+.5+.6]. CUT module-validation sentence (no-op for Pythia: ModuleValidator.fix L672 but GPTNeoX=LayerNorm-only, Opacus only swaps BatchNorm → never fires). Batch kept 2-3 sentences (NOT 1, per my pushback: q=B_eff/N is ε-determining). Dual-label epsilon+diff on subsubsec3 (both ext-ref); dropped opacus/clipping/batch labels (0/0/internal refs). Refs verified clean
- 3.2.7 Implementation Summary: table 5 rows (family/privacy/ε,δ) verified vs code+metadata (PATE-GAN ε=5/δ=1e-5 defaults, TF1 disable_v2; DP achieved_ε=4.998). EDITED: cut Software-stack column (5→4 cols) — redundant w/ Family + author admits "no bearing on analysis" + discussion.tex has 0 deployment payoff (dangling promise); DELETED Property 3 + "Three"→"Two". No orphan cites
- ⚠️⚠️ RUN-METADATA DISCOVERY (SUPERSEDES "CLEAN" on 3.2.4.4/3.2.5): thesis Pythia training knobs = generate-script CLI DEFAULTS, but every run OVERRODE them. GROUND TRUTH = run_metadata*.json (NOT code defaults — defaults matched thesis so checking code alone MISSED it):
  - 70M non-DP (data/pythia/run_metadata.json): batch=**64** (not 8), epochs=**5** (not 10), lr=**4e-4** (not 2e-5)
  - 1B non-DP (data/pythia_1b/run_metadata.json): batch=**32** (not 8), epochs=**5**, lr=**1e-4** → so 70M≠1B knobs ("identical config" FALSE)
  - 1B-DP (data/pythia_1b/run_metadata_dp.json): per_device=**16** grad_accum=**32** eff=512 (thesis SWAPPED 32/16); achieved_ε_prv=4.998, σ=0.789, sample_rate=512/14304 (=test N → per-split training, thesis "N_train" imprecise)
  - PENDING coordinated fix from metadata: §3.2.4.4 (64/5/4e-4), §3.2.5 (1B 32/5/1e-4 + drop "identical"), §3.2.6 batch swap→16/32, §3.2.7 already softened by user. LoRA adapter block (r=8/α=16/dropout hardcoded) STILL correct
  - README batch 8→32 edit was REVERTED by user (70M metadata=64, 1B=32 — user picking value)

---

## 4. Citation health

- Now cited (were unused/newly added): Choi2017MedGAN, kaabachi2025scoping, houssiau2022tapas, park2013perturbed, Borisov2023GReaT, lautrup2025syntheval, emam2021optimizing
- Fixed bib title artifact: lautrup2025syntheval ("...: AD Lautrup et al." → clean title)
- PDFs added to folder for verification: Goodfellow GAN, WGAN, WGAN-GP, medGAN, GReaT, SynthEval, TAPAS, scoping review, Perturbed Gibbs, Tabula, Pythia, LoRA(Hu), Li2022, Carlini, Dwork2006(Calibrating noise), Dwork2014(Algorithmic Foundations), Abadi2016(Deep Learning w/ DP), Mironov2017(RDP), Papernot2017(Semi-supervised KT), Opacus, Xie2018DPGAN, pategan(Jordon)
- Still uncited folder PDFs: DP-synthetic-data (scalable/general), cross-sectional synthetic EHR, privacy-utility combined chart
- _TODO: list bib keys never \cite'd._
- 2.5.1 defence-perspective note: keep DP definition + brief ε/δ interpretation + one-record influence intuition + composition/accounting motivation. Consider shortening theorem-heavy material because thesis uses established DP mechanisms rather than contributing DP theory. Advanced composition equation may be unnecessary if actual accounting relies on RDP/Opacus; if retained, frame only as motivation for tighter accountants. Also simplify prose around sensitivity/noise and note Gaussian noise depends on δ as well as ε.
