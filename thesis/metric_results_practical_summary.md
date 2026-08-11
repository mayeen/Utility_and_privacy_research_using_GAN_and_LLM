# Practical Metric-by-Metric Results Summary

## Interpretation approach

This summary treats small numerical differences as practical ties. A “slight edge” means that a model was numerically better, but not necessarily statistically significantly better. The notebook reports single estimates without confidence intervals or repeated-run uncertainty, so the comparisons below are descriptive.

## Utility and fidelity metrics

| Metric | Practical summary |
|---|---|
| **Mean KS ↓** | **HealthGAN clearly performed best** at **0.0480**. Pythia-1B (**0.0805**) and Pythia-70M (**0.0856**) performed almost the same, with a negligible edge to Pythia-1B. Pythia-1B-DP (**0.0992**) was slightly worse. PATE-GAN (**0.5381**) was far worse. |
| **Correlation difference ↓** | **Pythia-1B had the edge** at **0.0206**. Pythia-70M (**0.0308**) and HealthGAN (**0.0336**) were practically equivalent. Pythia-1B-DP (**0.0466**) was moderately weaker. PATE-GAN (**0.1142**) was the poorest. |
| **TSTR ROC-AUC ↑** | **Pythia-1B and Pythia-70M form the strongest synthetic group**, with a slight edge to Pythia-1B: **0.6818 vs. 0.6634**. Pythia-1B was closest to the real-data result of **0.7008**. HealthGAN was moderate at **0.6312**, Pythia-1B-DP was weak at **0.5827**, and PATE-GAN was essentially random at **0.5108**. |
| **TSTR F1 ↑** | **Pythia-1B and Pythia-70M performed identically in practice:** **0.5036 and 0.5005**. HealthGAN (**0.3824**) and the real baseline (**0.3668**) were broadly similar, although their precision/recall balances differed. Pythia-1B-DP was lower at **0.2714**, while PATE-GAN failed at **0.0093**. |
| **Accuracy ↑** | The values were relatively close: real **0.6962**, HealthGAN **0.6720**, PATE-GAN **0.6583**, Pythia-1B **0.6569**, Pythia-70M **0.6408**, and Pythia-1B-DP **0.6389**. However, **accuracy is not useful for ranking these models** because an all-negative classifier already achieves approximately 66%. PATE-GAN demonstrates this problem: high-looking accuracy but almost zero recall. |
| **Precision ↑** | The real-trained model clearly performed best at **0.6301**. HealthGAN was next at **0.5321**. Pythia-1B (**0.4959**) and Pythia-70M (**0.4749**) were broadly similar. Pythia-1B-DP was lower at **0.4327**, and PATE-GAN was lowest at **0.3433**. |
| **Recall ↑** | **Pythia-70M and Pythia-1B were effectively tied for best recall**, at **0.5290 and 0.5115**. HealthGAN (**0.2984**) and the real model (**0.2587**) formed the middle group. Pythia-1B-DP was lower at **0.1977**, while PATE-GAN detected almost no positive cases, with recall **0.0047**. |
| **Class-balance preservation** | Pythia-70M, Pythia-1B, and Pythia-1B-DP reproduced the real positive rate exactly at **0.340**. HealthGAN was also close at **0.321–0.336**. PATE-GAN was substantially incorrect at approximately **0.50**. |
| **TRTS AUC ↑** | **Pythia-1B performed best**, with GB AUC **0.6980**, almost matching the real-data benchmark. HealthGAN was second at **0.6531**. Pythia-70M (**0.5939**) and Pythia-1B-DP (**0.5338**) were weaker; PATE-GAN (**0.4601**) failed. |
| **TSTS AUC ↑** | Pythia-1B had the highest internal synthetic performance: GB **0.7119** and RF **0.8343**. However, this does not mean better real-world utility. The very high RF value probably reflects synthetic-specific patterns because its real-test AUC was only **0.6682**. |
| **Feature-rank Spearman correlation ↑** | HealthGAN (**0.950**), Pythia-70M (**0.970**), Pythia-1B (**0.970**), and Pythia-1B-DP (**0.952**) all performed similarly well. The slight numerical edge goes to the two non-private Pythia models. PATE-GAN was clearly lower at **0.615**. |
| **Real top-10 importance share** | HealthGAN (**0.745**), Pythia-70M (**0.779**), Pythia-1B (**0.761**), and Pythia-1B-DP (**0.799**) were broadly similar. Pythia-1B-DP was numerically highest, but higher is not automatically better—it may indicate excessive concentration on a small number of variables. PATE-GAN was clearly lower at **0.430**. |
| **Feature-importance L1 agreement ↑** | **HealthGAN and Pythia-1B were practically tied for best**, at **0.914 and 0.911**. Pythia-70M was close at **0.889**. Pythia-1B-DP was somewhat lower at **0.856**, and PATE-GAN was substantially lower at **0.675**. |
| **Cox CI overlap ↑** | HealthGAN (**58.3%**), Pythia-1B (**53.8%**), and Pythia-1B-DP (**53.8%**) were effectively the strongest group. Pythia-70M was lower at **30.8%**, and PATE-GAN had **0%** overlap. However, the Cox duration/event construction is problematic, so this ranking should not be used as clinical survival evidence. |

### Utility conclusion

- **Pythia-1B and Pythia-70M are the leading downstream-prediction group**, with Pythia-1B having a small overall edge.
- **HealthGAN is best for marginal-distribution fidelity and overall feature-importance agreement**, but its predictive utility is lower.
- **Pythia-1B-DP loses considerable predictive utility.**
- **PATE-GAN performs poorly and is generally unusable.**

## Privacy metrics

| Metric | Practical summary |
|---|---|
| **Synthetic-to-real NN distance** | Pythia-1B (**1.406**), Pythia-70M (**1.440**), and Pythia-1B-DP (**1.533**) were practically similar, with Pythia-1B closest to real records. HealthGAN was farther away at **2.233**. PATE-GAN’s value of **219.879** indicates distributional failure, not meaningful privacy. |
| **NNDR ≈ 1** | **HealthGAN was closest to the neutral reference**, with **1.063**. Pythia-1B-DP (**0.763**), Pythia-70M (**0.721**), and Pythia-1B (**0.698**) formed a similar below-one group, with Pythia-1B-DP having a slight privacy-oriented edge. PATE-GAN’s **104.601** is an off-manifold result and should be excluded from practical comparison. |
| **NNAA test accuracy ≈ 0.5** | Pythia-1B (**0.6204**) and Pythia-70M (**0.6499**) were the least distinguishable useful datasets, with a slight edge to Pythia-1B. HealthGAN (**0.7199**) and Pythia-1B-DP (**0.7139**) were practically identical and easier to distinguish. PATE-GAN was almost perfectly distinguishable at **0.9999**. |
| **NNAA privacy loss ≈ 0** | All models were practically tied near zero: PATE-GAN **−0.0001**, HealthGAN **+0.0017**, Pythia-70M **+0.0030**, Pythia-1B **−0.0003**, and Pythia-1B-DP **+0.0076**. Pythia-1B-DP was numerically highest, but even its gap was below one percentage point. |
| **Membership-inference AUC ≈ 0.5** | **All generators performed almost identically at chance level**, between **0.4926 and 0.5119**. Pythia-70M had the highest numerical attack AUC at **0.5119**, but the difference is negligible. No model showed a meaningful membership-leakage signal under this attack. |
| **MIA advantage ≈ 0** | Again, all models were effectively tied. Advantages ranged from **−0.0074 to +0.0119**. Pythia-70M was numerically worst at **+0.0119**, but this is still a very small signal. |
| **Attribute-inference advantage ↓** | HealthGAN, Pythia-70M, and Pythia-1B-DP showed essentially no meaningful advantage. Pythia-70M’s maximum was only **+0.0064**. **Pythia-1B was the exception**, with maximum advantage **+0.0410**, mainly for the primary diagnosis. Thus, Pythia-1B had the clearest attribute-leakage signal. Pythia-1B-DP reduced this signal to **−0.0056**. |
| **Outlier linkability ↓** | All models had very low outlier linkability: HealthGAN **0%**, Pythia-1B-DP **0%**, Pythia-70M **0.1%**, and Pythia-1B **0.6%**. These are broadly low, although Pythia-1B had the slightly highest risk. PATE-GAN’s 0% is uninformative because it also failed to cover normal records. |
| **Non-outlier coverage** | Pythia-1B (**53.3%**) and Pythia-70M (**50.6%**) were effectively the strongest group. Pythia-1B-DP followed at **46.8%**, then HealthGAN at **42.0%**. PATE-GAN covered **0%**, confirming distributional failure. |

### Privacy conclusion

- For **membership inference**, all generators performed approximately the same and showed no meaningful leakage.
- For **nearest-neighbour privacy**, HealthGAN had the most neutral real-like spacing.
- For **attribute inference and outlier linkability**, Pythia-1B showed the greatest risk, although the absolute values remained relatively small.
- **Pythia-1B-DP improved the attribute and outlier results**, but not every privacy measure improved.
- PATE-GAN’s apparently good privacy results should be disregarded because its generated data are unrealistic.

## Aggregated results

| Aggregated measure | Practical summary |
|---|---|
| **Mean utility** | **Pythia-1B was best at 0.975**. HealthGAN (**0.887**) and Pythia-70M (**0.854**) were practically the second group, with a slight edge to HealthGAN. Pythia-1B-DP was lower at **0.709**. PATE-GAN was unusable at **0.000**. |
| **Mean privacy risk** | Excluding the invalid PATE-GAN result, **HealthGAN had the lowest composite risk at 0.100**. Pythia-1B-DP was intermediate at **0.349**. Pythia-70M (**0.496**) and Pythia-1B (**0.500**) were effectively tied for the highest composite risk. |
| **Utility PCA** | Pythia-1B was highest at **1.925**. HealthGAN (**1.323**) and Pythia-70M (**1.160**) were broadly similar. Pythia-1B-DP was considerably lower at **0.133**, while PATE-GAN was far below all models at **−4.541**. |
| **Risk PCA** | Pythia-1B had the highest risk score at **2.771**. Pythia-70M followed at **0.683**. Pythia-1B-DP (**−0.564**) and HealthGAN (**−1.164**) showed lower risk, with HealthGAN having the edge. This component explains only **51.2%** of privacy variation, so it is not a definitive risk ranking. |
| **Composite Pareto result** | Among meaningful models, **HealthGAN and Pythia-1B represent the main trade-off**. HealthGAN offers lower risk with strong but not maximal utility; Pythia-1B offers maximal utility with higher observed risk. Pythia-70M and Pythia-1B-DP are dominated in the particular composite calculation. |
| **DP effect** | Pythia-1B-DP reduces attribute advantage from **+0.0410 to −0.0056** and outlier linkability from **0.6% to 0%**, but AUC falls from **0.6818 to 0.5827**. Therefore, the privacy improvement comes with a substantial utility cost. |

## Final overall interpretation

- **Best predictive utility:** Pythia-1B and Pythia-70M, with a slight overall edge to **Pythia-1B**.
- **Best statistical/marginal fidelity:** **HealthGAN**, especially for KS and feature-importance agreement.
- **Best correlation preservation:** **Pythia-1B**, with Pythia-70M and HealthGAN close behind.
- **Lowest meaningful overall privacy risk:** **HealthGAN**.
- **Strongest utility but highest observed privacy-risk signals:** **Pythia-1B**.
- **Best evidence of a DP privacy benefit:** Pythia-1B-DP, particularly for attribute and outlier attacks, but with a clear AUC loss.
- **Membership-inference performance:** practically identical and close to chance for every generator.
- **PATE-GAN:** should not be selected based on either utility or apparent privacy because its results primarily indicate distributional collapse.

In practical terms, **HealthGAN is the more conservative balanced option**, while **Pythia-1B is the preferred option when predictive utility is the main priority and its higher attribute/linkability risk is acceptable**.

## Interpretation cautions

1. Small numerical differences should not be described as statistically significant because confidence intervals and repeated-run variability were not reported.
2. Accuracy is misleading for this outcome because approximately 66% of cases belong to the negative class.
3. A membership-inference AUC below 0.5 can be inverted by an attacker; it is not automatically safer than an AUC above 0.5.
4. PATE-GAN’s extreme nearest-neighbour distances indicate distributional failure rather than useful privacy protection.
5. The Cox analysis does not use a valid time-to-readmission duration, so its hazard-ratio and confidence-interval results require substantial qualification.
6. The composite and PCA rankings depend on metric scaling, direction, weighting, and the generators included in the comparison.
