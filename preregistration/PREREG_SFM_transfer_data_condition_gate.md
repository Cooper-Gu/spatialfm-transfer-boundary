# Preregistration — SFM transfer data-condition gate

**Study ID:** SFM-TRANSFER-PREREG-001  
**Freeze time:** 2026-09-27 19:26:22 UTC  
**Status:** public registry record published  
**Scope:** the next experiment only. Results already generated in earlier donor-heldout and spatial analyses are historical and are not retroactively declared preregistered.

## 1. Primary question

Can the external DLPFC donor-heldout deficit of the frozen spatial RNA representation be explained by the amount of donor diversity and the accessible gene panel, or does the deficit persist at the fullest available data condition?

## 2. Confirmatory hypotheses

**H1 — data-condition response.** The donor-level difference between the frozen representation and each baseline will be evaluated as a function of training-donor count and nested gene-panel coverage. The prespecified direction is that more training donors and more accessible genes should not reduce the model-minus-baseline difference if data sufficiency is the limiting factor.

**H2 — full-condition boundary.** At the fullest available condition (two training donors, 100% of the frozen 5,442-gene overlap), the frozen representation will be interpreted as data-rescued only if its donor-level mean macro-F1 exceeds both raw genes and train-only PCA and the direction is positive in at least two of the three held-out donors. Otherwise the result remains a representation/readout boundary.

H1 is a directional diagnostic; H2 is a prespecified decision rule. Neither is a claim about all spatial foundation models.

## 3. Data and holdout

- **Target source:** spatialLIBD human DLPFC Visium, 12 slices and three donors already audited as absent from the training registry.
- **Holdout:** leave-one-donor-out. For each held-out donor, the two remaining donors are the training pool.
- **Primary training-donor conditions:** one donor (exploratory sensitivity) and both available donors (primary full condition). For the one-donor condition, the lower donor identifier is selected deterministically within each fold; no result-driven donor choice is permitted.
- **Gene-panel fractions:** 10%, 25%, 50%, 75% and 100% of the 5,442-gene common panel. Gene sets are nested and selected from training-donor variance only, with descending variance and lexical gene-name tie break.
- **Target labels:** never used for gene selection, representation fitting, probe fitting or condition selection. They are used only once for the final held-out evaluation.

## 4. Representations and baselines

- **Frozen representation:** the current label_heldout checkpoint, with weights fixed and no target-donor adaptation.
- **Raw baseline:** the same nested gene panel, training-donor standardization and the existing raw-gene probe protocol.
- **PCA baseline:** train-only randomized PCA on the same nested panel; component count is `min(64, n_genes - 1)`.
- **Probe:** fixed SGD macro-F1 probe with seeds 260, 261 and 262, the same training-step budget and class-cap rule across all methods and conditions.
- **Feasibility gate:** before the confirmatory run, a read-only smoke test must verify that the frozen checkpoint accepts every nested panel without changing weights. If this fails, the condition is recorded as not executable and no substitute panel rule may be introduced after inspection of outcomes.

## 5. Endpoints

### Primary endpoint

Held-out donor-level macro-F1 difference for frozen representation minus raw genes at the 100% panel and two-training-donor condition. Report each donor, the arithmetic mean, median and exact sign count.

### Secondary endpoints

- frozen representation minus train-only PCA macro-F1 difference;
- macro-F1 across every prespecified panel fraction and donor-count condition;
- marker-program Pearson r, using only training-donor-derived marker programs;
- continuous Moran’s I absolute and signed error at k=4, 6 and 12;
- classifier-derived within-layer consistency, boundary recall and edge agreement as diagnostic endpoints.

## 6. Statistical rules

- Donor is the biological statistical unit. Spots, slices, programs and probe seeds are hierarchical repeats.
- No spot-level p-values, pooled spot bootstrap or claim of population-level significance will be used.
- With three donors, report exact sign counts and two-sided sign p-values descriptively; bootstrap intervals are descriptive sensitivity summaries.
- The primary comparison is paired within held-out donor and condition. No meta-analysis across unrelated sources is permitted.
- Missing PCA or invalid panel conditions are reported as missing with the reason; they are not imputed.
- All condition labels, output file names and analysis seeds are fixed before reading outcome files.

## 7. Interpretation and stopping rules

- If H2 is met, the current representation receives a narrowly scoped data-rescue signal and a separate positive-gate protocol is required before any foundation-model claim.
- If H2 is not met, the current representation is frozen as a failure-boundary result and no same-objective scaling is allowed.
- If the feasibility gate fails, the experiment stops before outcome collection and the protocol is amended only in a new preregistration version.
- No additional condition, donor selection, probe family or metric may be added after outcome inspection under this version.

## 8. Reproducibility package

The preregistration package contains this protocol and `analysis_contract.json`. Before execution, the implementation script and environment lockfile must be added to the package and hashed. The frozen package must be archived on a public registry (OSF, Zenodo or an institutional equivalent) and the DOI recorded in the manuscript. A local hash and a server copy alone are not equivalent to public preregistration.

## 9. Relationship to the current manuscript

The current earlier donor-heldout and spatial analyses results support the already completed donor-heldout and spatial-readout diagnostics. This protocol begins prospectively with the data-condition curve and prevents those historical results from being relabeled as preregistered. The manuscript continues to use the bounded claim: transferability, data conditions and failure boundaries of one frozen spatial RNA representation.
