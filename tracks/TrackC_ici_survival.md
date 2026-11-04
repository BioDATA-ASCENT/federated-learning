# Track C: survival on immune checkpoint inhibitors across hospitals

Back to [README](../README.md). Data access: [quickstart.md](../quickstart.md), cBioPortal section.

## Food for thought

Read Andreux for the method and a cohort paper for endpoint definitions (references below).

- Can OS and PFS define one prediction target? How will we verify each cohort's time origin, event coding, and patient overlap?
- How should expansion handle censoring within a bin and events on bin boundaries? Should aggregation weight patients or expanded rows for the chosen loss?
- Would agreement with continuous-time Cox in lifelines validate a binned logistic model? What should match a centralized implementation of the same discrete-time loss?
- Are TMB units and missing values comparable across assays? How would we test whether mutation features improve held-out prediction beyond TMB?
- Which patient pairs are absent from client C-indices? Can treated-only cohorts establish ICI benefit, or coefficient rankings establish biological effects?

Choose one question. Start with one cohort to understand the outcome and loss, then build a two-client prototype with a common endpoint.

| Item | Value |
|---|---|
| Task | Predict survival after immune checkpoint inhibitor (ICI) treatment |
| Data | cBioPortal: binary mutation calls on the gene intersection, tumor mutational burden (TMB), cancer type |
| Clients | One per hospital cohort, table below |
| Label | OS or PFS from ICI start, with event indicator |
| Model | Discrete-time Cox (logistic regression on a person-period table) |
| Metric | C-index per client |
| Stage 3 output | Genes with hazard coefficients beyond TMB, federated versus per client |
| Status | Parallel track choice |

## Clients

Verified Nov 3 by a keyword scan of all cBioPortal studies for ICI terms, filtered for an OS or PFS attribute. 16 studies matched. The usable clients, one per institution:

| Client | Study ID | Institution | Cancer | n | Assay | Outcome |
|---|---|---|---|---|---|---|
| 1 | `tmb_mskcc_2018` | MSK (Samstein) | pan-cancer | 1,661 | MSK-IMPACT | OS |
| 2 | `mixed_allen_2018` | Broad (Miao) | pan-cancer | 249 | WES | OS, PFS, RECIST |
| 3 | `mel_dfci_2019` | DFCI (Liu) | melanoma | 144 | WES | OS, PFS |
| 4 | `skcm_dfci_2015` | DFCI and Broad (Van Allen) | melanoma | 110 | WES | OS, durable benefit |
| 5 | `blca_iatlas_imvigor210_2017` | Genentech trial | bladder | 347 | check for mutation calls | OS, response |
| 6 | `rcc_iatlas_immotion150_2018` | Genentech trial | kidney | 263 | check for mutation calls | PFS, response |

Smaller or duplicate: `nsclc_pd1_msk_2018` (240, PFS) and `nsclc_mskcc_2018` (75, PFS) are MSK, so they merge into client 1 or stay out. `mel_iatlas_liu_2019` duplicates client 3. `mel_iatlas_riaz_nivolumab_2017` (107), `mel_iatlas_gide_2019` (91), `paad_iatlas_prince_2022` (93), `skcm_mskcc_2014` (64), `mel_ucla_2016` (38) are optional small clients. `ccrcc_dfci_2019` (Braun, 592 kidney) is not on cBioPortal.

Two checks before use:

1. The iAtlas studies are RNA-seq cohorts. Confirm each has a mutation profile (`GET /api/studies/{id}/molecular-profiles`).
2. Patient overlap between the DFCI melanoma cohorts and `mixed_allen_2018`. Miao 2018 reused earlier Broad melanoma cases.

Attributes per study. `tmb_mskcc_2018`: `OS_MONTHS`, `OS_STATUS` from first ICI, `DRUG_TYPE`, `TMB_NONSYNONYMOUS`, `CANCER_TYPE`. `mel_dfci_2019`: OS, PFS, `IO_THERAPY`, `PRIOR_CTLA4`, RECIST best response, `TMB_NONSYNONYMOUS`. `skcm_dfci_2015`: OS, `TREATMENT`, `DURABLE_CLINICAL_BENEFIT`, `MUTATION_LOAD`. `nsclc_mskcc_2018`: PFS, `BEST_OVERALL_RESPONSE`, `PDL1_EXP`, `TMB_NONSYNONYMOUS`.

There is no download step. The shared loader fetches each study on first use and caches it in `data/cbio/<study_id>/`. Tests run on the cached files. Test the time expansion on client 3 (`mel_dfci_2019`, 144 patients), the smallest client with both OS and PFS.

## Method

Features: binary mutation calls on the gene intersection (the MSK-IMPACT panel, because the other cohorts are WES), log TMB, and cancer type as one-hot. Choose OS-only or PFS-only cohorts for each experiment; discuss endpoint compatibility before combining clients.

The Cox partial likelihood is not separable across clients, because the risk set at each time sums over all patients. A naive FedAvg on Cox gives a stratified model. Andreux et al. (2020) replace it with a discrete-time Cox model: bin time, expand each patient into one row per bin at risk, and fit logistic regression on the expanded table with bin indicators as the baseline hazard. The loss is separable and the FedAvg loop is unchanged. Each client expands its own table. Bin edges are fixed server-side so the baseline hazard parameters align.

Stage 2: each client computes the C-index on local held-out patients. The C-index is pairwise, so a cross-client C-index needs patient pairs across sites. That is an open problem and future work.

Stage 3: coefficients per gene from the federated model versus per client. The question is whether genes beyond TMB appear in the federated model and not in any local one.

## Contracts for this track

| Contract | How Track C meets it |
|---|---|
| Feature | Gene intersection fixed in the loader. Plus log TMB, plus cancer type one-hot over a fixed type list. |
| Label | Time in months, event 0/1. Outcome type (OS or PFS) recorded per client and reported. Time bin edges fixed server-side. |
| Preprocessing | Binarization and TMB log transform in the loader. No per-client scaling. |
| Model | Logistic regression on the expanded table. Weight shape is (1, n_genes + 1 + n_types + n_bins). |

## Possible prototype milestones

Choose milestones to suit your question; these are starting points for exploration.

| ID | Task | Done when |
|---|---|---|
| Prototype | cBioPortal loader (shared with Track B) | The loader returns `(X, time, event)` for 2 clients |
| Explore | Expansion and all three stages | The expansion passes event and censoring checks, and matches a centralized fit of the same loss. A result table with the C-index per client |

## Upgrade with registration

AACR Project GENIE BPC has drug-level treatment lines and PFS across MSK, DFCI, Vanderbilt, and UHN, with one harmonized schema. It needs Synapse registration and a data use agreement. For the paper, that is the dataset.

## Checked and dropped

Breast cancer survival across METABRIC, TCGA-BRCA, and MSK-CHORD. Mutation plus CNA is a weak OS predictor in breast cancer. Time zero differs by cohort (diagnosis versus first sequenced sample). Treatment data are uneven: METABRIC has yes/no flags, TCGA has no chemotherapy or hormone flag in cBioPortal, MSK-CHORD has agent names. MSK-CHORD is a single institution. Its breast cohort has 5,368 samples and 4,160 patients with at least one endocrine agent (letrozole 2,152, fulvestrant 1,547, anastrozole 1,471, tamoxifen 1,415, exemestane 1,217, leuprolide 1,106, overlapping). That supports a single-site endocrine agent analysis, not a federated one.

## References

- Andreux et al. 2020, [Federated Survival Analysis with Discrete-Time Cox Models](https://arxiv.org/abs/2006.08997). The loss this track uses.
- Westers et al. 2025, [Horizontal federated learning and assessment of Cox models](https://doi.org/10.3389/fdgth.2025.1603630), Frontiers in Digital Health. Survival stacking plus federated logistic regression, with code.
- Samstein et al. 2019, [Tumor mutational load predicts survival after immunotherapy across multiple cancer types](https://doi.org/10.1038/s41588-018-0312-8), Nature Genetics. Client 1.
- Miao et al. 2018, [Genomic correlates of response to immune checkpoint blockade in microsatellite-stable solid tumors](https://doi.org/10.1038/s41588-018-0200-2), Nature Genetics. Client 2.
- Liu et al. 2019, [Integrative molecular and clinical modeling of clinical outcomes to PD1 blockade in patients with metastatic melanoma](https://doi.org/10.1038/s41591-019-0654-5), Nature Medicine. Client 3.
- Van Allen et al. 2015, [Genomic correlates of response to CTLA-4 blockade in metastatic melanoma](https://doi.org/10.1126/science.aad0095), Science. Client 4.
