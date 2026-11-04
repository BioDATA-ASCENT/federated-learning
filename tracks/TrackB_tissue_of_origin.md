# Track B: cancer tissue of origin across institutions

Back to [README](../README.md). Data access: [quickstart.md](../quickstart.md), cBioPortal section.

## Food for thought

Read MSK-CHORD and inspect the cBioPortal API (references below).

- Does a shared study-level gene list guarantee coverage for every sample and panel version? When is an absent mutation or CNA record a valid zero?
- Could TCGA and PCAWG contain the same patients? How would overlap across cohorts or repeated samples within a patient affect train/test splits?
- Do the cohorts encode cancer types consistently, including lung subtypes? Could label or assay differences explain apparent client gains?
- If amplification and deletion share one binary feature, which signal is lost? What evidence would justify interpreting a predictive coefficient as a driver?

Choose one question and begin with two clients and a small shared feature set. Add CNA or another cohort as the experiment develops.

| Item | Value |
|---|---|
| Task | Predict cancer type from tumor genomics |
| Data | cBioPortal: mutation and copy number, binary per gene, on the shared panel genes |
| Clients | One per institution. `msk_chord_2024` (MSK, 5 cancer types). A TCGA PanCanAtlas union of the same 5 types (`brca`, `luad` plus `lusc`, `coadread`, `prad`, `paad`). `pancan_pcawg_2020` (ICGC/TCGA whole genomes) |
| Label | `CANCER_TYPE` |
| Model | Logistic regression |
| Metric | Macro F1 |
| Stage 3 output | Driver genes per cancer type, federated versus per client |
| Status | Parallel track choice |

## Data

Features: one binary column per gene for mutation and one for CNA (amplification or deep deletion), from the cBioPortal web API with no login. Keep the genes that all clients share, which is close to the MSK-IMPACT panel. Each client must hold several cancer types, so the TCGA client is a union of PanCanAtlas studies. Keep the cancer types that appear in at least two clients.

Do not add `msk_impact_2017`. Its patients overlap with MSK-CHORD.

Verified Nov 3: `msk_chord_2024` sample counts by `CANCER_TYPE`: NSCLC 7,809, colorectal 5,543, breast 5,368, prostate 3,211, pancreatic 3,109.

Not yet verified: that `pancan_pcawg_2020` has `CANCER_TYPE` values matching the five, and that its mutation and CNA profiles are exposed through the API. If it fails, Track B runs with two clients.

There is no download step. The loader fetches each study on first use and caches it in `data/cbio/<study_id>/`. The loader is the only code that touches the network. Tests run on the cached files.

## API calls

- Study list and attributes: `GET /api/studies/{id}/clinical-attributes`
- Cancer type per sample: `POST /api/studies/{id}/clinical-data/fetch?clinicalDataType=SAMPLE` with `{"attributeIds":["CANCER_TYPE"]}`
- Mutations: `POST /api/molecular-profiles/{id}_mutations/mutations/fetch` with the sample list and Entrez gene IDs
- CNA: `POST /api/molecular-profiles/{id}_cna/molecular-data/fetch` (profile suffix is `_gistic` for TCGA)

Tracks B and C can share one loader.

## Contracts for this track

| Contract | How Track B meets it |
|---|---|
| Feature | Gene intersection computed once from the three panel definitions and fixed in the loader. Two columns per gene, fixed order. |
| Label | Cancer types present in at least two clients. Fixed list server-side. |
| Preprocessing | Binarization thresholds (CNA ±2) fixed in the loader. No scaling needed for binary features. |
| Model | Same as Track A. |

## Possible prototype milestones

Choose milestones to suit your question; these are starting points for exploration.

| ID | Task | Done when |
|---|---|---|
| Prototype | cBioPortal loader | The loader returns data for 2 clients |
| Explore | All three stages | A result table with local, FedAvg, and pooled columns |

## References

- Jee et al. 2024, [Automated real-world data integration improves cancer outcome prediction](https://doi.org/10.1038/s41586-024-08167-5), Nature. MSK-CHORD.
- cBioPortal web API: https://www.cbioportal.org/api/swagger-ui/index.html
