# Track B: cancer tissue of origin across institutions

Back to [README](../README.md). Data access: [quickstart.md](../quickstart.md), cBioPortal section.

| Item | Value |
|---|---|
| Task | Predict cancer type from tumor genomics |
| Data | cBioPortal: mutation and copy number, binary per gene, on the shared panel genes |
| Clients | One per institution. `msk_chord_2024` (MSK, 5 cancer types). A TCGA PanCanAtlas union of the same 5 types (`brca`, `luad` plus `lusc`, `coadread`, `prad`, `paad`). `pancan_pcawg_2020` (ICGC/TCGA whole genomes) |
| Label | `CANCER_TYPE` |
| Model | Logistic regression |
| Metric | Macro F1 |
| Stage 3 output | Driver genes per cancer type, federated versus per client |
| Status | Runs in parallel if time allows |

## Why this track

The feature shift across clients is real and known: MSK-IMPACT is a 505-gene panel, TCGA is whole exome, PCAWG is whole genome. After the gene intersection, the clients still differ in sequencing depth, calling pipelines, and CNA thresholds. A loss to the pooled model is a result, not a failure. The task has a clean label and a known answer (tissue-specific drivers), so Stage 3 is checkable.

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

The loader lives in `fedbio/cbio.py` and is shared with Track C.

## Contracts for this track

| Contract | How Track B meets it |
|---|---|
| Feature | Gene intersection computed once from the three panel definitions and fixed in `fedbio/cbio.py`. Two columns per gene, fixed order. |
| Label | Cancer types present in at least two clients. Fixed list server-side. |
| Preprocessing | Binarization thresholds (CNA ±2) fixed in the loader. No scaling needed for binary features. |
| Model | Same as Track A. |

## Tasks

| ID | Task | Done when |
|---|---|---|
| M2 | cBioPortal loader | `load_client("tissue", k)` returns data for 3 clients |
| P2 | All three stages | `results/B_table.csv` with local, FedAvg, pooled columns |

## References

- Jee et al. 2024, [Automated real-world data integration improves cancer outcome prediction](https://doi.org/10.1038/s41586-024-08167-5), Nature. MSK-CHORD.
- cBioPortal web API: https://www.cbioportal.org/api/swagger-ui/index.html
