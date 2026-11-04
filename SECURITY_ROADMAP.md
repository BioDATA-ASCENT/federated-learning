# Security hardening roadmap

**Status: proposed future work, not implemented protections or a security certification.**

This document recommends how to move the public-MNIST demo toward a controlled research deployment. First establish authenticated, least-privilege execution on public data. Before using sensitive biological or patient data, approve a separate privacy design and dataset-specific governance review.

For today's commands, see [DEPLOYMENT.md](DEPLOYMENT.md). This roadmap does not change the running services.

## Current baseline

Based on the checked-in deployment scripts and guide, not a live infrastructure audit:

- `scripts/server_up.sh` starts the Flower coordinator and dashboard on server loopback addresses, under `tmux`.
- `scripts/join_federation.sh` opens SSH tunnels and starts a SuperNode. Both scripts use Flower's `--insecure` mode; the server script does not explicitly enable registered-node admission control.
- SSH provides the tunnel's transport encryption and account-level access. The actual firewall, SSH account restrictions, and host configuration still need verification.
- Clients train locally. Messages include model weights, configuration, sample counts, and metrics—not just weights.
- The server distributes the application to participating nodes. Executing that application is a trust boundary, not merely a networking operation.
- The demo does not implement differential privacy or secure aggregation. Model files, metrics, and logs need protection too.
- `pyproject.toml` pins Flower to `1.42.0`. Verify flags, API compatibility, and feature/edition availability against the deployed version before implementation. Online documentation can change.

## Decisions to make first

Assign a named infrastructure owner, application maintainer, and data/privacy owner. Record:

1. Which datasets and outputs are sensitive, who owns them, and who may access them.
2. Whether the coordinator is trusted to inspect individual updates; whether clients may be malicious or compromised; and the expected collusion threat.
3. Who may join, launch runs, approve executable code, view metrics, or administer infrastructure.
4. Whether managed institutional machines are required instead of personal laptops.
5. The minimum acceptable participating cohort, failure policy, retention period, and incident-response contacts.

Protect against unauthorized participants, compromised credentials, malicious training code, update leakage, poisoned updates, and service disruption. A compromised site that can already read its own raw dataset is not made safe merely by federated learning.

## Phase 1 — Authenticate connections and restrict access

**Suggested owner:** infrastructure owner. **Gate:** complete before broadening the demo beyond a small trusted team.

### Required changes

- Replace Flower `--insecure` deployment settings with verified TLS. Use institution-managed certificates or a managed private CA; never disable hostname verification to make a tunnel work. Choose a connection hostname and certificate identity that agree. See reference 1.
- Enable registered-node admission control and issue a unique authentication key per node. Flower's documented automatic authentication identifies nodes but does not restrict admission; CLI-managed authentication restricts admission to registered keys and requires TLS. See reference 2.
- Separate node participation, run submission, dashboard viewing, and administration. Require strong operator authentication, preferably organizational SSO with MFA. Confirm actual authorization support rather than assuming login establishes role separation.
- Keep services private. Use restricted SSH tunnels or an institution-approved private network, such as Tailscale, with explicit access policies. A private network is an additional boundary, not a replacement for application authentication.
- If retaining SSH, use unique accounts/keys, restrict forwarding destinations, disable unnecessary shell and agent-forwarding access for tunnel-only accounts, and verify host keys.
- Inventory all coordinator and client listeners. Keep runtime interfaces local unless a reviewed deployment requires otherwise. A shared port can expose several API surfaces; a port allow rule alone does not separate client and administrator permissions. See reference 3.
- Protect the dashboard separately; do not grant every client visibility of sensitive per-site metrics by default.
- Store private keys outside Git and application bundles, with restrictive permissions. Document enrollment, renewal, rotation, and emergency revocation.

### Acceptance checks

- [ ] An unknown node is rejected even when it can reach the coordinator.
- [ ] A revoked node loses access, including during an active run; the behavior is recorded.
- [ ] Invalid, expired, or wrong-host certificates fail closed without insecure fallback.
- [ ] Client credentials cannot register nodes, submit arbitrary applications, or administer runs.
- [ ] Unauthenticated users cannot read dashboard data or invoke control operations.
- [ ] Network tests confirm intended listeners and block unapproved client-to-client access.

## Phase 2 — Control the code that runs at each site

**Suggested owners:** application maintainer and each site's administrator. **Gate:** complete before giving training applications access to sensitive files.

### Required changes

- Run each node/application under dedicated unprivileged identities in a restricted container or VM. A separate process alone is not a sufficient sandbox.
- Mount only the approved dataset, read-only where possible, plus a dedicated output directory. Do not expose home directories, SSH keys, cloud credentials, or container-engine sockets.
- Restrict outbound connections and pre-stage dependencies and datasets where practical. Approved code can still leak data through permitted model/metric outputs, so network restrictions alone are insufficient.
- Require each site to approve an immutable application artifact/hash and allowed run configuration. Enforce that approval locally before execution, including after application delivery by the coordinator.
- Build reproducibly from the lockfile, pin deployment images by digest, scan dependencies and images, and record provenance. Define a patch/rebuild process.
- Apply CPU, memory, storage, and job-duration limits. Replace manually maintained `tmux` production processes with supervised services and least-privilege restart policies.

### Acceptance checks

- [ ] A deliberately unapproved application or configuration is rejected before execution.
- [ ] A test application cannot read host credentials, write outside permitted paths, or contact an unapproved destination.
- [ ] Resource exhaustion is contained without disabling the host or other applications.
- [ ] Each run records the operator, application hash, dependency/image identity, participants, and approved configuration.

## Phase 3 — Approve a privacy design for sensitive data

**Suggested owners:** data/privacy owner and an FL/privacy specialist. **Gate:** no sensitive-data pilot until the design and release criteria are approved.

These controls address different risks:

| Control | Purpose | Important limitation |
|---|---|---|
| TLS / SSH | Protect communication in transit | Endpoints still see message contents |
| Secure aggregation | Hide individual updates from the coordinator under protocol assumptions | Does not by itself stop poisoning or protect every released output |
| Differential privacy (DP) | Bound disclosure under a specified privacy definition and budget | Requires a correct mechanism, accounting, and utility evaluation |

### Required decisions and experiments

- Prototype secure aggregation with the actual Message API training loop. Flower documents SecAgg/SecAgg+; verify workflow compatibility before committing to an implementation. Define collusion assumptions, dropout tolerance, and minimum survivors. Never silently fall back to plaintext individual updates. See reference 4.
- Distinguish availability from privacy: `min-nodes` is not a privacy guarantee. Define and enforce an aggregate-release threshold independently. Review repeated or overlapping cohorts as well as one round in isolation.
- For DP, explicitly choose patient-, record-, or client-level protection; define adjacency, clipping, noise, sampling assumptions, and a cumulative budget across rounds, retries, tuning, and releases. Adding `LocalDpMod` alone does not establish patient-level protection. See reference 5.
- Review all outputs: model checkpoints, sample counts, local metrics, attribution vectors, logs, and dashboard views. Suppress small cohorts and unnecessary detail. If claiming DP, include released statistics in the privacy analysis rather than relying on suppression alone.
- Evaluate utility and privacy on public data first. Document which residual risks are accepted, by whom, and why. If a protection is omitted, explicitly approve the resulting trust requirement.
- Establish dataset permissions, institutional approvals, retention/deletion rules, and any required agreements before onboarding data. This checklist is not evidence of regulatory compliance.

### Acceptance checks

- [ ] A privacy review identifies protected entities, adversaries, released outputs, and residual risks.
- [ ] Insufficient survivors or a protocol failure releases no prohibited aggregate and triggers no insecure fallback.
- [ ] Any DP claim has reproducible accounting and an enforced budget-exhaustion policy.
- [ ] Checkpoints, dashboards, and attribution outputs pass the same release review as training metrics.

## Phase 4 — Validate integrity and operational recovery

**Suggested owners:** infrastructure owner and application maintainer. **Gate:** complete before an ongoing production service; develop alongside the earlier phases.

- Validate parameter shapes, types, finite values, message sizes, and sample-count bounds. Authentication does not make a submitted update truthful. Evaluate poisoning/backdoor risks and aggregation defenses for the agreed threat model.
- Reconcile defenses with secure aggregation: individual-update inspection may no longer be available. Do not promise both without a compatible design.
- Encrypt storage and backups; restrict model, metrics, and log access; define retention and deletion, including backups. Avoid raw examples, credentials, and unnecessary identifiers in logs.
- Keep access-controlled audit records for enrollment, revocation, code approval, submissions, configuration changes, and releases. Alert on repeated failures, abnormal submissions, and certificate expiry.
- Distinguish response deadlines from process termination. Bound stuck jobs and test disconnected-node behavior, minimum-node waits, and safe cancellation.
- Document lost-device response, key revocation, site isolation, run cancellation, artifact quarantine, and notifications. Test coordinator restoration and credential rotation.

### Acceptance checks

- [ ] Malformed or oversized replies are rejected safely; tests cover misleading sample counts and non-finite tensors.
- [ ] Backup restoration succeeds in an isolated environment without restoring revoked access.
- [ ] A lost-device drill revokes access and leaves a usable audit trail.
- [ ] A partial failure cannot bypass privacy release rules or leave jobs running indefinitely.

## Suggested implementation sequence in this repository

| Work item | Likely files/components | Evidence before closing |
|---|---|---|
| TLS and registered nodes | `scripts/server_up.sh`, `scripts/join_federation.sh`, operator connection profiles | Valid connections work; invalid credentials fail |
| Roles and private access | Deployment/network configuration, `apps/mnist_fedavg/dashboard.py` | Client, operator, viewer, administrator permission tests |
| Approved-code enforcement and isolation | New release pipeline and site launcher/container configuration | Unapproved artifact and host-access tests |
| Validation and audit events | `apps/mnist_fedavg/mnist_fedavg/server_app.py`, `client_app.py` | Malformed-message tests and complete run audit records |
| Privacy prototype | Aggregation workflow, client training, metric/release policy | Protocol dropout tests and reviewed privacy analysis |
| Operations and recovery | Service definitions, backup configuration, `DEPLOYMENT.md` | Restore, rotation, and revocation drills |

**Recommended next milestone:** reproduce the MNIST run with TLS, explicit node registration, separate operator permissions, restricted execution, and tested revocation. Do not move to patient data merely because that milestone succeeds.

## Official references

The following were consulted for this roadmap. Match documentation to the pinned release; `main` documentation is not a guarantee that a feature exists in the deployed version. The recommendations above are proposed project policy, not claims that Flower automatically enforces every control.

1. [Flower: Enable TLS connections](https://flower.ai/docs/framework/how-to-enable-tls-connections.html)
2. [Flower: Authenticate SuperNodes](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html)
3. [Flower: Network Communication (main)](https://flower.ai/docs/framework/main/en/ref-flower-network-communication.html)
4. [Flower: Secure Aggregation Protocols](https://flower.ai/docs/framework/explanation-ref-secure-aggregation-protocols.html)
5. [Flower: Use Differential Privacy](https://flower.ai/docs/framework/how-to-use-differential-privacy.html)
