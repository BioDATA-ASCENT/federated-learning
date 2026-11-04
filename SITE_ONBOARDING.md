# Joining the federation: a guide for partner sites

This guide is for a hospital, lab or clinic that is thinking about joining our federated learning project. It is written for everyone at your site who will be involved: the project lead, the person who looks after the data, IT, and the privacy or ethics office. You don't need a technical background to follow it. The commands for your technical contact are in [DEPLOYMENT.md](DEPLOYMENT.md).

**In short:** your data stays on a computer at your site. That computer trains a shared model on your data and sends back only the model's updated numbers. The technical setup takes an afternoon. Most of the time it takes to join goes into agreements and preparing the data.

## The idea

Think of a recipe passed between several kitchens. Each kitchen cooks with its own ingredients and writes small corrections on the recipe card. Only the card travels. The ingredients never leave the kitchen.

```
 Hospital A   data ──► training computer ──┐
 Hospital B   data ──► training computer ──┼──► coordinating server
 Your site    data ──► training computer ──┘    combines the updates
                                                into one shared model

          The data stays inside each site.
          Only model updates travel, and they are encrypted.
```

Training happens in **rounds**. In each round:

1. The coordinating server sends the current shared model to every site.
2. The computer at each site trains the model on its own data for a few minutes.
3. Each site sends back the updated model, the number of records it used, and how well the model did.
4. The server combines the updates (sites with more records count more) and starts the next round.

A full training session, called a **run**, is usually a few dozen rounds.

## What leaves your site, and what never does

| Stays at your site | Sent to the coordinating server each round |
|---|---|
| Patient records, images, samples, measurements | The updated model: a list of numbers (about 100,000 for our demo model) |
| Names, record numbers, dates, any identifiers | How many records were used for training |
| The data files and any copies of them | Summary scores: the training error, and the accuracy on a held-out part of your data |
| Which patient contributed what | How long training took |

A model update is not raw data. It isn't automatically anonymous either. Research has shown that under some conditions an update can reveal something about the records it was trained on. [Protections](#protections) lists what we already do about this and what we add before working with sensitive data. Please involve your privacy office early.

## Words you'll hear

| Word | Meaning |
|---|---|
| Federation | The group of sites training one model together |
| Site, node or client | One participant. In practice, your training computer |
| Coordinating server | Our computer that combines the updates. It never holds patient data |
| Model | What is being trained, for example a program that predicts cancer type from gene panel results |
| Weights, or update | The model's numbers. This is what travels |
| Round | One cycle: the model goes out, each site trains, the updates come back and are combined |
| Run | One full training session of many rounds |
| Dashboard | A web page that shows a run live: accuracy, each site's progress, the settings |

## Who is involved

| At your site | Role | Rough time needed |
|---|---|---|
| Site lead (PI or clinician) | Decides to join and approves results before they are shared | A meeting at the start and one after each run |
| Data steward | Prepares the dataset in the agreed format | One to five days, depending on the data |
| IT contact | Sets up the training computer and the network rule | About half a day |
| Privacy or ethics office | Reviews the agreement and the data flow | Depends on your institution |

On our side, the **federation coordinator** is your main contact. They schedule runs and share results. The **technical lead** runs the coordinating server and helps your IT contact with the setup.

## Joining in six stages

| Stage | What happens | Who | Done when |
|---|---|---|---|
| 1. First conversation | Agree on the question and check that your data fits | Site lead, coordinator | Both sides want to go ahead |
| 2. Agreements | Sign the collaboration agreement. Ethics and privacy review | Site lead, privacy office | The agreement is signed and approvals are in hand |
| 3. Prepare the data | Put the dataset in the agreed format, on a computer at your site | Data steward | The data matches the data dictionary |
| 4. Set up the computer | Install the software and allow one outgoing connection | IT, technical lead | The test connection works |
| 5. Rehearsal | A practice run on public data, together | IT, coordinator | Your site appears on the dashboard and the run finishes |
| 6. First real run | Train on your data and review the results together | Everyone | The site lead approves the results |

### Stage 1. First conversation (about an hour)

We talk through:

- The question the model answers, and why more sites help. For example, a model that predicts where a cancer started from gene panel results is more reliable if it has seen patients from many hospitals and sequencing panels.
- What data you hold: what kind, roughly how many records, and how it was collected.
- Whether your data can be put in the shared format (Stage 3).
- What your site gets back (see [What your site gets](#what-your-site-gets)).

Please don't send us any data, not even samples. A short description is enough.

### Stage 2. Agreements and approvals

Federated learning reduces how much data moves, but it doesn't replace the usual approvals. A typical set is:

- **A collaboration agreement** between the institutions. It should answer these questions:
  - Who owns the trained model, and who may use it.
  - Who may publish results, and how sites are credited: by name, or anonymously as "Site 3".
  - Which summary scores the other sites see.
  - How a site leaves, and what happens to models already trained with its help. A trained model can't "forget" one site, so the agreement says whether those models are kept or deleted.
- **An ethics review** (IRB or equivalent). Many institutions treat this as secondary use of existing data. Your office decides.
- **A privacy review** under the rules that apply to you, for example HIPAA in the US or GDPR in the EU, where a data protection impact assessment is common.

For these reviews we provide this guide, the data flow table above, the [Protections](#protections) table, and the code. The code is public at [github.com/BioDATA-ASCENT/federated-learning](https://github.com/BioDATA-ASCENT/federated-learning).

### Stage 3. Prepare the data

Every site has to describe its data the same way, or the updates can't be combined. Kitchens sharing a recipe card have to use the same units. We send you a **data dictionary**: the list of fields, their order, their units, and the allowed labels. Here is an example for a project that predicts cancer type from gene panel results:

| Field | Meaning | Format |
|---|---|---|
| `TP53`, `KRAS`, ... | Whether each gene on the agreed list is mutated in the tumour | 0 or 1, one column per gene, in the agreed order |
| `cancer_type` | Where the cancer started. This is what the model predicts | One of the agreed labels, for example `breast`, `lung`, `colorectal` |

Checklist for the data steward:

- [ ] Select only the records covered by your approval.
- [ ] Remove identifiers following your usual rules. Training never needs names, record numbers or dates of birth.
- [ ] Convert the data to the dictionary's format: the same fields, in the same order, with the same units, and labels only from the agreed list.
- [ ] If a label from the list doesn't occur in your data, that's fine. Don't add or rename labels.
- [ ] Put the file in one folder on the training computer. Only the account that runs the training should be able to read it.

The software automatically sets aside 10% of your records to check the model. Those records are never used for training, so the accuracy reported for your site is a fair test.

### Stage 4. Set up the training computer (IT, about half a day)

What's needed:

| | Requirement |
|---|---|
| Computer | A desktop, laptop, server or virtual machine inside your network that can read the prepared data. macOS or Linux. On Windows, use WSL |
| Capacity | Any recent laptop runs the demo. For larger datasets we agree the size in Stage 1. A GPU helps but isn't required. Allow about 10 GB of disk for the software |
| Network | One outgoing connection: SSH (TCP port 22) to the coordinating server's address, which we give you. Nothing incoming |
| Account | A dedicated user account on that computer, with access to the data folder only |

**For your network team:**

- The training computer makes **one outgoing**, encrypted SSH connection to one address.
- **Nothing connects into your network.** You don't need an inbound firewall rule, a VPN, or port forwarding.
- All federation traffic travels inside that encrypted connection.
- The coordinating server can't use the connection to reach other machines at your site.

**Setup, done together on a call:**

1. We create a personal login for your site on the coordinating server and send it to you securely.
2. Your IT contact installs the software with one command and registers a key, so the computer connects without a password.
3. A test login confirms that the connection works.

The exact commands are in [DEPLOYMENT.md](DEPLOYMENT.md), under "Join as a node".

**Good practice at your site:**

- Use a computer, or a virtual machine, that does only this job.
- Keep the data folder readable only by the training account.
- Keep disk encryption and your usual security software on.
- Keep the key on that computer only. If it's lost or the computer is replaced, tell us and we turn off that login.

### Stage 5. Rehearsal with public data (about 30 minutes)

Before your data is touched, we do a full practice run on **MNIST**, a public collection of 70,000 handwritten digits that is widely used for teaching. Your computer downloads it and trains on it exactly as it would on your data.

1. IT starts the connection (`bash scripts/join_federation.sh`) and leaves the window open.
2. The window shows `SuperNode ID: ...`. Your site is now connected.
3. On the same computer, open **http://localhost:8050** in a browser. This is the live dashboard.
4. We start a run. Within a minute your site appears in the dashboard's client table. The window shows one line per round, and the accuracy chart passes 90% within a few rounds.

The rehearsal proves that the connection, the software and the computer all work, without any risk to your data.

> **A setup visit.** If we come to your site, Stages 4 and 5 fit in one afternoon:
> walk through this guide with the site lead and IT (30 min), set up the computer together (1 hour),
> run the rehearsal (30 min), then questions and a date for the first real run (30 min).

### Stage 6. First real run

- We agree the date and time in advance. The training computer needs to be on and connected during the run.
- The site lead or IT can watch the dashboard. The window on your computer shows one line per round: how many records were used and the accuracy on your held-out records.
- After the run we meet to review the results. How does the shared model do on your data, compared with a model trained on your data alone?
- Nothing about your site is shared outside the federation until the site lead approves it.

## After you join

**Routine runs.** We schedule runs in advance and tell you what changed since the last one. The code is versioned. We tell you before any change, and you can review it first. If your computer is off during a run, the run goes ahead without you and nothing breaks. A site that doesn't reply within two minutes is skipped for that step.

**Pausing.** Close the window, or press Ctrl+C in it. The connection closes at once.

**Leaving.** Tell the coordinator. We turn off your login, and you delete the software and the key. Your data never left your site, so there is nothing to get back from us. Models trained before you left include what was learned from your data. The agreement says whether they are kept.

## What your site gets

- A model trained on many more cases than any single site has, for you to use as the agreement allows.
- A report on how the shared model performs on your own held-out records, next to a model trained on your data alone.
- A comparison of which features the shared model relies on and which your local model relies on.
- Credit in publications, as set out in the agreement.

## Protections

| Protection | What it does | Status |
|---|---|---|
| Data stays on site | Training runs where the data is. Only model updates travel | In place |
| Encrypted connection | All federation traffic goes through an SSH tunnel | In place |
| No incoming access | Your computer connects out. Nothing connects in | In place |
| A personal login per site | Each site has its own key. We can turn off one site without affecting the others | Created for each site at Stage 4 |
| Public, reviewable code | Your IT team can read the training code before every run | In place |
| Tunnel-only login | The site login can open the tunnel, but can't run commands on our server | Before real data |
| Site authentication in Flower | The coordinating server accepts only registered sites | Before real data |
| Differential privacy | Adds carefully measured noise to each update, so no single record can be singled out | Planned |
| Secure aggregation | The server sees only the combined update, never a single site's | Planned |

Items marked "Before real data" aren't part of the current demo. We turn them on before any site trains on real records.

**Today's demo compared with a real deployment:**

| | Hackathon demo | Real deployment |
|---|---|---|
| Data | Public MNIST digits, downloaded by each computer | Your own dataset, in a local folder |
| Which records a site trains on | The server gives each computer a slice of MNIST | You decide: the computer reads only the folder you prepared |
| Login | A shared demo login | One login per site, limited to the tunnel |
| Security | SSH encryption | SSH encryption, plus site authentication in Flower |
| Privacy add-ons | None | Differential privacy and secure aggregation, as agreed |

## Questions sites often ask

**Can you see our patients' data?**
No. The data stays on your computer. We receive model updates and the summary scores listed in [What leaves your site](#what-leaves-your-site-and-what-never-does).

**Can you get into our network?**
Not through the connection. Your computer makes it, outward, to one address, and we can't use it to open a connection to your computer or to anything else at your site. The training code we send does run on your computer, though. The next question covers that.

**What software runs on our computer, and who controls it?**
There are two parts. You install the first part yourself: Flower, an open-source framework for federated learning, and its Python packages. The second part is the training code. The coordinating server sends it to your computer at the start of every run. So the code that trains on your data comes from us. That's why the code is public, why we announce changes in advance, and why we recommend a dedicated computer that can see only the approved data. Your IT team can review exactly what will run.

**Does a small site still count?**
Each update counts in proportion to the number of records behind it, so a site with 500 records counts less than one with 5,000. Small sites still matter. They add patients, equipment and populations the model wouldn't otherwise see, and your site report shows whether the model works for your patients.

**Our data is formatted differently from the dictionary.**
That's the most common issue, and it's what Stage 3 is for. We help you map your fields to the dictionary before anything runs.

**What if our computer crashes or loses the connection during a run?**
The run carries on with the other sites. Restart the connection, and your site joins again from the next run.

**How much staff time does this take once we're set up?**
Very little. During a scheduled run the computer has to be on and connected. Someone may want to look at the dashboard, but nobody has to.

## Checklist

**Stage 1.** First conversation
- [ ] Site lead and coordinator agree on the question
- [ ] Rough description of the data shared (no records)

**Stage 2.** Agreements
- [ ] Collaboration agreement signed
- [ ] Ethics determination received
- [ ] Privacy review completed

**Stage 3.** Data
- [ ] Data dictionary received
- [ ] Dataset converted and checked against it
- [ ] File placed on the training computer, readable only by the training account

**Stage 4.** Computer
- [ ] Training computer and account ready
- [ ] Outgoing SSH to the coordinating server allowed
- [ ] Software installed, key registered, test connection works

**Stage 5.** Rehearsal
- [ ] Site appears on the dashboard
- [ ] Practice run finished

**Stage 6.** First real run
- [ ] Date agreed
- [ ] Run finished
- [ ] Results reviewed and approved by the site lead

## For the technical contact

The commands are in [DEPLOYMENT.md](DEPLOYMENT.md), under "Join as a node". In short: set the login from the README's [Federation server](README.md#federation-server) section, run `make env`, then run `bash scripts/join_federation.sh` and leave it running.
