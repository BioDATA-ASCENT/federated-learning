"""Live dashboard: the federation's nodes and their readiness, and mnist_fedavg runs.

Standard library only. Run it on the SuperLink machine (where the ServerApp
writes metrics and the SuperLink keeps its database):
    python apps/mnist_fedavg/dashboard.py            # serves on 0.0.0.0:8050
Then open http://<superlink-ip>:8050, or http://localhost:8050 through the tunnel.

The Federation panel reads the connected nodes from the SuperLink database every
second. Readiness comes from the HLCA app's check (check-only=true), which the
panel's button starts. With --auto-check N, the dashboard also starts one every
N seconds while a connected node is not ready, but never during a run.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import threading
import time
from contextlib import closing
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
RECENT_S = 600  # keep nodes that left in the last ten minutes on the list

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FedAvg Live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--accent:#2f6fde}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--fg:#ecebe7;--muted:#9a9993;--line:#33322f;--accent:#6b9bff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
h1{font-size:20px;margin:0}select{font:inherit;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--muted);font-size:12px}.tile .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:12px}
.card h2{font-size:13px;font-weight:600;margin:0 0 8px;color:var(--muted)}
.chart{position:relative;height:240px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500;font-size:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.live{color:#1f9d55}.done{color:var(--muted)}.err{color:#d64545}.warn{color:#b7791f}
.fedhead{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:8px}
.fedhead .big{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.fedhead .note{color:var(--muted);font-size:12px;flex:1;min-width:200px}
button{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid var(--line);background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
.scroll{overflow-x:auto}tr.gone td{color:var(--muted)}td.detail{font-size:12px;word-break:break-word;min-width:220px}
pre{white-space:pre-wrap;font-size:12px;color:var(--muted);margin:0}
</style></head><body><main>
<header><h1 id="title">FedAvg &middot; live</h1>
<div>Run <select id="run"></select> <span id="status" class="done"></span></div></header>
<div class="card" style="margin-bottom:16px"><h2>Federation &middot; nodes and readiness (live)</h2>
 <div class="fedhead"><span><span class="big" id="f-conn">–</span> connected</span><span><span class="big live" id="f-ready">–</span> ready</span>
  <span class="note" id="f-note"></span><button id="f-check">Check now</button></div>
 <div class="scroll"><table id="fed"><thead><tr><th>Connection</th><th>Node ID</th><th>For</th><th>Readiness</th><th>Partition</th><th>Detail</th></tr></thead><tbody></tbody></table></div></div>
<div class="tiles">
 <div class="tile"><div class="k">Round</div><div class="v" id="t-round">–</div></div>
 <div class="tile"><div class="k" id="k-acc">Global test accuracy</div><div class="v" id="t-acc">–</div></div>
 <div class="tile"><div class="k" id="k-loss">Global test loss</div><div class="v" id="t-loss">–</div></div>
 <div class="tile"><div class="k">Clients seen</div><div class="v" id="t-clients">–</div></div>
 <div class="tile"><div class="k">Elapsed</div><div class="v" id="t-time">–</div></div>
</div>
<div class="grid">
 <div class="card"><h2 id="h-acc"></h2><div class="chart"><canvas id="c-acc"></canvas></div></div>
 <div class="card"><h2 id="h-loss"></h2><div class="chart"><canvas id="c-loss"></canvas></div></div>
 <div class="card"><h2 id="h-ctrain"></h2><div class="chart"><canvas id="c-ctrain"></canvas></div></div>
 <div class="card"><h2 id="h-cval"></h2><div class="chart"><canvas id="c-cval"></canvas></div></div>
 <div class="card" id="card-class" style="display:none"><h2>F1 per cell type (federated, from the summed counts)</h2><div class="chart"><canvas id="c-class"></canvas></div></div>
 <div class="card" id="card-classbar" style="display:none"><h2>F1 per cell type, latest round</h2><div class="chart"><canvas id="c-classbar"></canvas></div></div>
</div>
<div class="card" style="margin-top:12px"><h2>Clients (latest round)</h2><div class="scroll"><table id="clients"><thead id="clients-head"></thead><tbody></tbody></table></div></div>
<div class="card" style="margin-top:12px"><h2>Run config</h2><pre id="cfg"></pre></div>
</main><script>
const PALETTE=["#2f6fde","#e8710a","#1f9d55","#a33ea1","#d64545","#0f9fb5","#8a6d00","#6b6b66","#c2185b"];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
Chart.defaults.color=css("--muted");Chart.defaults.borderColor=css("--line");Chart.defaults.font.family=getComputedStyle(document.body).fontFamily;
// Each chart carries its value format in chart.$fmt, so one canvas can show accuracy (MNIST) or F1 (HLCA).
const TICK={pct:v=>(v*100).toFixed(0)+"%",f1:v=>(+v).toFixed(2),num:v=>v};
const FMT={pct:v=>(v*100).toFixed(2)+"%",f1:v=>v.toFixed(3),num:v=>v.toFixed(4)};
function mk(id,yTitle,fmt){const ch=new Chart(document.getElementById(id),{type:"line",data:{datasets:[]},
 options:{animation:false,maintainAspectRatio:false,parsing:false,interaction:{mode:"nearest",axis:"x",intersect:false},
 scales:{x:{type:"linear",title:{display:true,text:"round"},ticks:{precision:0}},y:{title:{display:true,text:yTitle},
 ticks:{callback:function(v){return (TICK[this.chart.$fmt]||TICK[fmt])(v)}}}},
 plugins:{legend:{position:"bottom",labels:{boxWidth:10}},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${(FMT[c.chart.$fmt]||FMT[fmt])(c.parsed.y)}`}}}}});
 ch.$fmt=fmt;return ch}
const charts={acc:mk("c-acc","accuracy","pct"),loss:mk("c-loss","loss","num"),ctrain:mk("c-ctrain","train loss (log scale)","num"),
 cval:mk("c-cval","accuracy","pct"),cls:mk("c-class","F1","f1")};
// Log scale, so one client's diverging loss does not flatten the others.
charts.ctrain.options.scales.y.type="logarithmic";
charts.ctrain.options.scales.y.ticks.callback=v=>{const e=Math.log10(v);return Math.abs(e-Math.round(e))<1e-9?(+v).toPrecision(1):""};
const barChart=new Chart(document.getElementById("c-classbar"),{type:"bar",data:{labels:[],datasets:[]},
 options:{indexAxis:"y",animation:false,maintainAspectRatio:false,scales:{x:{min:0,max:1,title:{display:true,text:"F1"}}},
 plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`F1 ${c.parsed.x.toFixed(3)}`}}}}});
function view(o){document.getElementById("title").innerHTML=o.title;
 for(const k of ["acc","loss","ctrain","cval"])document.getElementById("h-"+k).textContent=o.h[k];
 document.getElementById("k-acc").textContent=o.k[0];document.getElementById("k-loss").textContent=o.k[1];
 for(const[k,f,y]of o.fmt){charts[k].$fmt=f;charts[k].options.scales.y.title.text=y}
 for(const id of ["card-class","card-classbar"])document.getElementById(id).style.display=o.classes?"":"none";
 document.getElementById("clients-head").innerHTML="<tr>"+o.cols.map(c=>`<th>${c}</th>`).join("")+"</tr>"}
const MNIST_VIEW={title:"FedAvg on MNIST &middot; live",k:["Global test accuracy","Global test loss"],classes:false,
 h:{acc:"Global model accuracy (server test set, 10k images; federated = mean of client val)",loss:"Global model loss",
  ctrain:"Local training loss per client (last local epoch)",cval:"Global model on each client's local validation split"},
 fmt:[["acc","pct","accuracy"],["cval","pct","accuracy"]],
 cols:["Client","Node ID","Train samples","Train loss","Val acc","Local train time"]};
const HLCA_VIEW={title:"FedAvg on the Human Lung Cell Atlas &middot; live",k:["Federated macro F1","Validation loss"],classes:true,
 h:{acc:"Macro F1 of the global model (summed over all clients' held-out donors)",loss:"Validation loss of the global model (weighted mean over clients)",
  ctrain:"Local training loss per client (last local epoch)",cval:"Global model's macro F1 on each client's held-out donors"},
 fmt:[["acc","f1","macro F1"],["cval","f1","macro F1"]],
 cols:["Client","Node ID","Train cells","Train loss","Val cells","Local macro F1"]};
const f1=(tp,fp,fn)=>tp.map((t,i)=>{const d=2*t+fp[i]+fn[i];return tp[i]+fn[i]>0&&d>0?2*t/d:null});
function elapsed(recs){const tl=recs.length?recs[recs.length-1].t:0;return tl>=60?`${Math.floor(tl/60)}m ${Math.round(tl%60)}s`:`${tl.toFixed(0)}s`}
function runStatus(errs,end){const st=document.getElementById("status");
 if(errs.length){st.className="err";st.textContent=`● ${errs.length} client error(s)`}
 else if(end){st.className="done";st.textContent="● finished"}else{st.className="live";st.textContent="● running"}}
function renderHLCA(recs,start,end){view(HLCA_VIEW);
 const fe=recs.filter(r=>r.type==="federated_eval"),ct=recs.filter(r=>r.type==="client_train"),ce=recs.filter(r=>r.type==="client_eval");
 const lr=recs.find(r=>r.type==="labels"),last=fe[fe.length-1],N=start?start.config["num-server-rounds"]:"?";
 const names=lr?lr.labels:(last?last.tp.map((_,i)=>`class ${i}`):[]);
 document.getElementById("t-round").textContent=`${last?last.round:0} / ${N}`;
 document.getElementById("t-acc").textContent=last?last.macro_f1.toFixed(3):"–";
 document.getElementById("t-loss").textContent=last?last.eval_loss.toFixed(4):"–";
 document.getElementById("t-clients").textContent=new Set(ct.map(r=>r.node)).size;
 document.getElementById("t-time").textContent=elapsed(recs);
 runStatus(recs.filter(r=>r.type.endsWith("_error")),end);
 setSeries(charts.acc,[{label:"federated macro F1 (summed counts)",data:fe.map(r=>({x:r.round,y:r.macro_f1}))},
  {label:"mean of the clients' own macro F1",data:fe.map(r=>({x:r.round,y:r.local_macro_f1})),dash:[5,4]}]);
 setSeries(charts.loss,[{label:"validation loss",data:fe.map(r=>({x:r.round,y:r.eval_loss}))}]);
 setSeries(charts.ctrain,byClient(ct,"train_loss"));setSeries(charts.cval,byClient(ce,"local_macro_f1"));
 const per=fe.map(r=>({round:r.round,f:f1(r.tp,r.fp,r.fn)}));
 setSeries(charts.cls,names.map((n,i)=>({label:n,data:per.filter(p=>p.f[i]!==null).map(p=>({x:p.round,y:p.f[i]}))})));
 const lf=per.length?per[per.length-1].f:[];
 barChart.data.labels=names;barChart.data.datasets=[{data:lf.map(v=>v??0),
  backgroundColor:lf.map(v=>v===null?css("--line"):v<0.5?"#d64545":v<0.9?"#e8710a":"#1f9d55")}];barChart.update();
 const latest=new Map();ct.forEach(r=>latest.set(r.node,{...(latest.get(r.node)||{}),t:r}));ce.forEach(r=>latest.set(r.node,{...(latest.get(r.node)||{}),e:r}));
 document.querySelector("#clients tbody").innerHTML=[...latest.entries()].sort((a,b)=>label(a[1].t||a[1].e).localeCompare(label(b[1].t||b[1].e))).map(([n,v],i)=>
  `<tr><td><span class="dot" style="background:${PALETTE[i%PALETTE.length]}"></span>${label(v.t||v.e)}</td><td>${n}</td>
  <td>${v.t?v.t["num-examples"]:"–"}</td><td>${v.t?v.t.train_loss.toFixed(4):"–"}</td><td>${v.e?v.e["num-examples"]:"–"}</td>
  <td class="${v.e&&v.e.local_macro_f1<0.5?"err":""}">${v.e?v.e.local_macro_f1.toFixed(3):"–"}</td></tr>`).join("");
 document.getElementById("cfg").textContent=start?JSON.stringify(start.config,null,1):""}
function setSeries(chart,series){chart.data.datasets=series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||PALETTE[i%PALETTE.length],
 backgroundColor:s.color||PALETTE[i%PALETTE.length],borderWidth:2,pointRadius:2,tension:0.15,borderDash:s.dash||[]}));chart.update()}
const runSel=document.getElementById("run");let current=null;
async function loadRuns(){const runs=await (await fetch("api/runs")).json();const prev=runSel.value;
 if(runs.join()!==[...runSel.options].map(o=>o.value).join()){runSel.innerHTML=runs.map(r=>`<option>${r}</option>`).join("");
  if(runs.length&&(!prev||current===null||prev===runs[1]))runSel.value=runs[0];else if(prev)runSel.value=prev;}
 current=runSel.value||null}
runSel.onchange=()=>{current=runSel.value;refresh()};
function label(r){return r["partition-id"]!==undefined?`client ${r["partition-id"]}`:`node ${String(r.node).slice(-4)}`}
function byClient(rows,key){const m=new Map();rows.forEach(r=>{const l=label(r);if(!m.has(l))m.set(l,[]);m.get(l).push({x:r.round,y:r[key]})});
 return [...m.entries()].sort().map(([l,d])=>({label:l,data:d}))}
async function refresh(){await loadRuns();if(!current)return;
 const recs=await (await fetch("api/metrics?run="+encodeURIComponent(current))).json();
 const start=recs.find(r=>r.type==="run_start"),end=recs.find(r=>r.type==="run_end");
 if(recs.some(r=>r.type==="labels"||r.type==="federated_eval")||(start&&"gene-list" in start.config))return renderHLCA(recs,start,end);
 view(MNIST_VIEW);
 const g=recs.filter(r=>r.type==="global_eval"),fe=recs.filter(r=>r.type==="fed_eval");
 const ct=recs.filter(r=>r.type==="client_train"),ce=recs.filter(r=>r.type==="client_eval");
 const errs=recs.filter(r=>r.type.endsWith("_error"));
 const N=start?start.config["num-server-rounds"]:"?",last=g[g.length-1];
 document.getElementById("t-round").textContent=`${last?last.round:0} / ${N}`;
 document.getElementById("t-acc").textContent=last?(last.test_acc*100).toFixed(2)+"%":"–";
 document.getElementById("t-loss").textContent=last?last.test_loss.toFixed(4):"–";
 document.getElementById("t-clients").textContent=new Set(ct.map(r=>r.node)).size;
 const tl=recs.length?recs[recs.length-1].t:0;document.getElementById("t-time").textContent=tl>=60?`${Math.floor(tl/60)}m ${Math.round(tl%60)}s`:`${tl.toFixed(0)}s`;
 const st=document.getElementById("status");
 if(errs.length){st.className="err";st.textContent=`● ${errs.length} client error(s)`}
 else if(end){st.className="done";st.textContent="● finished"}else{st.className="live";st.textContent="● running"}
 setSeries(charts.acc,[{label:"global (server test set)",data:g.map(r=>({x:r.round,y:r.test_acc}))},
  {label:"federated (client val, weighted)",data:fe.map(r=>({x:r.round,y:r.eval_acc})),dash:[5,4]}]);
 setSeries(charts.loss,[{label:"global (server test set)",data:g.map(r=>({x:r.round,y:r.test_loss}))},
  {label:"federated (client val, weighted)",data:fe.map(r=>({x:r.round,y:r.eval_loss})),dash:[5,4]}]);
 setSeries(charts.ctrain,byClient(ct,"train_loss"));setSeries(charts.cval,byClient(ce,"eval_acc"));
 const latest=new Map();ct.forEach(r=>latest.set(r.node,{...(latest.get(r.node)||{}),t:r}));ce.forEach(r=>latest.set(r.node,{...(latest.get(r.node)||{}),e:r}));
 document.querySelector("#clients tbody").innerHTML=[...latest.entries()].map(([n,v],i)=>`<tr><td><span class="dot" style="background:${PALETTE[i%PALETTE.length]}"></span>${v.t?label(v.t):""}</td><td>${n}</td>
  <td>${v.t?v.t["num-examples"]:"–"}</td><td>${v.t?v.t.train_loss.toFixed(4):"–"}</td><td>${v.e?(v.e.eval_acc*100).toFixed(2)+"%":"–"}</td><td>${v.t?v.t.train_time_s.toFixed(1)+"s":"–"}</td></tr>`).join("");
 document.getElementById("cfg").textContent=start?JSON.stringify(start.config,null,1):"";}
refresh();setInterval(refresh,2000);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ago=s=>s<60?`${Math.max(0,Math.round(s))}s`:s<3600?`${Math.floor(s/60)}m ${Math.round(s%60)}s`:`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`;
const RCLS={"READY":"live","NOT READY":"err","DUPLICATE":"err","ERROR":"err","NO REPLY":"warn"};
const btn=document.getElementById("f-check");
btn.onclick=async()=>{btn.disabled=true;const r=await (await fetch("api/check",{method:"POST"})).json();
 if(!r.started)document.getElementById("f-note").textContent=r.reason;refreshFed()};
async function refreshFed(){let f;
 try{f=await (await fetch("api/federation")).json()}catch(e){document.getElementById("f-note").textContent="dashboard unreachable";return}
 const on=f.nodes.filter(n=>n.online),rd=n=>f.readiness[n.node];
 document.getElementById("f-conn").textContent=on.length;
 document.getElementById("f-ready").textContent=on.filter(n=>rd(n)&&rd(n).status==="READY").length;
 const c=f.check,parts=[];
 if(f.error)parts.push(f.error);
 parts.push(c.running?"checking the nodes…":c.checked_at?`last check ${ago(f.now-c.checked_at)} ago`:"no readiness check yet");
 if(f.active_run&&!c.running)parts.push(`run in progress (${f.active_run}), checks paused`);
 else if(c.auto)parts.push(`auto-check every ${c.auto}s while a node isn't ready`);
 document.getElementById("f-note").textContent=parts.join(" · ");
 btn.disabled=c.running||!!f.active_run;
 document.querySelector("#fed tbody").innerHTML=f.nodes.map(n=>{const r=rd(n);
  const st=n.online?(r?r.status:"not checked"):"–",cls=n.online?(RCLS[st]||"done"):"";
  return `<tr class="${n.online?"":"gone"}"><td><span class="dot" style="background:${n.online?"#1f9d55":"var(--muted)"}"></span>${n.online?"online":"left"}</td>
  <td>${n.node}</td><td>${n.online?ago(f.now-n.since):ago(f.now-n.left)+" ago"}</td><td class="${cls}">${esc(st)}</td>
  <td>${n.online&&r?esc(r.partition):""}</td><td class="detail">${n.online&&r?esc(r.detail):""}</td></tr>`}).join("")||
  `<tr><td colspan="6" class="done">No nodes connected</td></tr>`}
refreshFed();setInterval(refreshFed,1000);
</script></body></html>"""


def _ts(value) -> float:
    """SuperLink timestamps are ISO strings; 0.0 when missing."""
    try:
        return datetime.fromisoformat(str(value)).timestamp() if value else 0.0
    except ValueError:
        return 0.0


class Federation:
    """Connected nodes from the SuperLink database, and the readiness checks."""

    def __init__(self, db: Path, readiness: Path, connection: str, auto_every: float, log: Path):
        self.db, self.readiness_path, self.log = db, readiness, log
        self.cmd = ["flwr", "run", "apps/hlca_fedavg", connection, "--stream", "-c", "check-only=true"]
        self.auto_every = auto_every
        self.proc: subprocess.Popen | None = None
        self.last_end = 0.0
        self.lock = threading.Lock()
        if auto_every > 0:
            threading.Thread(target=self._auto, daemon=True).start()

    def _query(self, sql: str) -> list[tuple]:
        with closing(sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=1)) as c:
            return c.execute(sql).fetchall()

    def nodes(self) -> list[dict]:
        now, out = time.time(), []
        rows = self._query(
            "SELECT node_id, status, last_activated_at, last_deactivated_at, unregistered_at, online_until FROM node"
        )
        for nid, status, activated, deactivated, unregistered, until in rows:
            online = status == "online" and (until or 0) > now
            left = 0.0 if online else max(_ts(unregistered), _ts(deactivated), min(until or 0, now))
            if online or now - left < RECENT_S:
                out.append({"node": str(nid % 2**64), "online": online, "since": _ts(activated), "left": left})
        return sorted(out, key=lambda n: (not n["online"], n["since"] if n["online"] else -n["left"]))

    def active_run(self) -> str | None:
        """The app of an unfinished run started in the last six hours, if any."""
        rows = self._query(
            "SELECT r.fab_id, t.pending_at FROM run r JOIN task t ON t.task_id = r.primary_task_id "
            "WHERE t.type = 'flwr-serverapp' AND (t.finished_at IS NULL OR t.finished_at = '')"
        )
        recent = [fab for fab, pending in rows if time.time() - _ts(pending) < 6 * 3600]
        return recent[0] if recent else None

    def readiness(self) -> dict:
        try:
            return json.loads(self.readiness_path.read_text())
        except (OSError, ValueError):
            return {}

    def running(self) -> bool:
        with self.lock:
            if self.proc is not None and self.proc.poll() is not None:
                self.proc, self.last_end = None, time.time()
            return self.proc is not None

    def start_check(self) -> tuple[bool, str]:
        if self.running():
            return False, "A check is already running."
        busy = self.active_run()
        if busy:
            return False, f"A run is in progress ({busy}). Check again when it finishes."
        with self.lock:
            log = self.log.open("a")
            self.proc = subprocess.Popen(self.cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        return True, "Check started."

    def _auto(self) -> None:
        while True:
            time.sleep(2)
            try:
                if self.running() or time.time() - self.last_end < self.auto_every:
                    continue
                checked = {r["node"]: r["status"] for r in self.readiness().get("nodes", [])}
                online = [n["node"] for n in self.nodes() if n["online"]]
                if any(checked.get(n) != "READY" for n in online):
                    self.start_check()
                    self.last_end = time.time()  # space the attempts even if the start was refused
            except Exception as e:  # noqa: BLE001  keep the loop alive; the page shows errors
                print(f"auto-check: {e}", flush=True)

    def snapshot(self) -> dict:
        snap = {"now": time.time(), "nodes": [], "readiness": {}, "active_run": None,
                "check": {"running": self.running(), "checked_at": None, "auto": self.auto_every}}
        try:
            snap["nodes"], snap["active_run"] = self.nodes(), self.active_run()
        except sqlite3.Error as e:
            snap["error"] = f"cannot read {self.db}: {e}"
        r = self.readiness()
        snap["check"]["checked_at"] = r.get("checked_at")
        snap["readiness"] = {row["node"]: row for row in r.get("nodes", [])}
        return snap


def make_handler(root: Path, fed: Federation):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if url.path == "/api/runs":
                # Runs of every app under root (<app>/<timestamp>), newest first.
                # A root that is itself one app's out-dir also works.
                runs = []
                for p in root.iterdir() if root.exists() else []:
                    if (p / "metrics.jsonl").exists():
                        runs.append(p.name)
                    elif p.is_dir():
                        runs += [f"{p.name}/{q.name}" for q in p.iterdir() if (q / "metrics.jsonl").exists()]
                runs.sort(key=lambda r: r.rsplit("/", 1)[-1], reverse=True)
                return self._send(json.dumps(runs).encode(), "application/json")
            if url.path == "/api/metrics":
                run = parse_qs(url.query).get("run", [""])[0]
                path = (root / run / "metrics.jsonl").resolve()
                if not run or root.resolve() not in path.parents or not path.exists():
                    return self._send(b"[]", "application/json", 404)
                rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                return self._send(json.dumps(rows).encode(), "application/json")
            if url.path == "/api/federation":
                return self._send(json.dumps(fed.snapshot()).encode(), "application/json")
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if urlparse(self.path).path == "/api/check":
                started, reason = fed.start_check()
                return self._send(json.dumps({"started": started, "reason": reason}).encode(), "application/json")
            self._send(b"not found", "text/plain", 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="~/fedbio-runs", help="parent of the apps' out-dirs (mnist_fedavg, hlca_fedavg)")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other laptops")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--superlink-db", default="~/fl-logs/superlink.db", help="the SuperLink's --database")
    ap.add_argument("--readiness", default="~/fedbio-runs/hlca_fedavg/readiness.json",
                    help="written by: flwr run apps/hlca_fedavg <connection> -c check-only=true")
    ap.add_argument("--connection", default="aws", help="flwr connection used to start readiness checks")
    ap.add_argument("--auto-check", type=float, default=30,
                    help="seconds between automatic checks while a node is not ready; 0 = only the button")
    args = ap.parse_args()
    root = Path(os.path.expanduser(args.dir))
    fed = Federation(
        Path(os.path.expanduser(args.superlink_db)),
        Path(os.path.expanduser(args.readiness)),
        args.connection,
        args.auto_check,
        Path(os.path.expanduser("~/fl-logs/readiness-check.log")),
    )
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(root, fed))
    print(f"Dashboard for {root} on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
