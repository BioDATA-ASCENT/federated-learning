"""Live training dashboard for mnist_fedavg runs. Standard library only.

Run it on the SuperLink laptop (where the ServerApp writes metrics):
    python apps/mnist_fedavg/dashboard.py            # serves on 0.0.0.0:8050
Then open http://<superlink-ip>:8050 from any laptop on the network.
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
.live{color:#1f9d55}.done{color:var(--muted)}.err{color:#d64545}
pre{white-space:pre-wrap;font-size:12px;color:var(--muted);margin:0}
</style></head><body><main>
<header><h1>FedAvg on MNIST &middot; live</h1>
<div>Run <select id="run"></select> <span id="status" class="done"></span></div></header>
<div class="tiles">
 <div class="tile"><div class="k">Round</div><div class="v" id="t-round">–</div></div>
 <div class="tile"><div class="k">Global test accuracy</div><div class="v" id="t-acc">–</div></div>
 <div class="tile"><div class="k">Global test loss</div><div class="v" id="t-loss">–</div></div>
 <div class="tile"><div class="k">Clients seen</div><div class="v" id="t-clients">–</div></div>
 <div class="tile"><div class="k">Elapsed</div><div class="v" id="t-time">–</div></div>
</div>
<div class="grid">
 <div class="card"><h2>Global model accuracy (server test set, 10k images; federated = mean of client val)</h2><div class="chart"><canvas id="c-acc"></canvas></div></div>
 <div class="card"><h2>Global model loss</h2><div class="chart"><canvas id="c-loss"></canvas></div></div>
 <div class="card"><h2>Local training loss per client (last local epoch)</h2><div class="chart"><canvas id="c-ctrain"></canvas></div></div>
 <div class="card"><h2>Global model on each client's local validation split</h2><div class="chart"><canvas id="c-cval"></canvas></div></div>
</div>
<div class="card" style="margin-top:12px"><h2>Clients (latest round)</h2><table id="clients"><thead><tr><th>Client</th><th>Node ID</th><th>Train samples</th><th>Train loss</th><th>Val acc</th><th>Local train time</th></tr></thead><tbody></tbody></table></div>
<div class="card" style="margin-top:12px"><h2>Run config</h2><pre id="cfg"></pre></div>
</main><script>
const PALETTE=["#2f6fde","#e8710a","#1f9d55","#a33ea1","#d64545","#0f9fb5","#8a6d00","#6b6b66"];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
Chart.defaults.color=css("--muted");Chart.defaults.borderColor=css("--line");Chart.defaults.font.family=getComputedStyle(document.body).fontFamily;
function mk(id,yTitle,pct){return new Chart(document.getElementById(id),{type:"line",data:{datasets:[]},
 options:{animation:false,maintainAspectRatio:false,parsing:false,interaction:{mode:"nearest",axis:"x",intersect:false},
 scales:{x:{type:"linear",title:{display:true,text:"round"},ticks:{precision:0}},y:{title:{display:true,text:yTitle},
 ticks:pct?{callback:v=>(v*100).toFixed(0)+"%"}:{}}},
 plugins:{legend:{position:"bottom",labels:{boxWidth:10}},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${pct?(c.parsed.y*100).toFixed(2)+"%":c.parsed.y.toFixed(4)}`}}}}})}
const charts={acc:mk("c-acc","accuracy",true),loss:mk("c-loss","loss",false),ctrain:mk("c-ctrain","train loss",false),cval:mk("c-cval","accuracy",true)};
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
</script></body></html>"""


def make_handler(root: Path):
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
                runs = sorted(
                    (p.name for p in root.iterdir() if (p / "metrics.jsonl").exists()),
                    reverse=True,
                ) if root.exists() else []
                return self._send(json.dumps(runs).encode(), "application/json")
            if url.path == "/api/metrics":
                run = parse_qs(url.query).get("run", [""])[0]
                path = (root / run / "metrics.jsonl").resolve()
                if not run or root.resolve() not in path.parents or not path.exists():
                    return self._send(b"[]", "application/json", 404)
                rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                return self._send(json.dumps(rows).encode(), "application/json")
            self._send(b"not found", "text/plain", 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="~/fedbio-runs/mnist_fedavg", help="the app's out-dir")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other laptops")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    root = Path(os.path.expanduser(args.dir))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    print(f"Dashboard for {root} on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
