"""
b76 visualize — interaktiv HTML-graf via pyvis
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nouse.field.surface import FieldSurface

# Domänfärger — robust även när antalet domäner överstiger paletten.
_PALETTE = [
    "#4e9af1", "#f16b4e", "#4ef19a", "#f1c44e", "#b04ef1",
    "#f14eb0", "#4ef1e8", "#f1f14e", "#8af14e", "#f14e4e",
    "#4e4ef1", "#f1a04e", "#4ef160", "#a04ef1", "#4ef1c4",
]

_DOMAIN_COLORS: dict[str, str] = {}


def _color(domain: str) -> str:
    if domain in _DOMAIN_COLORS:
        return _DOMAIN_COLORS[domain]

    idx = len(_DOMAIN_COLORS)
    if idx < len(_PALETTE):
        color = _PALETTE[idx]
    else:
        # Deterministisk fallback för fler domäner än palettfärger.
        hue = abs(hash(domain)) % 360
        color = f"hsl({hue}, 68%, 58%)"

    _DOMAIN_COLORS[domain] = color
    return color


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict] = []
    for e in edges:
        key = (
            str(e.get("from", "")),
            str(e.get("to", "")),
            str(e.get("label", "")),
            str(e.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _node_sizes(node_ids: list[str], edges: list[dict]) -> dict[str, float]:
    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        src = str(e.get("from", ""))
        tgt = str(e.get("to", ""))
        if src:
            degree[src] += 1
        if tgt:
            degree[tgt] += 1

    max_degree = max(degree.values(), default=1)
    sizes: dict[str, float] = {}
    for nid in node_ids:
        d = degree.get(nid, 0)
        scaled = 11.0 + 18.0 * (d / max_degree) ** 0.5
        sizes[nid] = max(10.0, min(30.0, scaled))
    return sizes


def _net_options() -> str:
    return """
    {
      "nodes": {
        "font": { "size": 11 },
        "shape": "dot"
      },
      "edges": {
        "font": { "size": 9, "align": "middle" },
        "smooth": { "type": "dynamic", "roundness": 0.28 },
        "selectionWidth": 3
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "physics": {
        "enabled": true,
        "stabilization": { "iterations": 180 }
      }
    }
    """


def _header_html(total_nodes: int, total_edges: int, legend_html: str) -> str:
    return f"""
<div style="position:fixed;top:10px;left:10px;z-index:9999;
     background:#111;padding:10px 14px;border-radius:8px;
     font-family:monospace;font-size:12px;color:#ccc;
     border:1px solid #444;max-height:80vh;overflow-y:auto;max-width:300px">
  <b style="color:#4e9af1">b76 graph</b><br>
  {total_nodes} noder &nbsp;·&nbsp; {total_edges} kanter<br><br>
  <b>Domäner:</b><br>{legend_html}
</div>
"""


def _controls_html(domains_used: list[str]) -> str:
    options = "".join(
        f'<option value="{escape(d)}">{escape(d)}</option>' for d in domains_used
    )

    return f"""
<div id="b76-controls" style="position:fixed;top:10px;right:10px;z-index:9999;
     background:#10131e;padding:10px 12px;border-radius:8px;
     font-family:monospace;font-size:12px;color:#d7dbe8;
     border:1px solid #34405a;max-width:300px;min-width:270px">
  <b style="color:#7ec8ff">Analys</b><br>
  <div style="margin-top:8px">Sok/fokus nod</div>
  <input id="b76-search" list="b76-node-list" placeholder="t.ex. LLM"
         style="width:100%;background:#0c1020;border:1px solid #314067;color:#d7dbe8;padding:4px" />
  <datalist id="b76-node-list"></datalist>
  <div style="margin-top:6px;display:flex;gap:6px">
    <button onclick="b76FocusNode()" style="flex:1">Fokusera</button>
    <button onclick="b76ResetAll()" style="flex:1">Reset</button>
  </div>

  <div style="margin-top:10px">Domänfilter</div>
  <select id="b76-domain-filter"
          style="width:100%;background:#0c1020;border:1px solid #314067;color:#d7dbe8;padding:4px">
    <option value="">alla domäner</option>
    {options}
  </select>
  <div style="margin-top:6px;display:flex;gap:6px">
    <button onclick="b76ApplyDomainFilter()" style="flex:1">Filtrera</button>
    <button onclick="b76TogglePhysics()" style="flex:1">Fysik av/på</button>
  </div>

  <div style="margin-top:10px">Minsta koppling</div>
  <select id="b76-path-from"
          style="width:100%;background:#0c1020;border:1px solid #314067;color:#d7dbe8;padding:4px"></select>
  <select id="b76-path-to"
          style="width:100%;margin-top:4px;background:#0c1020;border:1px solid #314067;color:#d7dbe8;padding:4px"></select>
  <button onclick="b76HighlightShortestPath()" style="width:100%;margin-top:6px">Visa minsta koppling</button>
  <div id="b76-path-result" style="margin-top:6px;color:#9fb3d6"></div>
</div>

<script type="text/javascript">
(function () {{
  if (!window.nodes || !window.edges || !window.network) return;

  const originalNodes = nodes.get();
  const originalEdges = edges.get();
  const originalNodeById = new Map(originalNodes.map(n => [n.id, n]));
  const originalEdgeById = new Map(originalEdges.map(e => [e.id, e]));

  function visibleNodeIds() {{
    return new Set(nodes.get().filter(n => !n.hidden).map(n => n.id));
  }}

  function refreshLists() {{
    const ns = nodes.get().filter(n => !n.hidden).sort((a, b) =>
      String(a.label).localeCompare(String(b.label), 'sv'));
    const list = document.getElementById('b76-node-list');
    const from = document.getElementById('b76-path-from');
    const to = document.getElementById('b76-path-to');
    if (!list || !from || !to) return;

    list.innerHTML = ns.map(n => `<option value="${{String(n.label).replace(/"/g, '&quot;')}}"></option>`).join('');
    const opts = ['<option value="">valj nod</option>']
      .concat(ns.map(n => `<option value="${{String(n.id).replace(/"/g, '&quot;')}}">${{String(n.label)}} </option>`));
    from.innerHTML = opts.join('');
    to.innerHTML = opts.join('');
  }}

  function clearHighlights() {{
    const nodeReset = nodes.get().map(n => {{
      const base = originalNodeById.get(n.id) || n;
      return {{ id: n.id, color: base.color, size: base.size, hidden: n.hidden }};
    }});
    nodes.update(nodeReset);

    const edgeReset = edges.get().map(e => {{
      const base = originalEdgeById.get(e.id) || e;
      return {{ id: e.id, color: base.color, width: base.width, hidden: e.hidden }};
    }});
    edges.update(edgeReset);
  }}

  window.b76ApplyDomainFilter = function () {{
    const selected = document.getElementById('b76-domain-filter')?.value || '';
    clearHighlights();

    const nodeUpdates = originalNodes.map(n => ({{
      id: n.id,
      hidden: selected ? String(n.group || '') !== selected : false
    }}));
    nodes.update(nodeUpdates);

    const visible = visibleNodeIds();
    const edgeUpdates = originalEdges.map(e => ({{
      id: e.id,
      hidden: !(visible.has(e.from) && visible.has(e.to))
    }}));
    edges.update(edgeUpdates);

    refreshLists();
    document.getElementById('b76-path-result').textContent = '';
  }};

  window.b76FocusNode = function () {{
    clearHighlights();
    const q = (document.getElementById('b76-search')?.value || '').trim().toLowerCase();
    if (!q) return;

    const visible = nodes.get().filter(n => !n.hidden);
    const hit = visible.find(n => String(n.label || '').toLowerCase() === q)
      || visible.find(n => String(n.label || '').toLowerCase().includes(q));
    if (!hit) return;

    nodes.update([{{ id: hit.id, color: '#ffd166', size: Math.max(24, Number(hit.size || 14) + 10) }}]);
    network.selectNodes([hit.id]);
    network.focus(hit.id, {{ scale: 1.6, animation: true }});
  }};

  window.b76TogglePhysics = function () {{
    const enabled = network.physics.options.enabled !== false;
    network.setOptions({{ physics: {{ enabled: !enabled }} }});
  }};

  window.b76ResetAll = function () {{
    nodes.update(originalNodes.map(n => ({{ id: n.id, hidden: false, color: n.color, size: n.size }})));
    edges.update(originalEdges.map(e => ({{ id: e.id, hidden: false, color: e.color, width: e.width }})));
    network.fit({{ animation: true }});
    document.getElementById('b76-path-result').textContent = '';
    const domainSelect = document.getElementById('b76-domain-filter');
    if (domainSelect) domainSelect.value = '';
    refreshLists();
  }};

  window.b76HighlightShortestPath = function () {{
    clearHighlights();
    const from = document.getElementById('b76-path-from')?.value || '';
    const to = document.getElementById('b76-path-to')?.value || '';
    const out = document.getElementById('b76-path-result');
    if (!from || !to || !out) return;

    const visNodes = visibleNodeIds();
    const visEdges = edges.get().filter(e => !e.hidden && visNodes.has(e.from) && visNodes.has(e.to));

    const adj = new Map();
    for (const n of visNodes) adj.set(n, []);
    for (const e of visEdges) {{
      if (!adj.has(e.from)) adj.set(e.from, []);
      adj.get(e.from).push({{ to: e.to, edgeId: e.id }});
    }}

    const parent = new Map();
    const parentEdge = new Map();
    const queue = [from];
    parent.set(from, null);

    while (queue.length > 0 && !parent.has(to)) {{
      const cur = queue.shift();
      const neighbors = adj.get(cur) || [];
      for (const nb of neighbors) {{
        if (parent.has(nb.to)) continue;
        parent.set(nb.to, cur);
        parentEdge.set(nb.to, nb.edgeId);
        queue.push(nb.to);
      }}
    }}

    if (!parent.has(to)) {{
      out.textContent = 'Ingen koppling hittades i nuvarande filter.';
      return;
    }}

    const pathNodes = [];
    const pathEdges = [];
    let cur = to;
    while (cur !== null) {{
      pathNodes.push(cur);
      const eId = parentEdge.get(cur);
      if (eId !== undefined) pathEdges.push(eId);
      cur = parent.get(cur);
    }}
    pathNodes.reverse();
    pathEdges.reverse();

    nodes.update(pathNodes.map(id => {{
      const base = originalNodeById.get(id);
      return {{ id, color: '#ffde7d', size: Math.max(24, Number(base?.size || 14) + 8) }};
    }}));

    edges.update(pathEdges.map(id => {{
      const base = originalEdgeById.get(id);
      return {{ id, color: {{ color: '#ffd166', highlight: '#ffe29a' }}, width: Math.max(4, Number(base?.width || 2) + 2) }};
    }}));

    network.selectNodes(pathNodes);
    network.fit({{ nodes: pathNodes, animation: true }});
    out.textContent = `Hittad koppling med ${{pathEdges.length}} hopp.`;
  }};

  refreshLists();
}})();
</script>
"""


def _inject_html_shell(
    out: Path,
    total_nodes: int,
    total_edges: int,
    legend_html: str,
    domains_used: list[str],
) -> None:
    html = out.read_text(encoding="utf-8")
    html = html.replace("<body>", f"<body>{_header_html(total_nodes, total_edges, legend_html)}", 1)
    html = html.replace("</body>", f"{_controls_html(domains_used)}</body>", 1)
    out.write_text(html, encoding="utf-8")


def build_html(
    field: "FieldSurface",
    output_path: str | Path,
    domain: str | None = None,
    min_strength: float = 0.0,
    max_nodes: int = 200,
) -> str:
    from pyvis.network import Network

    concepts = field.concepts(domain=domain)[:max_nodes]
    node_names = {c["name"] for c in concepts}

    edges_payload: list[dict] = []
    for name in node_names:
        for rel in field.out_relations(name):
            tgt = rel["target"]
            if tgt not in node_names:
                continue
            strength = float(rel.get("strength") or 0.0)
            if strength < min_strength:
                continue
            width = max(1.0, min(6.0, 1.0 + strength * 2))
            why = rel.get("why") or ""
            label = rel.get("type") or ""
            edges_payload.append(
                {
                    "from": name,
                    "to": tgt,
                    "title": f"{label}<br>{why}" if why else label,
                    "label": label if len(label) < 20 else "",
                    "width": width,
                    "arrows": "to",
                    "color": {"color": "#888888", "highlight": "#ffffff"},
                }
            )

    edges_payload = _dedupe_edges(edges_payload)
    sizes = _node_sizes([c["name"] for c in concepts], edges_payload)

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
        damping=0.09,
    )

    for c in concepts:
        dom = c.get("domain", "okänd")
        color = _color(dom)
        name = c["name"]
        net.add_node(
            name,
            label=name,
            group=dom,
            title=f"<b>{name}</b><br>domän: {dom}",
            color=color,
            size=sizes.get(name, 14.0),
        )

    for e in edges_payload:
        net.add_edge(
            e["from"],
            e["to"],
            title=e["title"],
            label=e["label"],
            width=e["width"],
            arrows=e["arrows"],
            color=e["color"],
        )

    domains_used = sorted({c.get("domain", "okänd") for c in concepts})
    legend_html = "<br>".join(
        f'<span style="color:{_color(d)}">■</span> {escape(d)}' for d in domains_used
    )

    net.set_options(_net_options())

    out = Path(output_path)
    net.save_graph(str(out))

    _inject_html_shell(
        out=out,
        total_nodes=len(concepts),
        total_edges=len(edges_payload),
        legend_html=legend_html,
        domains_used=domains_used,
    )
    return str(out)


def build_html_from_data(
    data: dict,
    output_path: str | Path,
    domain: str | None = None,
    min_strength: float = 0.0,
) -> str:
    """
    Bygg HTML-graf från API-data (dict med 'nodes', 'edges', 'stats').
    Ingen direkt DB-access krävs — används när daemon kör.
    """
    from pyvis.network import Network

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if domain:
        nodes = [n for n in nodes if n.get("group") == domain]
    kept = {n["id"] for n in nodes}

    edges_payload: list[dict] = []
    for e in edges:
        if e["from"] not in kept or e["to"] not in kept:
            continue
        strength = float(e.get("value") or 0.0)
        if strength < min_strength:
            continue
        width = max(1.0, min(6.0, 1.0 + strength * 2))
        label = e.get("label", "")
        edges_payload.append(
            {
                "from": e["from"],
                "to": e["to"],
                "title": label,
                "label": label if len(label) < 20 else "",
                "width": width,
                "arrows": "to",
                "color": {"color": "#888888", "highlight": "#ffffff"},
            }
        )

    edges_payload = _dedupe_edges(edges_payload)
    sizes = _node_sizes([n["id"] for n in nodes], edges_payload)

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
        damping=0.09,
    )

    for n in nodes:
        dom = n.get("group", "okänd")
        node_id = n["id"]
        color = _color(dom)
        net.add_node(
            node_id,
            label=n.get("label", node_id),
            group=dom,
            color=color,
            size=sizes.get(node_id, 14.0),
            title=f"<b>{node_id}</b><br>domän: {dom}",
        )

    for e in edges_payload:
        net.add_edge(
            e["from"],
            e["to"],
            label=e["label"],
            title=e["title"],
            width=e["width"],
            arrows=e["arrows"],
            color=e["color"],
        )

    domains_used = sorted({n.get("group", "okänd") for n in nodes})
    legend_html = "<br>".join(
        f'<span style="color:{_color(d)}">■</span> {escape(d)}' for d in domains_used
    )

    net.set_options(_net_options())

    out = Path(output_path)
    net.save_graph(str(out))
    _inject_html_shell(
        out=out,
        total_nodes=len(nodes),
        total_edges=len(edges_payload),
        legend_html=legend_html,
        domains_used=domains_used,
    )
    return str(out)
