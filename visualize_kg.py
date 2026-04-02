"""
知识图谱可视化：生成交互式 HTML（基于 D3.js，无需额外依赖）

用法：
    python visualize_kg.py                              # 默认 output/entities/img_0010.json
    python visualize_kg.py output/entities/img_0007.json
    python visualize_kg.py output/entities/img_0010.json --out kg.html
"""

import argparse
import json
import os
import re
import webbrowser


def build_graph(data: dict) -> tuple[list, list]:
    """从 entity json 构建节点和边列表。"""
    img_entity_names: set[str] = {e["name"].strip().lower() for e in data.get("entities", [])}

    # 辅助：判断节点是否在图中（精确 + 包含匹配）
    def is_in_image(name: str) -> bool:
        key = name.strip().lower()
        if key in img_entity_names:
            return True
        for img_name in img_entity_names:
            if img_name in key or key in img_name:
                return True
        return False

    # 收集所有节点
    node_map: dict[str, dict] = {}

    def ensure_node(name: str):
        key = name.strip().lower()
        if key not in node_map:
            node_map[key] = {
                "id": key,
                "label": name.strip(),
                "in_image": is_in_image(name),
            }
        return key

    edges = []
    seen_edges: set[tuple] = set()

    for t in data.get("triples", []):
        head = t.get("head", "").strip()
        tail = t.get("tail", "").strip()
        relation = t.get("relation", "").strip()
        if not head or not tail or head.lower() == tail.lower():
            continue
        h_key = ensure_node(head)
        t_key = ensure_node(tail)
        edge_key = (h_key, relation.lower(), t_key)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append({
            "source": h_key,
            "target": t_key,
            "relation": relation,
            "fact": t.get("fact", ""),
            "source_type": t.get("source", ""),
        })

    nodes = list(node_map.values())
    return nodes, edges


def generate_html(data: dict, nodes: list, edges: list) -> str:
    img_id = data.get("img_id", "unknown")
    image_desc = data.get("image_description", "")
    n_in_image = sum(1 for n in nodes if n["in_image"])
    n_external = len(nodes) - n_in_image
    n_cross = sum(1 for e in edges if e.get("source_type") == "cross_entity")

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>知识图谱 · {img_id}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #0f1117; color: #e0e0e0; }}

  #header {{
    padding: 16px 24px;
    background: #1a1d27;
    border-bottom: 1px solid #2e3248;
    display: flex;
    align-items: flex-start;
    gap: 24px;
  }}
  #header h1 {{ font-size: 18px; color: #fff; white-space: nowrap; }}
  #desc {{ font-size: 12px; color: #8a8fa8; line-height: 1.6; max-width: 700px; }}
  #stats {{ display: flex; gap: 16px; flex-shrink: 0; }}
  .stat-badge {{
    background: #252836;
    border: 1px solid #2e3248;
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    min-width: 80px;
  }}
  .stat-badge .val {{ font-size: 22px; font-weight: 700; }}
  .stat-badge .lbl {{ font-size: 11px; color: #6b7280; margin-top: 2px; }}
  .val-img {{ color: #f59e0b; }}
  .val-ext {{ color: #60a5fa; }}
  .val-edge {{ color: #a78bfa; }}
  .val-cross {{ color: #34d399; }}

  #legend {{
    padding: 10px 24px;
    background: #151720;
    border-bottom: 1px solid #2e3248;
    display: flex;
    gap: 24px;
    align-items: center;
    font-size: 12px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .dot-img {{ background: #f59e0b; box-shadow: 0 0 6px #f59e0b88; }}
  .dot-ext {{ background: #60a5fa; }}
  .line-cross {{ width: 24px; height: 2px; background: #34d399; display: inline-block; border-radius: 2px; }}
  .line-normal {{ width: 24px; height: 1px; background: #4b5563; display: inline-block; }}

  #controls {{
    padding: 8px 24px;
    background: #151720;
    border-bottom: 1px solid #2e3248;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 12px;
  }}
  #controls label {{ color: #8a8fa8; }}
  #search-input {{
    background: #252836;
    border: 1px solid #2e3248;
    border-radius: 6px;
    color: #e0e0e0;
    padding: 4px 10px;
    font-size: 12px;
    width: 200px;
    outline: none;
  }}
  #search-input:focus {{ border-color: #f59e0b; }}
  .ctrl-btn {{
    background: #252836;
    border: 1px solid #2e3248;
    border-radius: 6px;
    color: #c0c4d0;
    padding: 4px 12px;
    font-size: 12px;
    cursor: pointer;
  }}
  .ctrl-btn:hover {{ background: #2e3248; color: #fff; }}

  #main {{ display: flex; height: calc(100vh - 140px); }}
  #canvas {{ flex: 1; position: relative; overflow: hidden; }}
  svg {{ width: 100%; height: 100%; }}

  #sidebar {{
    width: 300px;
    background: #1a1d27;
    border-left: 1px solid #2e3248;
    overflow-y: auto;
    padding: 16px;
    font-size: 13px;
  }}
  #sidebar h3 {{ color: #f59e0b; font-size: 14px; margin-bottom: 12px; }}
  #sidebar-content {{ color: #9ca3af; line-height: 1.7; }}
  .detail-section {{ margin-bottom: 12px; }}
  .detail-label {{ color: #6b7280; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
  .detail-value {{ color: #e0e0e0; }}
  .detail-tag {{
    display: inline-block;
    background: #252836;
    border: 1px solid #2e3248;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 2px;
    color: #c0c4d0;
  }}
  .tag-in-image {{ border-color: #f59e0b88; color: #f59e0b; }}
  .edge-item {{
    background: #252836;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    border-left: 3px solid #4b5563;
    font-size: 12px;
  }}
  .edge-item.cross {{ border-left-color: #34d399; }}
  .edge-rel {{ color: #a78bfa; font-weight: 600; }}
  .edge-fact {{ color: #9ca3af; margin-top: 4px; font-size: 11px; }}

  .node {{ cursor: pointer; }}
  .node circle {{ stroke-width: 2px; transition: r 0.15s; }}
  .node:hover circle {{ stroke-width: 3px; }}
  .node text {{ pointer-events: none; font-size: 11px; fill: #e0e0e0; }}
  .link {{ stroke-opacity: 0.5; }}
  .link-label {{ font-size: 9px; fill: #6b7280; pointer-events: none; }}
  .highlighted circle {{ stroke: #fff !important; stroke-width: 3px !important; }}
  .dimmed {{ opacity: 0.15; }}

  #tooltip {{
    position: absolute;
    background: #1a1d27;
    border: 1px solid #2e3248;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    max-width: 240px;
    z-index: 100;
  }}
  #tooltip .tt-name {{ color: #fff; font-weight: 600; margin-bottom: 4px; }}
  #tooltip .tt-info {{ color: #9ca3af; }}
</style>
</head>
<body>

<div id="header">
  <div>
    <h1>知识图谱 &nbsp;·&nbsp; {img_id}</h1>
    <div id="desc">{image_desc[:200]}{"…" if len(image_desc) > 200 else ""}</div>
  </div>
  <div id="stats">
    <div class="stat-badge"><div class="val val-img">{n_in_image}</div><div class="lbl">图中实体</div></div>
    <div class="stat-badge"><div class="val val-ext">{n_external}</div><div class="lbl">外部实体</div></div>
    <div class="stat-badge"><div class="val val-edge">{len(edges)}</div><div class="lbl">关系边</div></div>
    <div class="stat-badge"><div class="val val-cross">{n_cross}</div><div class="lbl">跨实体边</div></div>
  </div>
</div>

<div id="legend">
  <div class="legend-item"><span class="dot dot-img"></span> 图中实体</div>
  <div class="legend-item"><span class="dot dot-ext"></span> 外部知识实体</div>
  <div class="legend-item"><span class="line-cross"></span> 跨实体关联边</div>
  <div class="legend-item"><span class="line-normal"></span> 普通知识边</div>
  <span style="color:#6b7280;margin-left:8px">点击节点查看详情 · 滚轮缩放 · 拖拽平移</span>
</div>

<div id="controls">
  <label>搜索节点：</label>
  <input id="search-input" type="text" placeholder="输入实体名称..." />
  <button class="ctrl-btn" onclick="resetView()">重置视图</button>
  <button class="ctrl-btn" onclick="toggleLabels()">切换标签</button>
</div>

<div id="main">
  <div id="canvas">
    <svg id="svg"></svg>
    <div id="tooltip"><div class="tt-name"></div><div class="tt-info"></div></div>
  </div>
  <div id="sidebar">
    <h3>节点详情</h3>
    <div id="sidebar-content">← 点击图中节点查看详情</div>
  </div>
</div>

<script>
const ALL_NODES = {nodes_json};
const ALL_EDGES = {edges_json};

// ── 颜色 ──
const COLOR_IN_IMAGE = "#f59e0b";
const COLOR_EXTERNAL = "#60a5fa";
const COLOR_CROSS_EDGE = "#34d399";
const COLOR_NORMAL_EDGE = "#4b5563";

let showLabels = true;

const svg = d3.select("#svg");
const width = () => document.getElementById("canvas").clientWidth;
const height = () => document.getElementById("canvas").clientHeight;

const g = svg.append("g");

// ── zoom ──
const zoom = d3.zoom().scaleExtent([0.1, 8]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// ── arrow markers ──
svg.append("defs").selectAll("marker")
  .data(["normal", "cross"])
  .join("marker")
  .attr("id", d => `arrow-${{d}}`)
  .attr("viewBox", "0 -4 8 8")
  .attr("refX", 18)
  .attr("refY", 0)
  .attr("markerWidth", 6)
  .attr("markerHeight", 6)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M0,-4L8,0L0,4")
  .attr("fill", d => d === "cross" ? COLOR_CROSS_EDGE : COLOR_NORMAL_EDGE)
  .attr("opacity", 0.7);

// ── simulation ──
const sim = d3.forceSimulation(ALL_NODES)
  .force("link", d3.forceLink(ALL_EDGES).id(d => d.id).distance(d => {{
    const bothImg = d.source.in_image && d.target.in_image;
    return bothImg ? 160 : 110;
  }}).strength(0.4))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width() / 2, height() / 2))
  .force("collide", d3.forceCollide(30));

// ── edges ──
const link = g.append("g").selectAll("line")
  .data(ALL_EDGES).join("line")
  .attr("class", "link")
  .attr("stroke", d => d.source_type === "cross_entity" ? COLOR_CROSS_EDGE : COLOR_NORMAL_EDGE)
  .attr("stroke-width", d => d.source_type === "cross_entity" ? 2 : 1)
  .attr("marker-end", d => `url(#arrow-${{d.source_type === "cross_entity" ? "cross" : "normal"}})`)
  .on("mouseover", function(event, d) {{
    showTooltip(event, `<div class='tt-name'>${{d.relation}}</div><div class='tt-info'>${{d.fact || ""}}</div>`);
  }})
  .on("mouseout", hideTooltip);

// ── edge labels ──
const linkLabel = g.append("g").selectAll("text")
  .data(ALL_EDGES).join("text")
  .attr("class", "link-label")
  .text(d => d.relation.length > 20 ? d.relation.slice(0, 18) + "…" : d.relation);

// ── nodes ──
const nodeG = g.append("g").selectAll("g")
  .data(ALL_NODES).join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag",  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end",   (e, d) => {{ if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}))
  .on("click", (event, d) => {{ event.stopPropagation(); selectNode(d); }})
  .on("mouseover", (event, d) => {{
    const info = d.in_image ? "图中实体" : "外部知识";
    showTooltip(event, `<div class='tt-name'>${{d.label}}</div><div class='tt-info'>${{info}}</div>`);
  }})
  .on("mouseout", hideTooltip);

nodeG.append("circle")
  .attr("r", d => d.in_image ? 14 : 9)
  .attr("fill", d => d.in_image ? COLOR_IN_IMAGE : COLOR_EXTERNAL)
  .attr("fill-opacity", d => d.in_image ? 0.9 : 0.7)
  .attr("stroke", d => d.in_image ? "#fbbf24" : "#3b82f6");

nodeG.append("text")
  .attr("dy", d => d.in_image ? -18 : -13)
  .attr("text-anchor", "middle")
  .attr("font-weight", d => d.in_image ? "700" : "400")
  .attr("font-size", d => d.in_image ? "12px" : "10px")
  .attr("fill", d => d.in_image ? "#fbbf24" : "#93c5fd")
  .text(d => d.label.length > 20 ? d.label.slice(0, 18) + "…" : d.label);

sim.on("tick", () => {{
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2);
  nodeG.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

// ── 点击空白取消选中 ──
svg.on("click", () => clearSelection());

// ── 选中节点 ──
function selectNode(d) {{
  nodeG.classed("highlighted", n => n.id === d.id);
  nodeG.classed("dimmed", n => n.id !== d.id);
  link.classed("dimmed", e => e.source.id !== d.id && e.target.id !== d.id);

  const relatedEdges = ALL_EDGES.filter(e => e.source.id === d.id || e.target.id === d.id);
  const relatedIds = new Set(relatedEdges.flatMap(e => [e.source.id, e.target.id]));
  nodeG.classed("dimmed", n => n.id !== d.id && !relatedIds.has(n.id));

  // 侧栏
  const edgesOut = ALL_EDGES.filter(e => e.source.id === d.id);
  const edgesIn  = ALL_EDGES.filter(e => e.target.id === d.id);

  const tag = d.in_image
    ? `<span class='detail-tag tag-in-image'>图中实体</span>`
    : `<span class='detail-tag'>外部知识</span>`;

  let html = `
    <div class='detail-section'>
      <div class='detail-label'>实体名称</div>
      <div class='detail-value' style='font-size:15px;font-weight:700;color:#fff'>${{d.label}}</div>
    </div>
    <div class='detail-section'>${{tag}}</div>`;

  if (edgesOut.length > 0) {{
    html += `<div class='detail-section'><div class='detail-label'>出边（${{edgesOut.length}}）</div>`;
    edgesOut.forEach(e => {{
      const cls = e.source_type === "cross_entity" ? "edge-item cross" : "edge-item";
      html += `<div class='${{cls}}'><span class='edge-rel'>${{e.relation}}</span> → ${{e.target.label || e.target}}<div class='edge-fact'>${{e.fact}}</div></div>`;
    }});
    html += `</div>`;
  }}

  if (edgesIn.length > 0) {{
    html += `<div class='detail-section'><div class='detail-label'>入边（${{edgesIn.length}}）</div>`;
    edgesIn.forEach(e => {{
      const cls = e.source_type === "cross_entity" ? "edge-item cross" : "edge-item";
      html += `<div class='${{cls}}'>${{e.source.label || e.source}} → <span class='edge-rel'>${{e.relation}}</span><div class='edge-fact'>${{e.fact}}</div></div>`;
    }});
    html += `</div>`;
  }}

  document.getElementById("sidebar-content").innerHTML = html;
}}

function clearSelection() {{
  nodeG.classed("highlighted dimmed", false);
  link.classed("dimmed", false);
  document.getElementById("sidebar-content").innerHTML = "← 点击图中节点查看详情";
}}

// ── 搜索 ──
document.getElementById("search-input").addEventListener("input", function() {{
  const q = this.value.trim().toLowerCase();
  if (!q) {{ clearSelection(); return; }}
  const match = ALL_NODES.find(n => n.label.toLowerCase().includes(q));
  if (match) {{
    selectNode(match);
    svg.transition().duration(600).call(
      zoom.transform,
      d3.zoomIdentity.translate(width()/2 - match.x, height()/2 - match.y).scale(1.5)
    );
  }}
}});

// ── 重置 ──
function resetView() {{
  clearSelection();
  svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity);
}}

// ── 切换标签 ──
function toggleLabels() {{
  showLabels = !showLabels;
  nodeG.selectAll("text").style("display", showLabels ? null : "none");
  linkLabel.style("display", showLabels ? null : "none");
}}

// ── Tooltip ──
const tooltip = document.getElementById("tooltip");
function showTooltip(event, html) {{
  tooltip.innerHTML = html;
  tooltip.style.opacity = 1;
  tooltip.style.left = (event.offsetX + 12) + "px";
  tooltip.style.top  = (event.offsetY + 12) + "px";
}}
function hideTooltip() {{ tooltip.style.opacity = 0; }}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="知识图谱可视化")
    parser.add_argument("entity_file", nargs="?", default="/Users/xrose3159/Desktop/agenticdata_only_vlm/output/entities/img_0010.json")
    parser.add_argument("--out", default=None, help="输出 HTML 文件路径")
    args = parser.parse_args()

    with open(args.entity_file, encoding="utf-8") as f:
        data = json.load(f)

    nodes, edges = build_graph(data)
    print(f"图中实体: {sum(1 for n in nodes if n['in_image'])}  外部实体: {sum(1 for n in nodes if not n['in_image'])}  关系边: {len(edges)}")

    out_path = args.out or f"kg_{data.get('img_id', 'output')}.html"
    html = generate_html(data, nodes, edges)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"已生成: {out_path}")
    webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
