"""
knowledge_graph.py  —  Interactive D3 Knowledge Graph
Place in app/ alongside app.py.
"""

import json
import re
import math
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

CHAPTER_COLOURS = [
    "#4F46E5", "#0891B2", "#059669", "#D97706",
    "#DC2626", "#7C3AED", "#DB2777", "#065F46",
    "#1D4ED8", "#B45309", "#6D28D9", "#0F766E",
    "#BE185D", "#1E40AF", "#15803D", "#9F1239",
]

def _parse_chapter_num(section_path: str):
    m = re.match(r'^(\d+)', section_path.strip())
    return int(m.group(1)) if m else 0

def _build_graph_data(chunks_path: Path) -> dict:
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    section_counts = data["metadata"]["section_chunk_counts"]

    chapters = {}
    sections = {}

    # 1. Dynamically populate data (skips missing chapters like Ch 13 automatically)
    for section_path, chunk_count in section_counts.items():
        ch_num = _parse_chapter_num(section_path)
        if ch_num == 0:
            continue

        parts     = [p.strip() for p in section_path.split(">")]
        ch_label  = parts[0]
        sec_label = parts[-1] if len(parts) > 1 else parts[0]

        colour = CHAPTER_COLOURS[(ch_num - 1) % len(CHAPTER_COLOURS)]

        if ch_label not in chapters:
            chapters[ch_label] = {
                "num": ch_num, 
                "label": ch_label,
                "colour": colour, 
                "sections": [],
                "total_chunks": 0
            }
        
        chapters[ch_label]["sections"].append(section_path)
        chapters[ch_label]["total_chunks"] += chunk_count

        sections[section_path] = {
            "label":       sec_label,
            "ch_label":    ch_label,
            "ch_num":      ch_num,
            "colour":      colour,
            "chunk_count": chunk_count,
            "est_time":    chunk_count * 2,
            "full_path":   section_path,
        }

    nodes = []
    links = []
    node_id_map = {}

    # ── Chapter nodes (Dynamic Area Scaling) ──────────────────────────────────
    for ch_label, ch_data in sorted(chapters.items(), key=lambda x: x[1]["num"]):
        nid = f"ch_{ch_data['num']}"
        node_id_map[ch_label] = nid
        short = ch_label.split(" ", 1)[1] if " " in ch_label else ch_label
        
        # Determine size: Proportional to square root of chunks
        r_calc = 22 + (math.sqrt(ch_data["total_chunks"]) * 2.5) if ch_data["total_chunks"] > 0 else 22
        
        nodes.append({
            "id":           nid,
            "label":        ch_label,
            "short":        short,
            "type":         "chapter",
            "colour":       ch_data["colour"],
            "num":          ch_data["num"],
            "radius":       max(22, min(48, int(r_calc))), 
            "total_chunks": ch_data["total_chunks"],
            "sec_count":    len(ch_data["sections"]),
            "query":        f"Give me an overview of {ch_label}",
            "snippet":      f"Core foundational chapter. Contains {len(ch_data['sections'])} major sections and {ch_data['total_chunks']} chunks of conceptual data.",
        })

    # ── Section nodes (Dynamic Area Scaling) ──────────────────────────────────
    for sec_path, sec_data in sections.items():
        nid = "sec_" + re.sub(r'[^a-z0-9]', '_', sec_path.lower())[:45]
        node_id_map[sec_path] = nid
        
        # Determine size: Proportional to square root of chunks
        r_calc = 9 + (math.sqrt(sec_data["chunk_count"]) * 3.5)
        
        nodes.append({
            "id":          nid,
            "label":       sec_data["label"],
            "type":        "section",
            "colour":      sec_data["colour"],
            "ch_num":      sec_data["ch_num"],
            "ch_label":    sec_data["ch_label"],
            "chunk_count": sec_data["chunk_count"],
            "est_time":    sec_data["est_time"],
            "radius":      max(9, min(26, int(r_calc))), 
            "query":       f"Explain {sec_data['label']} from Chapter {sec_data['ch_num']}",
            "full_path":   sec_data["full_path"],
            "snippet":     f"Detailed breakdown of {sec_data['label']}. Focuses on specific mechanisms, definitions, and applications.",
        })

    # ── Chapter → Section links ───────────────────────────────────────────────
    for sec_path, sec_data in sections.items():
        ch_label = sec_data["ch_label"]
        if ch_label in node_id_map and sec_path in node_id_map:
            links.append({
                "source":   node_id_map[ch_label],
                "target":   node_id_map[sec_path],
                "colour":   sec_data["colour"],
                "strength": 0.8,
                "distance": 95,
                "type":     "ch_sec",
            })

    # ── Chapter → Chapter spine links (Connects only existing chapters in order)
    sorted_chapters = sorted([d for d in nodes if d["type"] == "chapter"], key=lambda x: x["num"])
    ch_ids = [n["id"] for n in sorted_chapters]
    
    for i in range(len(ch_ids)):
        links.append({
            "source":   ch_ids[i],
            "target":   ch_ids[(i + 1) % len(ch_ids)],
            "colour":   "#CBD5E1", # Light mode spine color
            "strength": 0.05,
            "distance": 260,
            "type":     "spine",
        })

    return {"nodes": nodes, "links": links}

def show_knowledge_graph_page(chunks_path: Path):
    # ── Mobile-Optimized CSS ──────────────────────────────────────────────────
    st.markdown("""
    <style>
        /* Force Streamlit columns to stack cleanly on mobile */
        @media (max-width: 768px) {
            [data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                padding: 0 !important;
                margin-bottom: 12px;
            }
        }

        /* Responsive Stats Grid */
        .kg-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        @media (max-width: 840px) {
            .kg-stat-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 580px) {
            .kg-stat-grid { grid-template-columns: 1fr; }
        }

        .kg-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 24px; /* Accurate and equal padding */
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
            transition: transform 0.2s;
        }
        .kg-card:hover { transform: translateY(-2px); border-color: #CBD5E1; }

        /* Button Styling for Quick Jump */
        .stButton > button {
            border-radius: 12px !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            border: 1px solid #E2E8F0 !important;
            transition: all 0.2s ease !important;
            background: white !important;
            font-size: 0.82rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.5rem 0 1.2rem;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:0.5rem;">
            <div style="width:48px;height:48px;border-radius:12px;flex-shrink:0;
                        background:linear-gradient(135deg, #4F46E5 0%, #312E81 100%);color:white;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 8px 16px rgba(79,70,229,0.25); border: 1px solid rgba(255,255,255,0.1);">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                     stroke-width="1.8" stroke="currentColor" style="width:26px;height:26px;">
                    <path stroke-linecap="round" stroke-linejoin="round"
                          d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987
                             8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0
                             016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018
                             18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"/>
                </svg>
            </div>
            <div>
                <h2 style="margin:0;font-size:1.75rem;font-weight:800;
                           color:#0F172A;letter-spacing:-0.03em;">
                    Knowledge Graph
                </h2>
                <p style="margin:4px 0 0;font-size:0.9rem;color:#64748B; font-weight:500;">
                    Node size indicates volume of material · Hover to explore semantic networks
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not chunks_path.exists():
        st.error(f"Chunks file not found: {chunks_path}")
        return

    with st.spinner("Rendering interactive topology…"):
        graph = _build_graph_data(chunks_path)

    n_sections = len([n for n in graph["nodes"] if n["type"] == "section"])
    n_links    = len([l for l in graph["links"] if l["type"] == "ch_sec"])

    # ── Optimized Stats Grid ─────────────────
    st.markdown(f"""
    <div class="kg-stat-grid">
        <div class="kg-card">
            <div style="font-size:1.8rem;font-weight:800;color:#4F46E5;line-height:1;">16</div>
            <div style="font-size:0.7rem;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-top:8px;">Total Chapters</div>
        </div>
        <div class="kg-card">
            <div style="font-size:1.8rem;font-weight:800;color:#059669;line-height:1;">{n_sections}</div>
            <div style="font-size:0.7rem;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-top:8px;">Sections Mapped</div>
        </div>
        <div class="kg-card">
            <div style="font-size:1.8rem;font-weight:800;color:#D97706;line-height:1;">{n_links}</div>
            <div style="font-size:0.7rem;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-top:8px;">Connections</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Legend ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:16px;align-items:center;
                font-size:0.72rem;color:#64748B;margin-bottom:14px;
                flex-wrap:wrap; font-weight: 500; background: #F8FAFC; 
                padding: 10px 14px; border-radius: 10px; border: 1px dashed #E2E8F0;">
        <span style="display:flex;align-items:center;gap:6px;">
            <svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#4F46E5" stroke="#fff" stroke-width="1.2"/></svg>
            Chapter Node
        </span>
        <span style="display:flex;align-items:center;gap:6px;">
            <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="#94A3B8"/></svg>
            Section Node
        </span>
        <span style="display:flex;align-items:center;gap:6px;">
            <svg width="16" height="4"><line x1="0" y1="2" x2="16" y2="2" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="3,2"/></svg>
            Semantic Spine
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── D3 graph ──────────────────────────────────────────────────────────────
    graph_json = json.dumps(graph)

    html_code = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*{{box-sizing:border-box;margin:0;padding:0;}}
body{{ background: transparent; font-family: 'Inter', system-ui, sans-serif; overflow: hidden; }}

#gc{{
    width: 100%; height: 600px; position: relative;
    border: 1px solid #E2E8F0; border-radius: 16px;
    background-color: #FDFDFF;
    background-image: radial-gradient(#E2E8F0 1px, transparent 1px);
    background-size: 24px 24px; overflow: hidden;
}}

svg{{width:100%;height:100%;cursor:grab;}}
svg:active{{cursor:grabbing;}}

/* Links */
.link-spine{{fill:none; stroke:#CBD5E1; stroke-width:1.5; stroke-dasharray:6,6; opacity:0.6;}}
.link-ch-sec{{fill:none; stroke-width:1.2; opacity:0.35;}}

.dimmed {{ opacity: 0.1 !important; filter: grayscale(100%); transition: all 0.3s ease; }}
.highlighted {{ opacity: 1 !important; stroke-width: 2.5px !important; transition: all 0.3s ease; }}

/* Nodes */
.ch-node circle{{ stroke-width: 2.5; stroke: #fff; cursor: pointer; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.08)); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }}
.ch-node:hover circle{{ stroke-width: 4; transform: scale(1.1); filter: drop-shadow(0 8px 20px rgba(0,0,0,0.12)); }}
.ch-node text{{ font-size: 13px; font-weight: 800; fill: white; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}

.sec-node circle{{ stroke-width: 1.2; stroke: rgba(0,0,0,0.05); cursor: pointer; transition: all 0.2s ease; }}
.sec-node:hover circle{{ stroke-width: 2.5; stroke: #fff; transform: scale(1.2); }}

/* Tooltip (Fixed for Mobile) */
#tt {{
    position: absolute; background: rgba(255, 255, 255, 0.98); backdrop-filter: blur(8px); border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 16px; color: #0F172A; pointer-events: none; 
    opacity: 0; transform: translateY(10px); transition: all 0.2s ease; 
    width: 260px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); z-index: 100;
}}
#tt.visible {{ opacity: 1; transform: translateY(0); }}

@media (max-width: 600px) {{
    #tt {{
        position: fixed; bottom: 20px; left: 20px !important; right: 20px !important;
        top: auto !important; width: auto; max-width: none;
    }}
}}

.tt-header {{ font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #64748B; margin-bottom: 6px; }}
.tt-title {{ font-weight: 700; font-size: 15px; margin-bottom: 12px; line-height: 1.3; color: #0F172A; }}
.tt-tags {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.tt-tag {{ 
    background: #F1F5F9; border: 1px solid #E2E8F0;
    padding: 4px 8px; border-radius: 6px; font-size: 10px; font-weight: 600; display: flex; align-items: center; gap: 4px; color: #475569;
}}
.tt-desc {{
    font-size: 11.5px; color: #64748B; line-height: 1.4; margin-top: 12px; 
    padding-top: 12px; border-top: 1px solid #F1F5F9;
}}

/* Controls */
#ctrls{{position:absolute;top:20px;right:20px;display:flex;flex-direction:column;gap:8px;}}
.cb{{
    background: #fff; border: 1px solid #E2E8F0;
    border-radius: 10px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 20px; color: #64748B; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.2s;
}}
.cb:hover{{ background: #F8FAFC; color: #4F46E5; border-color: #4F46E5; }}

</style>
</head>
<body>
<div id="gc">
  <svg id="svg"><defs id="svg-defs"></defs></svg>
  <div id="tt">
    <div class="tt-header" id="tt-h"></div>
    <div class="tt-title" id="tt-t"></div>
    <div class="tt-tags" id="tt-tags"></div>
    <div class="tt-desc" id="tt-desc"></div>
  </div>
  <div id="ctrls">
    <div class="cb" onclick="zi()" title="Zoom in">＋</div>
    <div class="cb" onclick="zo()" title="Zoom out">－</div>
    <div class="cb" onclick="zr()" title="Reset">⟲</div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const G = {graph_json};
const gc = document.getElementById('gc');
const W  = gc.clientWidth || 960;
const H  = gc.clientHeight || 600;

const svg = d3.select('#svg');
const defs = d3.select('#svg-defs');
const g = svg.append('g');

const zoom = d3.zoom().scaleExtent([0.15, 5]).on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);
function zi(){{svg.transition().duration(400).call(zoom.scaleBy, 1.4);}}
function zo(){{svg.transition().duration(400).call(zoom.scaleBy, 0.7);}}
function zr(){{svg.transition().duration(600).ease(d3.easeCubicOut).call(zoom.transform, d3.zoomIdentity.translate(W/2,H/2).scale(0.8));}}

const linkedByIndex = {{}};
G.links.forEach(d => {{
    linkedByIndex[`${{d.source}},${{d.target}}`] = true;
    linkedByIndex[`${{d.target}},${{d.source}}`] = true;
}});
function isConnected(a, b) {{ return linkedByIndex[`${{a.id}},${{b.id}}`] || a.id === b.id; }}

const spineLinks = G.links.filter(l => l.type === 'spine');
const secLinks   = G.links.filter(l => l.type === 'ch_sec');

const sim = d3.forceSimulation(G.nodes)
    .force('link', d3.forceLink(G.links).id(d=>d.id).distance(l => l.distance || 110).strength(l => l.strength || 0.6))
    .force('charge', d3.forceManyBody().strength(d => d.type === 'chapter' ? -800 : -150))
    .force('center', d3.forceCenter(0, 0).strength(0.05))
    .force('collide', d3.forceCollide().radius(d => d.radius + 15))
    .force('radial', d3.forceRadial(d => d.type === 'chapter' ? 240 : 0, 0, 0).strength(d => d.type === 'chapter' ? 0.8 : 0))
    .alphaDecay(0.02);

const spineLink = g.append('g').selectAll('path').data(spineLinks).join('path').attr('class','link-spine');
const secLink = g.append('g').selectAll('path').data(secLinks).join('path').attr('class','link-ch-sec').attr('stroke', d => d.colour);

const node = g.append('g').selectAll('g').data(G.nodes).join('g')
    .attr('class', d => d.type==='chapter' ? 'ch-node' : 'sec-node')
    .call(d3.drag()
        .on('start',(e,d)=>{{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
        .on('drag', (e,d)=>{{ d.fx=e.x; d.fy=e.y; }})
        .on('end',  (e,d)=>{{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}))
    .on('click', onClick)
    .on('mouseover', onOver)
    .on('mouseout',  onOut);

node.append('circle').attr('r', d => d.radius).attr('fill', d => d.colour);
node.filter(d => d.type==='chapter').append('text').text(d => d.num);

const tt = document.getElementById('tt');
const ttH = document.getElementById('tt-h');
const ttT = document.getElementById('tt-t');
const ttTags = document.getElementById('tt-tags');
const ttDesc = document.getElementById('tt-desc');

function onOver(event, d) {{
    node.classed('dimmed', o => !isConnected(d, o));
    secLink.classed('dimmed', o => o.source.id !== d.id && o.target.id !== d.id);
    secLink.classed('highlighted', o => o.source.id === d.id || o.target.id === d.id);
    spineLink.classed('dimmed', true);

    ttH.textContent = d.type === 'chapter' ? `${{d.num}} MODULE` : d.ch_label;
    ttT.textContent = d.label;
    
    let tagsHTML = "";
    if (d.type === 'chapter') {{
        tagsHTML += `<div class="tt-tag" style="color:${{d.colour}};">📚 ${{d.sec_count}} Sections</div>`;
        tagsHTML += `<div class="tt-tag">🧩 ${{d.total_chunks}} Chunks</div>`;
    }} else {{
        tagsHTML += `<div class="tt-tag">⏱️ ${{d.est_time}}m </div>`;
        tagsHTML += `<div class="tt-tag">🧩 ${{d.chunk_count}} Chunks</div>`;
    }}
    ttTags.innerHTML = tagsHTML;
    ttDesc.textContent = d.snippet;

    tt.classList.add('visible');
    moveTT(event);
}}

function onOut() {{
    node.classed('dimmed', false);
    secLink.classed('dimmed', false).classed('highlighted', false);
    spineLink.classed('dimmed', false);
    tt.classList.remove('visible');
}}

gc.addEventListener('mousemove', e => {{ if(tt.classList.contains('visible')) moveTT(e); }});

function moveTT(e) {{
    const r = gc.getBoundingClientRect();
    if (window.innerWidth < 600) return; // Keep fixed bottom on mobile
    let x = e.clientX - r.left + 24;
    let y = e.clientY - r.top + 24;
    if (x + 280 > W) x = e.clientX - r.left - 280;
    if (y + 180 > H) y = e.clientY - r.top - 180;
    tt.style.left = x + 'px'; tt.style.top = y + 'px';
}}

function onClick(event, d) {{
    event.stopPropagation();
    window.parent.postMessage({{type:'streamlit:setComponentValue',value:d.query}}, '*');
}}

sim.on('tick', () => {{
    const linkPath = (d) => {{
        const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
        const dr = Math.sqrt(dx * dx + dy * dy) * 1.5; 
        return `M${{d.source.x}},${{d.source.y}}A${{dr}},${{dr}} 0 0,1 ${{d.target.x}},${{d.target.y}}`;
    }};
    spineLink.attr('d', linkPath);
    secLink.attr('d', linkPath);
    node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

svg.call(zoom.transform, d3.zoomIdentity.translate(W/2,H/2).scale(0.8));
</script>
</body>
</html>"""

    components.html(html_code, height=620, scrolling=False)

    # ── Quick Explore Grid ────────────────────────────────────────────────────
    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="width:4px; height:16px; background:#4F46E5; border-radius:4px;"></div>
        <div style="font-size:0.85rem;font-weight:700;color:#0F172A;
                    text-transform:uppercase;letter-spacing:0.08em;">
            Quick Jump Directory
        </div>
    </div>
    """, unsafe_allow_html=True)

    chapters_sorted = sorted([n for n in graph["nodes"] if n["type"] == "chapter"], key=lambda x: x["num"])

    # Responsive Grid for buttons via Streamlit columns (CSS handles stacking)
    cols_per_row = 4
    for i in range(0, len(chapters_sorted), cols_per_row):
        row  = chapters_sorted[i:i + cols_per_row]
        cols = st.columns(cols_per_row, gap="small")
        for col, ch in zip(cols, row):
            with col:
                # Custom block wrapper styling
                short = ch.get("short", ch["label"])
                short = short[:25] + "…" if len(short) > 25 else short
                
                # Visual accent
                st.markdown(f'<div style="height:3px; width:100%; background:{ch["colour"]}; border-radius:3px 3px 0 0; margin-bottom:-8px; position:relative; z-index:1;"></div>', unsafe_allow_html=True)
                
                if st.button(f"Ch. {ch['num']} : {short}", key=f"kg_ch_{ch['num']}", use_container_width=True):
                    st.session_state["page"] = "chat"
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    st.session_state.messages.append({"role": "user", "content": ch["query"]})
                    st.session_state["_pending_query"] = ch["query"]
                    st.rerun()
        
        # Balance empty columns
        if len(row) < cols_per_row:
            for _ in range(cols_per_row - len(row)):
                cols[len(row) + _].empty()