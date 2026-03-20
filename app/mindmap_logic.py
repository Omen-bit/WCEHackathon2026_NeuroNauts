import streamlit as st
import json
import streamlit.components.v1 as components
from pymilvus import connections, Collection
import os

def fetch_hierarchy(collection_name="psychology2e_chunks"):
    """
    Fetch all unique section paths from Milvus to build the tree.
    """
    try:
        milvus_host = os.environ.get("MILVUS_HOST", "localhost")
        milvus_port = os.environ.get("MILVUS_PORT", "19530")
        try:
            connections.connect(host=milvus_host, port=milvus_port)
        except:
            pass # Already connected
            
        col = Collection(collection_name)
        
        # Verify section_path field exists
        available_fields = [f.name for f in col.schema.fields]
        if "section_path" not in available_fields:
            st.error(f"Field 'section_path' not found in Milvus collection. Available: {available_fields}")
            return []

        res = col.query(expr="chunk_id >= 0", output_fields=["section_path"], limit=10000)
        
        if not res:
            return []
            
        paths = sorted(list(set(r.get("section_path") for r in res if r.get("section_path"))))
        return paths
    except Exception as e:
        st.error(f"Error fetching hierarchy from Milvus: {e}")
        return []

def build_tree(paths):
    """
    Convert a list of 'Chap > Sect > Intro' strings into a nested dictionary.
    """
    tree = {}
    for path in paths:
        if not path: continue
        parts = [p.strip() for p in path.split(">")]
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return tree

def tree_to_markdown(tree, level=0):
    """
    Convert the nested dictionary into a Markdown list format suitable for Markmap.
    """
    lines = []
    for key, children in tree.items():
        indent = "#" * (level + 1)
        lines.append(f"{indent} {key}")
        lines.extend(tree_to_markdown(children, level + 1))
    return lines

def render_mindmap(paths, title="Psychology Mind Map"):
    """
    Main entry point: fetches, builds, and renders the Markmap component.
    """
    if not paths:
        st.warning("No sections found in the database. Please run the ingestion pipeline first.")
        return

    tree = build_tree(paths)
    md_lines = tree_to_markdown(tree)
    # The autoloader expects the markdown inside a <div class="markmap">
    # We escape anything that might break the HTML
    markdown_content = f"# {title}\n" + "\n".join(md_lines)
    
    # Premium Interactive Visualization using Markmap Autoloader
    # This is much more robust than the manual transformer initialization.
    html_content = f"""
    <div id="markmap-wrapper" style="height: 700px; width: 100%; border-radius: 16px; border: 1px solid #e5e7eb; overflow: hidden; background: #ffffff; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
        <div class="markmap" style="height: 100%; width: 100%;">
            <script type="text/template">
                {markdown_content}
            </script>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader@0.17.0"></script>
    
    <style>
    body {{ margin: 0; padding: 0; }}
    .markmap svg {{
        width: 100%;
        height: 100%;
        cursor: grab;
    }}
    .markmap svg:active {{
        cursor: grabbing;
    }}
    </style>
    """
    
    components.html(html_content, height=720)
