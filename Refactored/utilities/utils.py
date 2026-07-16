import pandas as pd
import numpy as np
import umap
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def generate_interactive_3d_plot(artifacts_dir: str = "artifacts", output_file: str = "index.html"):
    """
    Loads BERT embeddings, reduces them to 3D, and generates a standalone 
    HTML file with a custom JavaScript multi-select UI.
    """
    artifacts_path = Path(artifacts_dir)
    data_path = artifacts_path / "clean_manhwa_data.parquet"
    embeddings_path = artifacts_path / "synopsis_embeddings.npy"

    logger.info("Loading artifacts...")
    try:
        df = pd.read_parquet(data_path)
        embeddings = np.load(embeddings_path)
    except FileNotFoundError:
        logger.error(f"Could not find artifacts in {artifacts_dir}. Run pipeline.py first.")
        return

    logger.info("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.05, metric='cosine', random_state=42)
    reduced_matrix = reducer.fit_transform(embeddings)
    
    df['umap_x'] = reduced_matrix[:, 0]
    df['umap_y'] = reduced_matrix[:, 1]
    df['umap_z'] = reduced_matrix[:, 2]

    # Parse unique genres
    all_genres = set()
    for g_str in df['genres'].dropna():
        for g in str(g_str).split(','):
            g_clean = g.strip()
            if g_clean:
                all_genres.add(g_clean)
    unique_genres = sorted(list(all_genres))

    # Format hover text
    hover_text = df.apply(
        lambda row: f"<b>{row['title']}</b><br>"
                    f"Genres: {row['genres']}<br><br>"
                    f"<i>{str(row['synopsis'])[:120]}...</i>", axis=1
    )

    logger.info("Generating custom HTML and JavaScript payload...")
    
    # 1. Package the data for JavaScript
    js_data = json.dumps({
        'x': df['umap_x'].tolist(),
        'y': df['umap_y'].tolist(),
        'z': df['umap_z'].tolist(),
        'genres': df['genres'].fillna("").astype(str).tolist(),
        'hover': hover_text.tolist()
    })
    
    # 2. Build the HTML options for the multi-select box
    options_html = "".join([f'<option value="{g}">{g}</option>' for g in unique_genres])

    # 3. Create the raw HTML template with embedded Plotly.js and our logic
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive Manhwa Latent Space</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ margin: 0; background-color: #111; color: white; font-family: Arial, sans-serif; overflow: hidden; }}
            #ui-container {{ 
                position: absolute; top: 20px; left: 20px; z-index: 100; 
                background: rgba(30, 30, 30, 0.85); padding: 15px; 
                border-radius: 8px; border: 1px solid #444; backdrop-filter: blur(5px);
            }}
            select {{ 
                background: #222; color: white; border: 1px solid #555; 
                padding: 8px; height: 250px; width: 220px; outline: none; border-radius: 4px;
            }}
            option {{ padding: 4px; }}
            option:checked {{ background: #ff7f50 linear-gradient(0deg, #ff7f50 0%, #ff7f50 100%); color: #111; font-weight: bold; }}
            h3 {{ margin: 0 0 10px 0; font-size: 16px; }}
            p {{ margin: 0 0 10px 0; font-size: 12px; color: #aaa; }}
        </style>
    </head>
    <body>
        <div id="ui-container">
            <h3>Filter by Genre</h3>
            <p>Hold Ctrl (Cmd on Mac) to select multiple.</p>
            <select id="genre-select" multiple>
                {options_html}
            </select>
        </div>
        
        <div id="plot-div" style="width:100vw; height:100vh;"></div>

        <script>
            // Load the data injected by Python
            const rawData = {js_data};
            
            const baseColor = "rgba(100, 149, 237, 0.4)";
            const highlightColor = "rgba(255, 127, 80, 0.95)";
            const ghostColor = "rgba(80, 80, 80, 0.05)";

            // Draw the initial plot
            const trace = {{
                x: rawData.x, y: rawData.y, z: rawData.z,
                mode: 'markers', type: 'scatter3d',
                text: rawData.hover, hoverinfo: 'text',
                marker: {{ size: 3, color: rawData.x.map(() => baseColor), line: {{width: 0}} }}
            }};
            
            const layout = {{
                margin: {{l:0, r:0, b:0, t:0}},
                paper_bgcolor: '#111',
                scene: {{ 
                    xaxis: {{showticklabels: false, title: '', showgrid: false, zeroline: false}}, 
                    yaxis: {{showticklabels: false, title: '', showgrid: false, zeroline: false}}, 
                    zaxis: {{showticklabels: false, title: '', showgrid: false, zeroline: false}} 
                }}
            }};
            
            Plotly.newPlot('plot-div', [trace], layout);

            // Listen for clicks on the multi-select box
            document.getElementById('genre-select').addEventListener('change', function(e) {{
                // Get all selected genres
                const selected = Array.from(e.target.selectedOptions).map(opt => opt.value);
                
                let colors = [];
                let sizes = [];

                if (selected.length === 0) {{
                    // Reset if nothing is selected
                    colors = rawData.x.map(() => baseColor);
                    sizes = rawData.x.map(() => 3);
                }} else {{
                    // Apply highlighting logic
                    for(let i=0; i<rawData.genres.length; i++) {{
                        const g = rawData.genres[i];
                        
                        // OR LOGIC: If the Manhwa has ANY of the selected genres, highlight it.
                        const isMatch = selected.some(sel => g.includes(sel));
                        
                        colors.push(isMatch ? highlightColor : ghostColor);
                        sizes.push(isMatch ? 5 : 2);
                    }}
                }}
                
                // Efficiently update only the colors and sizes without redrawing the whole 3D scene
                Plotly.restyle('plot-div', {{ 'marker.color': [colors], 'marker.size': [sizes] }});
            }});
        </script>
    </body>
    </html>
    """

    # 4. Save to disk
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    logger.info(f"Success! Interactive multi-select visualization saved to {output_file}")

# if __name__ == "__main__":
#     generate_interactive_3d_plot()