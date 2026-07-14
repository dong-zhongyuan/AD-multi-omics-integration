# panel_c_ortholog.py — Mouse-to-human ortholog mapping (plotly Sankey, no extra text)
import os
import pandas as pd
import plotly.graph_objects as go
import kaleido  # for static image export

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT  = os.path.join(HERE, '..', 'output')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'fig1c_ortholog_sankey.csv'))
nodes_df = pd.read_csv(os.path.join(DATA, 'fig1c_ortholog_nodes.csv'))

# Build node list (unique, in order)
node_names = list(nodes_df['node'])
color_map = dict(zip(nodes_df['node'], nodes_df['color']))
node_colors = [color_map.get(n, '#999999') for n in node_names]
node_idx = {n: i for i, n in enumerate(node_names)}

sources = [node_idx[s] for s in df['source']]
targets = [node_idx[t] for t in df['target']]
values  = list(df['value'])
# ribbon color = source node color
link_colors = [color_map.get(s, '#999999') for s in df['source']]
link_colors_rgba = []
for c in link_colors:
    link_colors_rgba.append('rgba({},{},{},0.4)'.format(
        int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)))

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=30, thickness=25,
        line=dict(color='black', width=0.8),
        label=node_names,
        color=node_colors,
    ),
    link=dict(
        source=sources, target=targets, value=values,
        color=link_colors_rgba,
    ),
    arrangement='snap',
)])

fig.update_layout(
    width=800, height=800,
    font=dict(family='Arial', size=20, color='black'),
    margin=dict(l=10, r=10, t=10, b=10),
)

for ext in ['png', 'pdf', 'svg']:
    fig.write_image(os.path.join(OUT, f'c.{ext}'), scale=2)
print('Saved c.png/pdf/svg')
