import sponet.collective_variables as cv
import dash
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sponet import load_params
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import sponet_cv_testing.datamanagement as dm


def create_figure_from_network(network: nx.Graph, x: np.ndarray):
    seed = 100
    pos = nx.spring_layout(network, seed=seed)
    # pos = nx.planar_layout(network)
    # pos = nx.kamada_kawai_layout(network)
    pos = np.array(list(pos.values()))

    node_trace = go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers", marker=dict(showscale=True), hoverinfo='none')
    node_trace.marker.color = x

    edge_x = []
    edge_y = []
    for edge in network.edges():
        x0, y0 = pos[int(edge[0])]
        x1, y1 = pos[int(edge[1])]

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig


def calc_colors(x, network):
    color_options = ["shares",
                     "weighted shares",
                     # "interfaces",
                     # "cluster",
                     # "custom"
                     ]

    counts = cv.OpinionShares(3, normalize=True, weights=None)(x)
    colors = {"shares": counts[:, 0]}

    degree_sequence = np.array([d for n, d in network.degree()])
    counts = cv.OpinionShares(3, normalize=True, weights=degree_sequence)(x)
    colors["weighted shares"] = counts[:, 0]

    # counts = cv.Interfaces(network, True)(x)
    # colors["interfaces"] = counts[:, 0]

    # weights = calc_weights_biggest_cluster(network)
    # counts = cv.OpinionShares(2, True, weights)(x)
    # colors["cluster"] = counts[:, 0]

    return color_options, colors


def create_app(xi: np.ndarray, x_anchor: np.ndarray, network: nx.Graph):
    app = Dash(__name__)

    color_options, colors = calc_colors(x_anchor, network)

    network_fig_a = create_figure_from_network(network, x_anchor[np.random.randint(0, x_anchor.shape[0])])
    network_fig_b = create_figure_from_network(network, x_anchor[np.random.randint(0, x_anchor.shape[0])])
    network_fig_c = create_figure_from_network(network, x_anchor[np.random.randint(0, x_anchor.shape[0])])

    xixi_fig = px.scatter(x=xi[:, 0], y=xi[:, 1],
                          color=colors[color_options[0]],
                          labels={"x": r"$\xi_1$", "y": r"$\xi_2$", "color": "c"}
                          )

    app.layout = html.Div(children=[
        html.Div(children=[
            html.Div([
                html.Div([
                    html.Label("x-axis:"),
                    dcc.Dropdown([f"{i}" for i in range(1, xi.shape[1] + 1)], "1", id="dropdown-x-axis",
                                 style={'width': '10vw'}),
                ]),
                html.Div([
                    html.Label("y-axis:"),
                    dcc.Dropdown([f"{i}" for i in range(1, xi.shape[1] + 1)], "2", id="dropdown-y-axis",
                                 style={'width': '10vw'}),
                ]),
                html.Div([
                    html.Label("color:"),
                    dcc.Dropdown(color_options, color_options[0], id="dropdown-color",
                                 style={'width': '10vw'}),
                ]),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            dcc.Graph(
                id='cv-graph', figure=xixi_fig, mathjax=True, style={'width': '50vw', 'height': '45vh'}
            ),

            dcc.Graph(
                id='state-graph-a', figure=network_fig_a, mathjax=True, style={'width': '50vw', 'height': '45vh'}
            )
        ]),
        html.Div(children=[
            html.Label('Update network'),
            dcc.RadioItems(['a)', 'b)', 'c)'], 'a)', id="radio-items"),

            dcc.Graph(
                id='state-graph-b', figure=network_fig_b, mathjax=True, style={'width': '50vw', 'height': '45vh'}
            ),

            dcc.Graph(
                id='state-graph-c', figure=network_fig_c, mathjax=True, style={'width': '50vw', 'height': '45vh'}
            )
        ])
    ], style={'display': 'flex', 'flex-direction': 'row'})

    @app.callback(
        Output('cv-graph', 'figure'),
        Input('dropdown-x-axis', 'value'),
        Input('dropdown-y-axis', 'value'),
        Input('dropdown-color', 'value')
    )
    def update_xixi_plot(dropdown_x, dropdown_y, dropdown_color):
        return px.scatter(x=xi[:, int(dropdown_x) - 1], y=xi[:, int(dropdown_y) - 1],
                          color=colors[dropdown_color],
                          labels={"x": rf"$\xi_{dropdown_x}$", "y": rf"$\xi_{dropdown_y}$", "color": "c"}
                          )

    @app.callback(
        Output('state-graph-a', 'figure'),
        Output('state-graph-b', 'figure'),
        Output('state-graph-c', 'figure'),
        Input('cv-graph', 'clickData'),
        Input('radio-items', 'value'),
        Input('dropdown-x-axis', 'value'),
        Input('dropdown-y-axis', 'value')
    )
    def update_network_on_click(click_data, radio_item, dropdown_x, dropdown_y):
        if click_data is None:
            raise dash.exceptions.PreventUpdate
        if dash.callback_context.triggered_id != "cv-graph":
            raise dash.exceptions.PreventUpdate

        xi1, xi2 = click_data["points"][0]["x"], click_data["points"][0]["y"]
        dist = np.sum((xi[:, [int(dropdown_x) - 1, int(dropdown_y) - 1]] - np.array([xi1, xi2])) ** 2, axis=1)
        idx = np.argmin(dist)

        if radio_item == "a)":
            network_fig_a.data[1].marker.color = x_anchor[idx]
            network_fig_a.update_layout(
                title=rf"$a):\quad \xi_{dropdown_x}, \xi_{dropdown_y}={np.round(xi1, 4)}, {np.round(xi2, 4)}$")
        elif radio_item == "b)":
            network_fig_b.data[1].marker.color = x_anchor[idx]
            network_fig_b.update_layout(
                title=rf"$b):\quad \xi_{dropdown_x}, \xi_{dropdown_y}={np.round(xi1, 4)}, {np.round(xi2, 4)}$")
        else:
            network_fig_c.data[1].marker.color = x_anchor[idx]
            network_fig_c.update_layout(
                title=rf"$c):\quad \xi_{dropdown_x}, \xi_{dropdown_y}={np.round(xi1, 4)}, {np.round(xi2, 4)}$")

        return network_fig_a, network_fig_b, network_fig_c

    return app


def main():
    file_path = "../data/results/CNVM2_ab2_n500_r098-100_rt001-002_l400_a1000_s150/"
    xi = np.load(file_path + "transition_manifold.npy")
    x_anchor = np.load(file_path + "x_data.npz")["x_anchor"]
    #params = load_params(file_path + "params.pkl")
    network = dm.open_network(file_path, "network")

    appl = create_app(xi, x_anchor, network)
    appl.run(debug=True)


if __name__ == '__main__':
    main()
