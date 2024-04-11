from dash import Dash, html, dash_table, Output, Input, callback, dcc, Patch, ctx
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import colorlover
import pandas as pd
import sponet_cv_testing.datamanagement as dm
import numpy as np
import sponet.collective_variables as cv
import networkx as nx

app = Dash(__name__)

# Adjust path in following function call if necessary
results_path: str = "../data/results/"


df = dm.read_data_csv(f"{results_path}results_table.csv")
# Pre-filtering of the data can be done here
df = df[df["dim_estimate"] >= 1]


def create_figure_from_network(network: nx.Graph, x: np.ndarray):
    seed = 100
    pos = nx.spring_layout(network, seed=seed)
    # pos = nx.planar_layout(network)
    # pos = nx.kamada_kawai_layout(network)
    pos = np.array(list(pos.values()))

    degrees = np.array(network.degree)[:, 1:]

    node_trace = go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode="markers",
        marker=dict(showscale=True),
        customdata=degrees,
        hovertemplate="Degree: %{customdata}"
        )
    node_trace.marker.color = np.logical_not(x.astype(bool)).astype(int)

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


def discrete_background_color_bins(df: pd.DataFrame, n_bins: int=5, columns: str | list[str]='all'):
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))


def calc_colors(x, network):
    color_options = ["shares",
                     "weighted_shares",
                     # "interfaces",
                     # "cluster",
                     # "custom"
                     ]

    counts = cv.OpinionShares(3, normalize=True, weights=None)(x)
    colors = {"shares": counts[:, 0]}

    degree_sequence = np.array([d for n, d in network.degree()])
    counts = cv.OpinionShares(3, normalize=True, weights=degree_sequence)(x)
    colors["weighted_shares"] = counts[:, 0]

    # counts = cv.Interfaces(network, True)(x)
    # colors["interfaces"] = counts[:, 0]

    # weights = calc_weights_biggest_cluster(network)
    # counts = cv.OpinionShares(2, True, weights)(x)
    # colors["cluster"] = counts[:, 0]

    return color_options, colors


def create_table(data: pd.DataFrame, table_id: str) -> dash_table.DataTable:

    style_data_conditional = [
        {
            'if': {
                'row_index': 'odd'
            },
         'backgroundColor': 'rgb(220, 220, 220)'
        },
        {
            "if": {
                "filter_query": "{finished} contains 'true'",
                "column_id": "finished"
            },
            "backgroundColor": "GREEN"
        },
        {
            "if": {
                "filter_query": "{finished} contains 'false'",
                "column_id": "finished"
            },
            "backgroundColor": "RED"
        }
    ]
    colorscale_styles_r, _ = discrete_background_color_bins(df, columns=["r_ab", "r_ba"])
    colorscale_styles_rt, _ = discrete_background_color_bins(df, columns=["rt_ab", "rt_ba"])
    style_data_conditional.extend(colorscale_styles_r)
    style_data_conditional.extend(colorscale_styles_rt)

    columns_format = [
        dict(id="run_id", name="run_id", selectable=True, type="text"),
        dict(id="r_ab", name="r_ab", selectable=True, type="numeric",
             format=Format(precision=2, scheme=Scheme.fixed)),
        dict(id="r_ba", name="r_ba", selectable=True, type="numeric",
             format=Format(precision=2, scheme=Scheme.fixed)),
        dict(id="rt_ab", name="rt_ab", selectable=True, type="numeric",
             format=Format(precision=2, scheme=Scheme.fixed)),
        dict(id="rt_ba", name="rt_ba", selectable=True, type="numeric",
             format=Format(precision=2, scheme=Scheme.fixed)),
        dict(id="lag_time", name="lag_time", selectable=True, type="numeric",
             format=Format(precision=1, scheme=Scheme.fixed)),
        dict(id="dim_estimate", name="dim_estimate", selectable=True, type="numeric",
             format=Format(precision=3, scheme=Scheme.fixed)),
        dict(id="finished", name="finished", selectable=True, type="text")]

    show_df = data.reset_index()
    table = dash_table.DataTable(
        id=table_id,
        data=show_df.to_dict('records'),
        columns=columns_format,
        page_size=25,
        page_action="none",

        style_table={"height": "700px", "overflowY": "auto"},
        style_cell={"textAlign": "left"},
        fixed_rows={"headers": True},

        row_selectable="multi",
        selected_rows=[],

        column_selectable="multi",
        selected_columns=[],

        filter_action="native",

        sort_action="custom",
        sort_mode="multi",
        sort_by=[],

        style_data_conditional=style_data_conditional,
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold'
        }

    )
    return table


@callback(
    Output("data_table", "data"),
    Input("data_table", "sort_by")
)
def sort_table_cb(sort_by):
    show_df = df.reset_index()
    if len(sort_by):
        sdata = show_df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[col['direction'] == 'asc' for col in sort_by],
            inplace=False)
    else:
        sdata = show_df

    return sdata.to_dict("records")


@callback(
    Output("data_table", "selected_rows"),
    Input("unselect_all", "n_clicks")
)
def unselect_table_entries(clicks) -> list[int]:
    return []


def create_tabs(tabs_id: str) -> dcc.Tabs:
    tabs = dcc.Tabs(id=tabs_id, value="tab-1", children=[
        dcc.Tab(label="Overview Plots",
                value="tab-1",
                children=html.Div(id="overview_plot")),
        dcc.Tab(label="Coordinate Plot",
                value="tab_2",
                children=[
                    html.Div([
                        html.Div([
                            html.Label("Runs:"),
                            dcc.Dropdown(id="coordinates_plot_dropdown_runs",
                                         placeholder="Select entries in the table above by clicking on the boxes on the left side.",
                                         style={"width": "45vw"}
                            ),
                            dcc.Clipboard(
                                target_id="coordinates_plot_dropdown_runs",
                                title="Copy run id",
                                style={"width": "5vw"}
                            )
                        ]),
                        html.Div([
                            html.Label("x-axis:"),
                            dcc.Dropdown(id="coordinates_plot_dropdown_x",
                                         value="1",
                                         style={'width': '10vw'}),
                        ]),
                        html.Div([
                            html.Label("y-axis:"),
                            dcc.Dropdown(id="coordinates_plot_dropdown_y",
                                         value="2",
                                         style={'width': '10vw'}),
                        ]),
                        html.Div([
                            html.Label("z-axis:"),
                            dcc.Dropdown(id="coordinates_plot_dropdown_z",
                                         value="3",
                                         style={'width': '10vw'}),
                        ]),
                        html.Div([
                            html.Label("color:"),
                            dcc.Dropdown(id="coordinates_plot_dropdown_color",
                                         options=["shares", "weighted_shares"],
                                         value="weighted_shares",
                                         style={'width': '10vw'})
                        ])
                    ],
                        style={'display': 'flex', 'flex-direction': 'row'}
                    ),

                    html.Div([
                        dcc.Graph(
                            id="3d_coordinates_plot",
                            mathjax=True,
                            style={'width': '80vw', 'height': '80vh'}
                        ),
                        dcc.Graph(
                            id="network_plot",
                            mathjax=True,
                            style={'width': '80vw', 'height': '80vh'}
                        )
                    ], style={'display': 'flex', 'flex-direction': 'row'}),

                    html.Div(
                        children=[
                            html.H4("Logs"),
                            html.Pre("Krass hier steht text",
                                     id="runlog")
                            ],
                        #id="runlog"
                    )
                ]),

    ])
    return tabs


@callback(Output("coordinates_plot_dropdown_runs", "options"),
          Input("data_table", "data"),
          Input("data_table", "selected_rows"))
def update_run_dropdown(data, selected_rows: list[int]):
    if not selected_rows:
        return []
    return [data[i]["run_id"] for i in selected_rows]


@callback(
    Output("runlog", "children"),
    Input("coordinates_plot_dropdown_runs", "value")
)
def update_logs(selected_run: str) -> str:
    if selected_run is None:
        return ""
    file_path = f"{results_path}{selected_run}/"
    with open(file_path+"runlog.log", "r") as logfile:
        logs = logfile.readlines()
    return "".join(logs)


@callback(
    Output("coordinates_plot_dropdown_x", "options"),
    Output("coordinates_plot_dropdown_y", "options"),
    Output("coordinates_plot_dropdown_z", "options"),
    Input("coordinates_plot_dropdown_runs", "value")
)
def update_coord_plot_coord_dd_cb(run_id: str) -> list[list[str]]:
    if run_id is None:
        return [[""], [""], [""]]
    run = df.loc[run_id]
    options = [f"{i}" for i in range(1, run["cv_dim"] + 1)]
    return [options, options, options]


@callback(
    Output("3d_coordinates_plot", "figure"),
    Input("coordinates_plot_dropdown_runs", "value"),
    Input("coordinates_plot_dropdown_x", "value"),
    Input("coordinates_plot_dropdown_y", "value"),
    Input("coordinates_plot_dropdown_z", "value"),
    Input("coordinates_plot_dropdown_color", "value")
)
def update_3d_coordinates_plot(selected_run, dropdown_x, dropdown_y, dropdown_z, color):
    if dropdown_x is None or dropdown_y is None or dropdown_z is None or selected_run is None:
        return {}

    file_path = f"{results_path}{selected_run}/"
    xi = np.load(file_path + "transition_manifold.npy")
    x_anchor = np.load(file_path + "x_data.npz")["x_anchor"]
    network = dm.open_network(file_path, "network")

    color_options, colors = calc_colors(x_anchor, network)
    fig = px.scatter_3d(x=xi[:, int(dropdown_x) - 1],
                        y=xi[:, int(dropdown_y) - 1],
                        z=xi[:, int(dropdown_z) - 1],
                        color=colors[color])
    fig.update_layout(
        scene=dict(
            xaxis_title_text=rf"$xi_{dropdown_x}$",
            yaxis_title_text=rf"$xi_{dropdown_y}$",
            zaxis_title_text=rf"$xi_{dropdown_z}$"
        ),
        clickmode="select"
    )

    # TODO Latex zum funktionieren bringen
    fig.layout.uirevision = 1  # This fixes the orientation over different plots.

    return fig


@callback(
    Output("network_plot", "figure"),
    Input("coordinates_plot_dropdown_runs", "value"),
    Input('3d_coordinates_plot', 'clickData'),
)
def update_network_plot(selected_run, click_data):
    if selected_run is None:
        return {}

    file_path = f"{results_path}{selected_run}/"
    x_anchor = np.load(file_path + "x_data.npz")["x_anchor"]
    network = dm.open_network(file_path, "network")

    if ctx.triggered_id == "coordinates_plot_dropdown_runs":

        selected_anchor = x_anchor[np.random.randint(0, x_anchor.shape[0])]
        network_fig = create_figure_from_network(network, selected_anchor)
        return network_fig

    selected_anchor = x_anchor[click_data["points"][0]["pointNumber"]]
    fig_patch = Patch()
    fig_patch["data"][1]["marker"]["color"] = np.logical_not(selected_anchor.astype(bool)).astype(int)

    # degrees = np.array(list(network.degree))[:, 1:2].astype(int).flatten()
    # anchor_degrees = degrees[selected_anchor.astype(bool)]
    # avg_deg = sum(anchor_degrees) / len(anchor_degrees)

    return fig_patch


# App layout
app.layout = html.Div([
    html.Div(children="Run overview"),
    create_table(df, "data_table"),
    html.Button("Unselect all", id="unselect_all", n_clicks=0),
    create_tabs("plot_tabs")
    ])


def main() -> None:
    app.run(debug=True)


if __name__ == '__main__':
    main()