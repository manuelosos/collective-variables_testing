import json
import re
import sys

import colorlover
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as ps
import sponet.collective_variables as cv
from dash import Dash, html, dash_table, Output, Input, State, callback, dcc, Patch, ctx
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import sponet_cv_testing.datamanagement as dm

app = Dash(__name__)

args = sys.argv[1:]


# Adjust path in following function call if necessary
results_path: str = "../data/results/"

results_table_path = "results_table.csv"

# call with command line argument 1 to load test csv
#Calling with small dataset significantly reduces initial load time
if len(args) > 0:
    test = args[0]
    if test == "1":
        results_table_path = "test_table.csv"

df = dm.read_data_csv(f"{results_path}{results_table_path}")
# Pre-filtering of the data can be done here
#df = df[df["dim_estimate"] >= 1]


def compute_network_plot_trace(network: nx.Graph, seed: int=100):
    pos = nx.spring_layout(network, seed=seed, k=2 / np.sqrt(len(network)))
    pos = np.array(list(pos.values()))

    node_trace = go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode="markers",
        marker=dict(showscale=True),
    )

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

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
    )
    return node_trace, edge_trace


def discrete_background_color_bins(df: pd.DataFrame, n_bins: int = 5, columns: str | list[str]='all'):
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


def get_reruns(data, run_id: str) -> list[str]:

    reruns = []

    compare_run_id = re.sub("_r?\d{1,2}$","", run_id)

    for i in range(len(data)):
        entry = data[i]["run_id"]
        if re.sub("_r?\d{1,2}$", "", entry) == compare_run_id:
            if entry != run_id:
                reruns.append(entry)


    return reruns


def get_table_html(data: pd.DataFrame, table_id: str) -> dash_table.DataTable:

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
        },
        {
            "if": {
                "filter_query": "{remarks} contains 'slb'",
                "column_id": "dim_estimate"
            },
            "backgroundColor": "YELLOW"
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
        dict(id="finished", name="finished", selectable=True, type="text"),
        dict(id="remarks", name="remarks", selectable=True, type="text")
    ]

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


def get_tabs_html(tabs_id: str) -> dcc.Tabs:
    tabs = dcc.Tabs(id=tabs_id, value="tab_2", children=[
        #dcc.Tab(label="Overview Plots",
        #       value="tab_1",
        #      children=html.Div(id="overview_plot")),
        dcc.Tab(label="Coordinate Plot",
                value="tab_2",
                children=[
                    get_coords_network_dropdowns_html(),

                    get_run_specifics_html(),

                    get_coords_network_plots_html(),

                    get_cv_plots_html(),

                    html.Div([
                        html.H4("Logs"),
                        html.Pre("Krass hier steht text",
                                 id="runlog")
                    ]),

                ])
    ])
    return tabs


def get_coords_network_dropdowns_html():
    coords_network_plots = html.Div([
        # Dropdowns #####################
        html.Div([
            html.Label("Runs:"),
            dcc.Dropdown(id="coordinates_plot_dropdown_runs",
                         placeholder="Select entries in the table above by clicking on the boxes on the left side.",
                         style={"width": "45vw"}
                         ),
            html.Div([
                html.Div([
                    html.Button("Add reruns", id="add_reruns_button", n_clicks=0),
                ]),
                dcc.Clipboard(
                    target_id="coordinates_plot_dropdown_runs",
                    title="Copy run id",
                    style={"width": "2vw"}
                ),
            ], style={'display': 'flex', 'flex-direction': 'row'})
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
    ], style={'display': 'flex', 'flex-direction': 'row'})

    return coords_network_plots


def get_run_specifics_html():
    specifics = html.Div([
        html.Table([
            html.Thead(
                html.Tr([
                    html.Td("r_ab"),
                    html.Td("r_ba"),
                    html.Td("rt_ab"),
                    html.Td("r_ba"),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(id="selected_run_r_ab"),
                    html.Td(id="selected_run_r_ba"),
                    html.Td(id="selected_run_rt_ab"),
                    html.Td(id="selected_run_rt_ba"),
                ])

            ])
        ], style={"border": "1px solid"}
        )
    ])
    return specifics


def get_coords_network_plots_html():
    coords_network_plots = html.Div([
        dcc.Loading([
            dcc.Graph(
                id="3d_coordinates_plot",
                mathjax=True,
                style={'width': '60vw', 'height': '70vh'}
            )],
            overlay_style={"visibility": "visible", "opacity": .5},
            delay_show=200
        ),
        dcc.Loading([
            dcc.Graph(
                id="network_plot",
                mathjax=True,
                style={'width': '40vw', 'height': '70vh'}
            )],
            overlay_style={"visibility": "visible", "opacity": .5},
            delay_show=200
        ),
    ], style={'display': 'flex', 'flex-direction': 'row'})
    return coords_network_plots


def get_cv_plots_html():

    cv_plots = html.Div([
        html.Div([
            html.Div([
                html.Label("cv type:"),
                dcc.Dropdown(id="cv_selector_dropdown",
                             options=[
                                 {'label': 'Non Weighted', 'value': 'non_weighted'},
                                 {'label': 'Degree Weighted', 'value': 'degree_weighted'}
                             ],
                             value="degree_weighted",
                             style={"width": "10vw"}
                             )
            ],),
            html.Div([
                html.Label("node scaling"),
                dcc.Dropdown(id="node_scaling_dropdown",
                             options=[
                                 {'label': 'Linear', 'value': 'linear'},
                                 {'label': 'Logarithmic', 'value': 'logarithmic'}
                             ],
                             value="linear",
                             style={"width": "10vw"}
                             )
            ], ),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.Div([
            dcc.Loading([
                dcc.Graph(
                    id="cv_network_plots",
                    mathjax=True,
                    style={'width': '50vw', 'height': '125vh'}
                )],
                overlay_style={"visibility": "visible", "opacity": .5},
                delay_show=200,
            ),
            dcc.Loading([
                dcc.Graph(
                    id="cv_xixifit_plots",
                    mathjax=True,
                    style={'width': '50vw', 'height': '125vh'}
                )],
                overlay_style={"visibility": "visible", "opacity": .5},
                delay_show=200,
            )
        ], style={'display': 'flex', 'flex-direction': 'row'}),
    ])

    return cv_plots


@callback(Output("coordinates_plot_dropdown_runs", "options"),
          Input("data_table", "data"),
          Input("data_table", "selected_rows"),
          Input("add_reruns_button", "n_clicks"),
          State("coordinates_plot_dropdown_runs", "value")
          )
def update_run_dropdown(data, selected_rows: list[int], _, selected_run):
    if not selected_rows:
        return []

    selected_rows_values = [data[i]["run_id"] for i in selected_rows]

    reruns = []
    if ctx.triggered_id == "add_reruns_button":
        reruns = get_reruns(data, selected_run)
        selected_rows_values.extend(reruns)

    return  selected_rows_values


@callback(
    Output("coordinates_plot_dropdown_x", "options"),
    Output("coordinates_plot_dropdown_y", "options"),
    Output("coordinates_plot_dropdown_z", "options"),
    Input("coordinates_plot_dropdown_runs", "value")
)
def update_coord_plot_coord_dd_cb(run_id: str) -> list[list[str]]:
    if run_id is None:
        raise PreventUpdate
    run = df.loc[run_id]
    options = [f"{i}" for i in range(1, run["cv_dim"] + 1)]
    return [options, options, options]


@callback(
    Output("selected_run_r_ab", "children"),
    Output("selected_run_r_ba", "children"),
    Output("selected_run_rt_ab", "children"),
    Output("selected_run_rt_ba", "children"),
    Input("coordinates_plot_dropdown_runs", "value")
)
def update_selected_run_specifics_table(selected_run: str):
    if selected_run is None:
        return(["" for i in range(4)])
    file_path = f"{results_path}{selected_run}/"
    with open(file_path+"parameters.json", "r") as file:
        run_params = json.load(file)
    return dm._translate_run_rates(run_params["dynamic"]["rates"])



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
        raise PreventUpdate
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
            xaxis_title_text=rf"dim {dropdown_x}",
            yaxis_title_text=rf"dim {dropdown_y}",
            zaxis_title_text=rf"dim {dropdown_z}"
        ),
        font_size=15,
        clickmode="select"
    )

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

        node_trace, edge_trace = compute_network_plot_trace(network)

        selected_anchor = x_anchor[np.random.randint(0, x_anchor.shape[0])]
        node_trace.marker.color = np.logical_not(selected_anchor.astype(bool)).astype(int)

        node_trace.customdata = np.transpose(np.array([np.array(network.degree)[:, 1:].flatten().astype(int)]))
        node_trace.hovertemplate = ("<b>Degree</b>: %{customdata[0]}<br>"
                                    "<extra></extra>")
        node_trace.marker.showscale=False

        layout = go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            coloraxis=dict(showscale=False))

        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        return fig

    selected_anchor = x_anchor[click_data["points"][0]["pointNumber"]]
    fig_patch = Patch()
    fig_patch["data"][1]["marker"]["color"] = np.logical_not(selected_anchor.astype(bool)).astype(int)

    # degrees = np.array(list(network.degree))[:, 1:2].astype(int).flatten()
    # anchor_degrees = degrees[selected_anchor.astype(bool)]
    # avg_deg = sum(anchor_degrees) / len(anchor_degrees)

    return fig_patch


@callback(
    Output("cv_network_plots", "figure"),
    Input("coordinates_plot_dropdown_runs", "value"),
    Input("cv_selector_dropdown", "value"),
    Input("node_scaling_dropdown", "value"),
)
def update_cv_network_plots(selected_run: str, cv_type: str, scaling: str):
    if selected_run is None:
        return {}

    #{'points': [{'curveNumber': 5, 'pointNumber': 344, 'pointIndex': 344, 'x': -0.4659979045391083, 'y': -0.14735986292362213,
    # 'marker.size': 9, 'marker.color': -0.9999469822103596, 'bbox': {'x0': 216.69, 'x1': 225.69, 'y0': 2693.5875, 'y1': 2702.5875}, 'customdata': [2, -0.9999469822103596]}]}
    coords_n = 4

    file_path = f"{results_path}{selected_run}/"
    network = dm.open_network(file_path, "network")

    cv_path = "cv_optim.npz"
    if cv_type == "degree_weighted":
        cv_path = "cv_optim_degree_weighted.npz"

    cv_optim = np.load(file_path + cv_path)
    alphas = cv_optim["alphas"]

    degrees = np.array(network.degree)[:, 1:].flatten().astype(int)

    if scaling == "linear":
        lower_scaling = 2
        upper_scaling = 0.5
        size_adjustment = np.vectorize(lambda x: 0.5*x+8)
        #size_adjustment = np.vectorize(lambda x: ((x - min(degrees)) * (upper_scaling - lower_scaling) / (
         #       max(degrees) - min(degrees)) + lower_scaling) * x)
    else:
        size_adjustment = np.vectorize(lambda x: np.log(300 * x))

    degrees_adjusted = size_adjustment(degrees)

    # Patch if possible
    if ctx.triggered_id == "node_scaling_dropdown":
        fig_patch = Patch()

        for i in range(1, coords_n*2, 2):
            fig_patch["data"][i]["marker"]["size"] = degrees_adjusted

        return fig_patch

    if ctx.triggered_id == "cv_selector_dropdown":
        fig_patch = Patch()

        for i in range(0, coords_n):
            this_alpha = alphas[:, i] / np.max(np.abs(alphas[:, i]))

            fig_patch["data"][i*2 + 1]["marker"]["color"] = this_alpha
            fig_patch["data"][i*2 + 1]["marker"]["coloraxis"] = "coloraxis"

        return fig_patch

    fig = ps.make_subplots(rows=4,
                           cols=1,
                           vertical_spacing=0.005,
                           horizontal_spacing=0,
                           shared_xaxes=True)

    node_trace, edge_trace = compute_network_plot_trace(network)

    node_trace.marker.size = degrees_adjusted

    custom_data = [degrees]


    for i in range(0, coords_n):
        this_alpha = alphas[:, i] / np.max(np.abs(alphas[:, i]))


        node_trace.marker.color = this_alpha
        node_trace.marker.coloraxis = "coloraxis"

        #node_trace = pd.DataFrame(data={"degrees": degrees,
                                            # "color": this_alpha})

        node_trace.customdata = np.transpose(np.stack([degrees, this_alpha]))
        node_trace.hovertemplate = ("<b>Degree</b>: %{customdata[0]}<br>"
                                    "<b>Value</b>: %{customdata[1]:.3f}<br>"
                                    "<extra></extra>")

        fig.add_trace(edge_trace, row=i + 1, col=1)
        fig.add_trace(node_trace, row=i + 1, col=1)

    fig.update_layout(
                      showlegend=False,
                      hovermode='closest',
                      coloraxis=dict(colorscale="portland"))

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig

@callback(
    Output("cv_xixifit_plots", "figure"),
    Input("coordinates_plot_dropdown_runs", "value"),
    Input("cv_selector_dropdown", "value")
)
def update_cv_xixifit_plots(selected_run: str, cv_type: str):
    if selected_run is None:
        return {}

    file_path = f"{results_path}{selected_run}/"

    coords_n: int = 4

    cv_path = "cv_optim.npz"
    if cv_type == "degree_weighted":
        cv_path = "cv_optim_degree_weighted.npz"
    cv_optim = np.load(file_path + cv_path)
    xi = np.load(file_path + "transition_manifold.npy")
    xi_fit = cv_optim["xi_fit"]

    fig = ps.make_subplots(rows=4,
                           cols=1,
                           vertical_spacing=0.005,
                           horizontal_spacing=0,
                           shared_xaxes=True,
                           shared_yaxes=True)

    for i in range(0, coords_n):

        xi_xifit = go.Scatter(
            x=xi[:, i],
            y=xi_fit[:, i],
            mode="markers",
        )
        xi_xifit.marker.coloraxis = "coloraxis"
        fig.add_trace(xi_xifit, row=i + 1, col=1)
        fig.update_yaxes(title_text=f"$\\bar\\varphi_{i+1}$", row=i+1, col=1)


    fig.update_layout(showlegend=False)
    fig.update_xaxes(zeroline=False, showticklabels=True)
    fig.update_yaxes(zeroline=False, showticklabels=True)
    fig.update_xaxes(title_text="$\\varphi_i$", row=coords_n, col=1)

    return fig


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


# App layout
app.layout = html.Div([
    html.Div(children="Run overview"),
    get_table_html(df, "data_table"),
    html.Button("Unselect all", id="unselect_all", n_clicks=0),
    get_tabs_html("plot_tabs")
    ])


def main() -> None:
    app.run(debug=True)


if __name__ == '__main__':

    main()