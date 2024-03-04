from dash import Dash, html, dash_table, Output, Input, callback, dcc
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import colorlover
import pandas as pd
import sponet_cv_testing.datamanagement as dm
import numpy as np
import sponet.collective_variables as cv

app = Dash(__name__)

df = dm.read_data_csv().reset_index()


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

    colorscale_styles_r, _ = discrete_background_color_bins(df, columns=["r_ab", "r_ba"])
    colorscale_styles_rt, _ = discrete_background_color_bins(df, columns=["rt_ab", "rt_ba"])
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

    table = dash_table.DataTable(
        id=table_id,
        data=data.to_dict('records'),
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
def sort_table(sort_by):
    if len(sort_by):
        sdata = df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False)
    else:
        sdata = df

    return sdata.to_dict("records")


def create_tabs(tabs_id: str ) -> dcc.Tabs:
    tabs = dcc.Tabs(id=tabs_id, value="tab-1", children=[
        dcc.Tab(label="Overview",
                value="tab-1",
                children=html.Div(id="overview_plot")),
        dcc.Tab(
            label="Coordinate",
            value="tab-2",
            children=[
                dcc.Store(id="coordinates_plot_selected_runs", storage_type="memory"),
                html.Div([
                    html.Div([
                        html.Label("Runs:"),
                        dcc.Dropdown(id="coordinates_plot_dropdown_runs",
                                     style={"width": "45vw"})
                    ]),
                    html.Div([
                        html.Label("x-axis:"),
                        dcc.Dropdown(id="coordinates_plot_dropdown_x",
                                     style={'width': '10vw'}),
                    ]),
                    html.Div([
                        html.Label("y-axis:"),
                        dcc.Dropdown(id="coordinates_plot_dropdown_y",
                                     style={'width': '10vw'}),
                    ]),
                    html.Div([
                        html.Label("z-axis:"),
                        dcc.Dropdown(id="coordinates_plot_dropdown_z",
                                     style={'width': '10vw'}),
                    ]),
                    html.Div([
                        html.Label("color:"),
                        dcc.Dropdown(id="coordinates_plot_dropdown_color",
                                     options=["shares", "weighted_shares"],
                                     value="weighted_shares",
                                     style={'width': '10vw'})
                    ])
                ], style={'display': 'flex', 'flex-direction': 'row'}),
                html.Div(id="coordinates_plot"),
                html.Div(id="3d_coordinates_plot")]),
    ])
    return tabs


@callback(
    Output("overview_plot", "children"),
    Input("data_table", "data"),
)
def update_overview_graph(rows):
    dff = pd.DataFrame(rows)
    dff["ab_ratio"] = dff["r_ab"]/dff["rt_ab"]
    dff["ba_ratio"] = dff["r_ba"] / dff["rt_ba"]
    dim_estimate_mean = dff["dim_estimate"].mean()
    dff["dim_estimate"] = dff["dim_estimate"].fillna(-1)

    fig = (px.parallel_coordinates
           (dff, color="dim_estimate",
            dimensions=["ab_ratio", "ba_ratio", "lag_time", "dim_estimate"],
            color_continuous_midpoint=dim_estimate_mean,
            color_continuous_scale=px.colors.diverging.Picnic)) #["r_ab", "rt_ab", "r_ba", "rt_ba", "lag_time", "dim_estimate"]))
    return html.Div(dcc.Graph(figure=fig, mathjax=True))


@callback(
    Output("coordinates_plot_dropdown_runs", "options"),
    Output("coordinates_plot_dropdown_x", "options"),
    Output("coordinates_plot_dropdown_x", "value"),
    Output("coordinates_plot_dropdown_y", "options"),
    Output("coordinates_plot_dropdown_y", "value"),
    Output("coordinates_plot_dropdown_z", "options"),
    Output("coordinates_plot_dropdown_z", "value"),
    Input("data_table", "data"),
    Input("data_table", "selected_rows"),
    prevent_initial_call=True
)
def update_coords_plot_dropdown(data, selected_rows):
    if not selected_rows:
        return(
            ["select a run"],
            ["select a run"],
            "select a run",
            ["select a run"],
            "select a run",
            ["select a run"],
            "select a run"
        )


    dff = pd.DataFrame([data[i] for i in selected_rows])

    if not dff["finished"].any():
        return (
            ["select a run"],
            ["select a run"],
            "select a run",
            ["select a run"],
            "select a run",
            ["select a run"],
            "select a run"
        )

    finished_mask = dff["finished"]
    dff = dff[finished_mask]

    min_cv_coords: int = dff["cv_dim"].min()
    options = [f"{i}" for i in range(1, min_cv_coords + 1)]

    return (
        dff["run_id"].tolist(),
        options,
        "1",
        options,
        "2",
        options,
        "3"
    )


@callback(
    Output("coordinates_plot_selected_runs", "data"),
    Input("coordinates_plot_dropdown_runs", "value")
)
def update_coordinates_plot_selected_from_dropdown(value):
    return value


@callback(
    Output("coordinates_plot", "children"),
    Input("coordinates_plot_selected_runs", "data"),
    Input("coordinates_plot_dropdown_x", "value"),
    Input("coordinates_plot_dropdown_y", "value"),
    Input("coordinates_plot_dropdown_color", "value")
)
def update_coordinates_plot(selected_run, dropdown_x, dropdown_y, color):
    if not selected_run:
        return html.Div("No run selected")
    file_path = f"../data/results/{selected_run}/"
    xi = np.load(file_path + "transition_manifold.npy")
    x_anchor = np.load(file_path + "x_data.npz")["x_anchor"]
    network = dm.open_network(file_path, "network")

    color_options, colors = calc_colors(x_anchor, network)

    fig = px.scatter(x=xi[:, int(dropdown_x) - 1], y=xi[:, int(dropdown_y) - 1],
                      color=colors[color],
                      labels={"x": rf"$\xi_{dropdown_x}$", "y": rf"$\xi_{dropdown_y}$", "color": "c"}
                      )
    return dcc.Graph(figure=fig, mathjax=True, style={'width': '50vw', 'height': '45vh'})


@callback(
    Output("3d_coordinates_plot", "children"),
    Input("coordinates_plot_selected_runs", "data"),
    Input("coordinates_plot_dropdown_x", "value"),
    Input("coordinates_plot_dropdown_y", "value"),
    Input("coordinates_plot_dropdown_z", "value"),
    Input("coordinates_plot_dropdown_color", "value")
)
def update_3d_coordinates_plot(selected_run, dropdown_x, dropdown_y, dropdown_z, color):
    file_path = f"../data/results/{selected_run}/"
    xi = np.load(file_path + "transition_manifold.npy")
    x_anchor = np.load(file_path + "x_data.npz")["x_anchor"]
    network = dm.open_network(file_path, "network")

    color_options, colors = calc_colors(x_anchor, network)
    fig = px.scatter_3d(x=xi[:, int(dropdown_x) - 1],
                        y=xi[:, int(dropdown_y) - 1],
                        z=xi[:, int(dropdown_z) - 1],
                        color=colors[color],
                        labels={"x": rf"$\xi_{dropdown_x}$", "y": rf"$\xi_{dropdown_y}$", "color": "c"}
                     )
    return dcc.Graph(figure=fig, mathjax=True, style={'width': '80vw', 'height': '80vh'})



# App layout
app.layout = html.Div([
    html.Div(children="Run overview"),
    create_table(df, "data_table"),
    create_tabs("main_tabs")


    ])


def main() -> None:
    app.run(debug=True)


if __name__ == '__main__':
    main()

