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
def sort_table_cb(sort_by):
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

# App layout
app.layout = html.Div([
    html.Div(children="Run overview"),
    create_table(df, "data_table"),
    ])


def main() -> None:
    app.run(debug=True)


if __name__ == '__main__':
    main()