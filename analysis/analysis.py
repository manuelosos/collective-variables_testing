from dash import Dash, html, dash_table
import plotly.express as px
import pandas as pd
import sponet_cv_testing.datamanagement as dm



df = dm.read_data_csv()

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children="Run overview"),
    dash_table.DataTable(data=df.to_dict('records'), page_size=15)
])


def main() -> None:
    app.run(debug=True)

if __name__ == '__main__':
    main()

