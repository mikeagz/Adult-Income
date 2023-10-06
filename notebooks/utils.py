import pandas as pd

# Data Visualization
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def plot_continuos_histogram_matrix(data):
    
    '''
    This function identifies all continuous features within the dataset and plots
    a matrix of histograms for each attribute
    '''
    
    continuous_features=data.select_dtypes("number").columns.to_list()
    continuous_features.remove('Income')
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols)

    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            go.Histogram(
                x=data[feature],
                name=feature
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text='Frequency', row=row, col=col)
        fig.update_layout(
            title=f'<b>Histogram Matrix<br> <sup> Continuous Features</sup></b>',
            showlegend=False
        )

    fig.update_layout(
        height=350 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= 'plotly_white'
    )
    fig.show()

def plot_cat_histogram_matrix(dataframe):
    
    '''
    This function identifies all cat features within the dataset and plots
    a matrix of barplots for each attribute
    '''
    
    continuous_features=dataframe.select_dtypes("object").columns.to_list()
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols)

    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1
        data=dataframe[feature].value_counts().to_dict()
        x=list(data.keys())
        y=list(data.values())
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=feature
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)
        fig.update_layout(
            title=f'<b>Bar Plot Matrix<br> <sup> Categorical Features</sup></b>',
            showlegend=False
        )

    fig.update_layout(
        height=450 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= 'plotly_white'
    )
    fig.show()

def show_corr(correlation_matrix):
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            colorbar=dict(title="Correlación"),
            text=correlation_matrix.values.round(2),
            hoverinfo="text",
        )
    )
    annotations = []
    for i, col in enumerate(correlation_matrix.columns):
        annotations.append(
            dict(
                x=i,
                y=len(correlation_matrix.columns),
                text=col,
                showarrow=False,
                textangle=90,
            )
        )

    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.9,
            text="Valores de correlación",
            showarrow=False,
            font=dict(size=14),
        )
    )

    fig.update_layout(annotations=annotations)

    fig.update_layout(
        title="Matriz de Correlación de Pearson",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=800,
        height=900,
    )

    fig.show()