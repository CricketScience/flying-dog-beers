import dash
from dash.dependencies import Input, Output, State#, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

import numpy as np
from scipy import stats
import sympy as sym

from functools import reduce


app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
server = app.server
app.title= 'Newsvendor Model'


params = dict(demand_std = [200, 2000],
              service_loss_factor = [1,10],
              variable_cost = [0.0, 1],
              salvage_price = [-1, 1],
              transaction_cost = [0, 1])

default_params = dict(demand_std = 1000,
                      service_loss_factor = 3,
                      variable_cost = 0.1,
                      salvage_price = -0.05,
                      transaction_cost = 0.1)

sliders = reduce(list.__add__, [[html.Div(id = k + '-show'),
        dcc.Slider(
        id=k,
        min=low,
        max=high,
        value=default_params[k],
        step=(high-low)/100
        )
                    ]
            for k, (low, high) in params.items()])

app.layout = html.Div([
	
	html.H3('News Vendor Optimization Model'),
	html.P("""The newsvendor model determines the optimal inventory position for 'single-stage' problems, i.e., there is no opportunity to restock during
	the sales period (shelf-life) of the product."""),
	    html.P("""You can use the sliders below to explore how the optimal inventory position changes in response to certain factors."""),
    html.P("""
The following assumes that the expected demand is for 1000 products and our uncertainty takes a log-normal distribution."""),
    dcc.Markdown(
        """
Demand STD
: the standard deviation of our demand forecast.

Service Loss_factor
:  This scales our lost sales to account for the consumer lifetime loss due to our failing to serve the customer.  
: Use a factor of 1 if you want to assume that our only loss is the immediate loss of the sale revenue.

Variable cost
:  The fraction of the retail price that goes into production (and other costs ammortized across both sold and unsold inventory)

Transaction cost
:  The fraction of the retail price that is ammortized across only the sold items (e.g., shipping).

Salvage price
: The amount we recoup of unsold inventory less costs ammortized across only unsold items (e.g., destruction).
"""),
    html.Div(
            [
                dcc.Graph(id='test-figure', animate=False),
                html.Hr()

		] + sliders)])



def show_slider(slider_name):
    return lambda val: f"{slider_name}: {val}"


for param in params.keys():
    app.callback(Output(param + "-show", "children"),
                 Input(param, "value")).__call__(
                     show_slider(param))


def visualize(*args):

    #data = [go.Bar(x=np.arange(slider1), y=np.arange(slider1)),
    #        go.Bar(x=np.arange(slider2), y=np.arange(slider2))]
    #layout = go.Layout(title='test')

    #figure = go.Figure(data = data, layout = layout)

    print(args)

    the_args = dict(zip(params.keys(), args))
    the_args['retail_price'] = 1
    print(the_args)
                    
    
    figure = get_fig(**the_args)

    return figure


app.callback(Output('test-figure', 'figure'),
             [Input(param, 'value') for param in params.keys()]).__call__(
                 visualize)

def get_fig(demand_dist = stats.lognorm,
            transaction_cost = 0,
            demand_mean = 1000,
            demand_std = 1000,
            service_loss_factor = 1,
            retail_price = 20,
            variable_cost = 2,
            salvage_price = -1):
    if demand_dist is not stats.lognorm:
        raise NotImplementedError()
    #import pdb; pdb.set_trace()
    lost_sale_cost = service_loss_factor * retail_price - transaction_cost

    mu,sigma = sym.symbols('mu,sigma')
    mean_eq = sym.Eq(sym.exp(mu + sigma**2/2), demand_mean)
    var_eq = sym.Eq((sym.exp(sigma**2) - 1) * (sym.exp(2 * mu + sigma**2)), demand_std **2)
    result = sym.solve([mean_eq, var_eq], (mu, sigma))


    results = np.array([[ri.evalf() for ri in r] for r in result], dtype=float)
    mu, sigma = results[(results > 0).all(1)][0]
    dist = demand_dist(s=sigma, scale=np.exp(mu))
    assert np.allclose(dist.stats(), (np.array(demand_mean), np.array(demand_std**2)))


    critical_quantile = (lost_sale_cost - variable_cost) / (lost_sale_cost - salvage_price)
    optimal_inventory = dist.ppf(critical_quantile)
    probability_discretization = 1e-5
    x = np.linspace(0, dist.isf(probability_discretization), int(probability_discretization**-1))
    mids = (x[1:] + x[:-1]) / 2
    probs = dist.cdf(x[1:]) - dist.cdf(x[:-1])

    lost_sales = lost_sale_cost * ((mids * probs)[::-1].cumsum() - mids[::-1] * probs[::-1].cumsum())[::-1]
    destruction_loss = (-salvage_price) * (mids - (mids * probs).cumsum()) * probs.cumsum()
    total_cost = (lost_sales + destruction_loss + variable_cost * mids)
    data = pd.DataFrame.from_dict(dict(lost_sales=lost_sales, salvage_cost=destruction_loss, total_cost=total_cost,
                                       demand=probs, units=mids, variable_production=(mids * variable_cost)))

    i_max = np.where((probs.cumsum() > 0.999))[0][0]
    dollar_cols = ["lost_sales", "salvage_cost", "variable_production","total_cost"]
    index_col = "units"
    y_max = data.iloc[:i_max][dollar_cols].max().max()
    data['demand_distribution'] = data.demand * y_max / probs[:i_max].max()
    fig = px.line(data.iloc[:i_max], x=index_col, y=dollar_cols + ["demand_distribution"])
    fig.add_vline(x=optimal_inventory, line_dash="dot",
                  annotation_text="Optimal Inventory", 
                  annotation_position="top right")
    fig.add_vline(x=dist.mean(), line_dash="dot",
                  annotation_text="Expected Demand", 
                  annotation_position="top left")
    fig.add_vrect(x0=dist.mean(), x1=optimal_inventory, 
                  annotation_text=f"{100 * (optimal_inventory - dist.mean()) / optimal_inventory:.0f}% excess",
                  annotation_position="top left",
                  fillcolor="green", opacity=0.25, line_width=0)
    return fig

if __name__ == '__main__':
	app.run_server(debug = True)
        
