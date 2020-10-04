import plotly
import plotly.graph_objects as go
import plotly.io as pio
import os
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pio.renderers.default = "browser"

root_path = os.getcwd()

data_rental = {}
file_type = ['medianAskingRent', 'discountShare', 'rentalInventory']
apt_type = ['Studio', 'OneBd', 'TwoBd', 'ThreePlusBd']
rename = ['price', 'discount', 'inventory']
for apt in apt_type:
    data = []
    for i, file in enumerate(file_type):
        data_raw = pd.read_csv(os.path.join(root_path, 'Rental', apt, f'{file}_{apt}.csv'))
        cols = data_raw.columns.tolist()
        cols_val = [cols[i] for i in range(3, len(cols))]
        data_rental[(apt, file)] = pd.melt(data_raw, id_vars=['areaName', 'Borough', 'areaType'], value_vars=cols_val)
        data_rental[(apt, file)].columns = ['areaName', 'Borough', 'areaType', 'date', rename[i]]
        data_rental[(apt, file)].dropna(subset=['areaName', 'Borough', 'areaType'], inplace=True)
        data.append(data_rental[(apt, file)])
    data_rental[(apt, 'combo')] = pd.merge(pd.merge(data[0], data[1], on=['areaName', 'Borough', 'areaType', 'date']),
                                           data[2], on=['areaName', 'Borough', 'areaType', 'date'])

# ----quick check of the relationship using seaborn-----
# sns.lineplot(data=data_rental[('Studio','medianAskingRent')],x='date',y='price',hue='Borough')
# sns.lineplot(data=data_rental[('Studio','rentalInventory')],x='date',y='price',hue='Borough')
# sns.lineplot(data=data_rental[('Studio','discountShare')],x='date',y='price',hue='Borough')


# ----use plotly express------
# fig = px.line(data_rental[('Studio', 'medianAskingRent')], x='date', y='price', color='areaName')
# fig.show()

#-----use figure factory-----
areaList = data_rental[('Studio', 'combo')].areaName.unique().tolist()
buttons = []
for i, area in enumerate(areaList):

    visibility = [False] * len(areaList) * len(apt_type)*2
    for k in range(0, 8): visibility[i * len(apt_type)*2 + k] = True
    button_dict = {'method': 'restyle',
                   'label': f'{area}',
                   'args': [{'visible': visibility}]}
    buttons.append(button_dict)

fig = plotly.subplots.make_subplots(rows=2,  subplot_titles=('Median Ask Price', "Inventory"))
color_map=['#30517A','#ADD0FB','#61A6FA','#17D0FB']
for area in areaList:
    for i, apt in enumerate(apt_type):
        visibility_ = True if area == area_default else False
        df = data_rental[(apt, 'combo')][data_rental[(apt, 'combo')].areaName == area]
        fig.add_trace(go.Scatter(x=df['date'],
                                 y=df['price'],
                                 name=f'{area}_{apt}_price',
                                 visible=visibility_,
                                 line=dict(color=color_map[i]),
                                 showlegend=False,
                                 mode='lines+markers',
                                 hovertemplate=apt+': %{y:$.0f}<extra></extra>'),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=df['date'],
                             y=df['inventory'],
                             name=f'{area}_{apt}',
                             visible=visibility_,
                             marker_color=color_map[i],
                             hovertemplate=apt+': %{y:.0f}<extra></extra>'),
                      row=2, col=1)

fig.update_layout(title_text='NYC Rental Market',
                  title_x=0.5,
                  title_font_size=24,
                  width=1200,
                  height=700,
                  barmode='stack',
                  updatemenus=[dict(active=0,
                                    buttons=buttons,
                                    x=1.1,
                                    y=1,
                                    xanchor='left',
                                    yanchor='top')],
                  legend=dict(
                      yanchor="bottom",
                      y=0,
                      xanchor="left",
                      x=1.1),
                  hovermode="x unified"
                  )

#fig.show()
plotly.offline.plot(fig, filename='NYC rental market.html')

