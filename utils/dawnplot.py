
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import bokeh.layouts as bkl
import bokeh.plotting as bk
import bokeh.transform as btr
from io import BytesIO
import base64
from itertools import zip_longest, cycle
from bokeh.models import *
from bokeh.palettes import *

class EmbeddingPlot:
    def __init__(self, displayProperties = {}, theme = 'rainbow'):
        self.displayProperties = displayProperties
        self.theme = theme
        self.embedding_df = pd.DataFrame()
        self.make_color_maps()
        self.make_tooltip_graphs()
        for key, value in self.displayProperties.items():
            if value.get('tooltip') or value.get('color'):
                self.embedding_df[key] = value['data']
    
    def make_color_maps(self, log=False):
        palette = lambda n: [(int(r), int(g), int(b), a) for r, g, b, a in (plt.get_cmap(self.theme)(np.linspace(0, 1, n)) * [255, 255, 255, 1])]
        _color_attrs = [(key, value['data']) for key, value in self.displayProperties.items() if value.get('color')]
        color_maps = {}
        for key, value in _color_attrs:
            if isinstance(value[0], str):
                color_maps[key] = btr.factor_cmap(key, palette(len(list(set(value)))), list(set(value)))
            else:
                color_maps[key] = btr.linear_cmap(key, palette(600), low=min(value), high=max(value))
                color_maps[key+"_log"] = btr.log_cmap(key, palette(600), low=min(value), high=max(value))
        self.colorMaps = color_maps
        self._default_color_attr = list(color_maps.keys())[0]
        self._color_options = [k for k in color_maps.keys() if not k.endswith('_log')]
    
    def make_tooltip_graphs(self):
        data = [value['data'] for key, value in self.displayProperties.items() if value.get('graph')]
        graphs = []
        for y_data in zip_longest(*data):
            # make img src base64 of plot
            img = BytesIO()
            fig = plt.figure(figsize=(2,1))
            for p, color in zip(y_data, cycle(['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive'])):
                plt.plot(p, color=color, linewidth=0.5)
            plt.axis('off')
            plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            graphs.append('data:image/png;base64,' + base64.b64encode(img.getvalue()).decode())  
        self.embedding_df['graph'] = graphs

    def run_UMAP(self):
        self._embedding_options = []
        for key, value in self.displayProperties.items():
            if value.get('umap'):
                self._embedding_options.append(key)
                reducer = umap.UMAP(**value['umap'])    
                reducer.fit(value['data'])
                self.embedding_df[f'{key}_x'], self.embedding_df[f'{key}_y'] = reducer.embedding_[:,0], reducer.embedding_[:,1]
        self._default_embedding = self._embedding_options[0]
        
    def update(self, displayProperties=None):
        if displayProperties: self.displayProperties = displayProperties
        self.make_color_maps()
        self.make_tooltip_graphs()
        for key, value in self.displayProperties.items():
            if value.get('tooltip') or value.get('color'):
                self.embedding_df[key] = value['data']
    
    def _plot(self, **kwargs):
        self.tooltips = f"""
            <div>
                {"<div>" + "".join([f'<div><span style="font-size: 15px; font-weight: bold;">{attr_key}: @{attr_key}</span></div>' for attr_key, value in self.displayProperties.items() if value.get('tooltip')]) + "</div>"}
                {"<div><img src='@graph' style='margin: 5px 5px 5px 5px'/></div>" if any([value.get('graph') for key, value in self.displayProperties.items()]) else ""}
                <hr style="margin: 5px 5px 5px 5px"/><br>
            </div>
        """ if any([value.get('tooltip') for key, value in self.displayProperties.items()]) else None
        
        self.embeddingSelect = Select(
            title='Embedding',
            value=self._default_embedding,
            options=self._embedding_options
        )
        self.coloringSelect = Select(
            title='Color by',
            value=self._default_color_attr,
            options=self._color_options
        )
        self.logCheck = Checkbox(
            label='Log color scale',
            active=False,
        )
        self.fig = bk.figure(tools=(
            CopyTool(), 
            HoverTool(tooltips=self.tooltips), 
            LassoSelectTool()
        ), **kwargs)
        self.plotPoints = self.fig.circle(
            x=f'{self._default_embedding}_x',
            y=f'{self._default_embedding}_y',
            source=self.embedding_df, 
            size=7,
            line_color='black',
            line_width=0.25,
            fill_alpha=0.7,
            fill_color=self.colorMaps[self._default_color_attr],
        )
        self.colorbar = self.plotPoints.construct_color_bar(padding=1)
        self.fig.add_layout(self.colorbar, 'right')
        self.embeddingSelect.js_on_change('value', CustomJS(
            args=dict(plotPoints=self.plotPoints),
            code=" plotPoints.glyph.x = cb_obj.value + '_x'; plotPoints.glyph.y = cb_obj.value + '_y'; "
        ))
        self.coloringSelect.js_on_change('value', CustomJS(
            args=dict(plotPoints=self.plotPoints, color_maps=self.colorMaps, colorbar=self.colorbar, logCheck=self.logCheck),
            code="""
            const selectedAttr = cb_obj.value;
            
            if (logCheck.active) {
                plotPoints.glyph.fill_color = color_maps[selectedAttr + '_log'];
                colorbar.color_mapper = color_maps[selectedAttr + '_log']['transform'];
            } else {
                plotPoints.glyph.fill_color = color_maps[selectedAttr];
                colorbar.color_mapper = color_maps[selectedAttr]['transform'];
            }
            """
        ))
        self.logCheck.js_on_change('active', CustomJS(
            args=dict(plotPoints=self.plotPoints, color_maps=self.colorMaps, colorbar=self.colorbar, coloringSelect=self.coloringSelect),
            code="""
            const selectedAttr = coloringSelect.value;
            
            if (cb_obj.active) {
                plotPoints.glyph.fill_color = color_maps[selectedAttr + '_log'];
                colorbar.color_mapper = color_maps[selectedAttr + '_log']['transform'];
            } else {
                plotPoints.glyph.fill_color = color_maps[selectedAttr];
                colorbar.color_mapper = color_maps[selectedAttr]['transform'];
            }
            """
        ))
        self.fig.grid.visible = False
        self.fig.axis.visible = False
        self.plot = bkl.gridplot([[self.embeddingSelect, self.coloringSelect, self.logCheck], [self.fig]],toolbar_location='below')
        
    def show_notebook(self, save_filename='', **kwargs):
        bk.output_notebook()
        self._plot(**kwargs)
        bk.show(self.plot)
        if save_filename: 
            bk.save(self.plot, save_filename)