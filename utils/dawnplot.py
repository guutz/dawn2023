import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from umap import UMAP
import bokeh.layouts as bkl
import bokeh.plotting as bk
import bokeh.transform as btr
from io import BytesIO
import base64
from itertools import zip_longest, cycle
from bokeh.models import *
from bokeh.palettes import *


class TimeSeriesFFTProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, norm_inputs=False):
        self.norm_inputs = norm_inputs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ Transforms a list of time series into a list of their real FFTs. """
        pad_len = max([len(x) for x in X])
        X = [np.concatenate([x, np.zeros(pad_len - len(x))]) for x in X]
        if self.norm_inputs: 
            X = [series / np.max(series) for series in X]
        X_fft = np.abs(np.fft.rfft(X))
        return X_fft
    
    
class FFT_UMAP:
    """
    A transformer that takes a list of time series and UMAPs their FFTs.
    """
    def __init__(self, norm_inputs=False, targets=None, **umap_args):
        self.norm_inputs = norm_inputs
        self.targets = targets
        self.umap_args = umap_args
    
    def __repr__(self):
        return f'FFT_UMAP(norm_inputs={self.norm_inputs}, targets={self.targets}, {str(self.umap)[5:-1]})'

    def fit(self, X):
        self.umap = UMAP(
            n_jobs=4,
            **self.umap_args,
        )
        self.fft = TimeSeriesFFTProcessor(norm_inputs=self.norm_inputs)
        self.fft.fit(X)
        X_fft = self.fft.transform(X)
        self.umap.fit(X_fft, y=self.targets)
        return self

    def transform(self, X):
        X_fft = self.fft.transform(X)
        return self.umap.transform(X_fft)


class PlotData:
    def __init__(self, title, data, show_in_table=True):
        self.title = title
        self.data = data
        self.show_in_table = show_in_table
        
        
class TooltipData(PlotData):
    def __init__(self, title, data, show_in_table=True):
        super().__init__(title, data, show_in_table=show_in_table)
        self.html = f'<div><span style="font-size: 12px; font-weight: bold;">{self.title}: @{self.title}</span></div>'

class ColorMap(PlotData):
    def __init__(self, title, data, palette='gnuplot2', categorical=False, high_transparent=False, show_in_table=True):
        super().__init__(title, data, show_in_table=show_in_table)
        self.palette = palette
        self.categorical = categorical
        if isinstance(self.data[0], str) or self.categorical:
            pal = self._palette_list(len(list(set(self.data))))
            if high_transparent: 
                pal[-1] = tuple(list(pal[-1])[:-1]+[0]) # this crap is just to make the highest category color transparent
            self.map = self.map_log = btr.factor_cmap(self.title, pal, sorted([str(i) for i in set(self.data)]))
        else:
            self.map = btr.linear_cmap(self.title, self._palette_list(1000), low=min(self.data), high=max(self.data))
            self.map_log = btr.log_cmap(self.title, self._palette_list(1000), low=max(min(self.data),0.01), high=max(self.data))

    def _palette_list(self, n):
        return [(int(r), int(g), int(b), a) for r, g, b, a in (plt.get_cmap(self.palette)(np.linspace(0, 1, n)) * [255, 255, 255, 1])]

class ColorTooltipData(ColorMap, TooltipData):
    def __init__(self, title, data, palette='gnuplot2', categorical=False, high_transparent=False, show_in_table=True):
        ColorMap.__init__(self, title, data, palette=palette, categorical=categorical, high_transparent=high_transparent, show_in_table=show_in_table)
        TooltipData.__init__(self, title, data, show_in_table=show_in_table)

class PlotData1D(PlotData):
    def __init__(self, title, data, xlabels=None):
        super().__init__(title, data, show_in_table=False)
        self.xlabels = xlabels if xlabels is not None else [str(i) for i in range(len(data))]
    
class EmbeddingView(PlotData1D):
    def __init__(self, title, data, reducer, xlabels=None):
        super().__init__(title, data, xlabels=xlabels)       
        self.reducer = reducer
        self.embedding = self.reducer.fit_transform(data)

    def __getitem__(self, key):
        return self.__dict__[key]


class TooltipGraph(PlotData1D, TooltipData):
    def __init__(self, *data, title='TooltipGraph'):
        """
        data: a list of time series or similar, need not be the same length
        """
        _graphs = []
        for y_data in zip_longest(*data):
            img = BytesIO()
            fig = plt.figure(figsize=(3, 1))
            for p, color in zip(y_data, cycle(['gray', 'blue', 'black', 'green', 'purple', 'orange', 'brown', 'pink', 'red', 'olive'])):
                plt.plot(np.trim_zeros(p), color=color, linewidth=1, drawstyle="steps-post")
            plt.axis('off')
            plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            _graphs.append('data:image/png;base64,' + base64.b64encode(img.getvalue()).decode())
        super().__init__(f'{title}_graph', _graphs)
        self.html = f"<div><img src='@{title}_graph' style='margin: 5px 5px 5px 5px'/></div>"

class BokehInterface:
    def __init__(self, views, plot_elements):
        self.views = views
        self.plot_elements = plot_elements
        self.PLOT_DF = pd.DataFrame()
        for view in self.views:
            self.PLOT_DF[[view.title + '_x', view.title + '_y']] = view.embedding
        for element in self.plot_elements:
            self.PLOT_DF[element.title] = element.data
        self.tooltip_HTML = f"""
            <div>
                {''.join([element.html for element in self.plot_elements if isinstance(element, TooltipData)])}
            </div>
            <hr style="margin: 5px 5px 5px 5px"/><br>
        """
    
    def init_plot_elements(self):
        self.data_source = ColumnDataSource(self.PLOT_DF)
        self.embeddingSelect = Select(
            title="Embedding",
            value=self.views[0].title,
            options=[view.title for view in self.views]
        )
        self.coloringSelect = Select(
            title="Color by",
            value=[element.title for element in self.plot_elements if isinstance(element, ColorMap)][0],
            options=[element.title for element in self.plot_elements if isinstance(element, ColorMap)]
        )
        self.logCheck = Checkbox(
            label="Log color scale",
            active=False
        )
        self.fig = bk.figure(
            height=500,
            width=500,
            title=self.views[0].reducer.__repr__(),
            tools=(CopyTool(), LassoSelectTool(), HoverTool(tooltips=self.tooltip_HTML), BoxZoomTool(), ResetTool()),
        )
        self.plotPoints = self.fig.scatter(
            x=f'{self.views[0].title}_x',
            y=f'{self.views[0].title}_y',
            source=self.data_source,
            size=11,
            line_color='black',
            line_width=0.25,
            line_alpha=1,
            fill_alpha=0.7,
            fill_color=[element.map for element in self.plot_elements if isinstance(element, ColorMap)][0]
        )
        self.colorbar = ColorBar(
            color_mapper=[element.map for element in self.plot_elements if isinstance(element, ColorMap)][0].transform,
            padding=1
        )
        self.fig.add_layout(self.colorbar, 'right')
        self.embeddingSelect.js_on_change('value', CustomJS(
            args=dict(plotPoints=self.plotPoints, fig=self.fig, EMBEDDING_TITLES={view.title: view.reducer.__repr__() for view in self.views}),
            code="""
            const selectedValue = cb_obj.value;
            const x = selectedValue + '_x';
            const y = selectedValue + '_y';

            // Update the x and y attributes of the plotPoints glyph
            plotPoints.glyph.x = { field: x };
            plotPoints.glyph.y = { field: y };

            // Update the title
            fig.title.text = EMBEDDING_TITLES[selectedValue];

            // Trigger the glyph change event
            plotPoints.data_source.change.emit();
            """))
        self.color_maps = {attr.title: attr.map for attr in self.plot_elements if isinstance(attr, ColorMap)}
        self.color_maps.update({attr.title + '_log': attr.map_log for attr in self.plot_elements if isinstance(attr, ColorMap)})
        self.coloringSelect.js_on_change('value', CustomJS(
            args=dict(plotPoints=self.plotPoints, color_maps=self.color_maps, coloringSelect=self.coloringSelect, logCheck=self.logCheck),
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
            args=dict(plotPoints=self.plotPoints, color_maps=self.color_maps, colorbar=self.colorbar, coloringSelect=self.coloringSelect),
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

        table_cols = [TableColumn(field=col, title=col) for col in self.PLOT_DF.columns if not col.endswith('_x') and not col.endswith('_y') and not col.endswith('_graph')]

        stylesheet = ".slick-cell.selected { background-color: #ffff00!important; }"
        self.table = DataTable(
            source=self.data_source,
            columns=table_cols,
            resizable=True,
            stylesheets=[stylesheet]
        )

        self.selected_points_text = TextAreaInput(rows=5, value="")

        self.data_source.selected.js_on_change('indices', CustomJS(
            args=dict(source=self.data_source, selected_points_text=self.selected_points_text),
            code="""
            const indices = source.selected.indices;
            const tnsNames = source.data.tns_name;
            const selectedPoints = indices.map(index => tnsNames[index]);
            selected_points_text.value = "'" + selectedPoints.join("', \\n'") + "'";
        """))
        
    def show(self):
        self.fig.grid.visible = False
        self.fig.axis.visible = False
        plot = bkl.gridplot([[bkl.column(self.embeddingSelect, self.fig), bkl.column(self.coloringSelect, self.logCheck, self.table, self.selected_points_text)]],toolbar_location='below')
        bk.output_notebook()
        bk.show(plot)
        bk.output_file(title="Bokeh Plot", filename='bokeh_saves/test.html')
