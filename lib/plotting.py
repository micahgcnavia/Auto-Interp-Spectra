import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource, RangeSlider
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.io import output_notebook
from lib.get_config import *

def show_spectra(interp_steps, width=1400, height=600, line_width=1.5, x_start=6550, x_end=6600, start=6200, end=6800):

    """
        Creates an interactive plots displaying all interpolated spectra with Bokeh.

        :param interp_steps: List of DataFrames containing updated parameters and fluxes at each interpolation step.
        :type interp_steps: list[pandas.DataFrame]
        :param width: Width of the figure. Default = 1400.
        :type width: int
        :param height:  Height of the figure. Default = 600.
        :type height: int
        :param line_width: Line width. Default = 1.5.
        :type line_width: float
        :param x_start: Start wavelength in Ansgstroms (Å). Default = 6550 Å.
        :type x_start: int
        :param x_end: End wavelength in Ansgstroms (Å). Default = 6600 Å.
        :type x_end: int
        :param start: Lower limit of the wavelength in Angstroms (Å) to display on the plot. Default = 6200 Å.
        :type start: int
        :param end: Upper limit of the wavelength in Angstroms (Å) to display on the plot. Default = 6800 Å.
        :type end: int

    """
    # Getting user data
    config = get_config()

    wav_ref_path = config['USER_DATA']['reference_spectrum']
    wav_ref, _ = np.loadtxt(wav_ref_path, unpack=True)

    # Initialize Bokeh in notebook
    output_notebook()

    # Create ColumnDataSources for each spectrum
    sources = {'Teff = {} K, log g = {} dex, [Fe/H] = {} dex'.format(col[0], col[1], col[2]): ColumnDataSource(data={'wav': wav_ref, 'flux': col[3]})\
                for i in range(len(interp_steps)) for _, col in interp_steps[i].iterrows()}

    # Create figure
    p = figure(width=width, height=height, tools="pan,wheel_zoom,box_zoom,reset",
            title="Multi-Spectra Viewer", x_axis_label='Wavelength (Å)', 
            y_axis_label='Flux (erg/cm²/s/Å)')

    # Plot all spectra initially (but we'll control visibility)
    renderers = {}
    colors = Category10[len(sources)]

    for i, (name, source) in enumerate(sources.items()):
        renderers[name] = p.line('wav', 'flux', source=source, line_width=line_width,
                                color=colors[i], alpha=0.8, legend_label=name,
                                muted_color=colors[i], muted_alpha=0.1)

    # Configure visual elements

    p.x_range.start = x_start
    p.x_range.end = x_end
    p.xgrid.grid_line_color = 'lightgray'
    p.ygrid.grid_line_color = 'lightgray'
    p.legend.location = "top_right"
    p.legend.click_policy = "mute"  # Click on legend to mute/unmute

    # Add range slider for wavelength control
    range_slider = RangeSlider(start=start, end=end, value=(x_start, x_end), step=50,
                            title="Wavelength Range (Å)", width=600)

    range_callback = CustomJS(args=dict(x_range=p.x_range), code="""
        x_range.start = cb_obj.value[0];
        x_range.end = cb_obj.value[1];
    """)

    range_slider.js_on_change('value', range_callback)

    # Combine all elements
    controls = row(range_slider)
    layout = column(controls, p)

    show(layout)