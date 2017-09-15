# Import common plotly modules
import plotly as plty
import plotly.graph_objs as go
import cufflinks as cf

# Before using this, the following changes should be made to the offline.py file in the environment's plotly installation:
# * To get to plotly offline install, eg /Users/eczech/anaconda/envs/research3.5/lib/python3.5/site-packages/plotly/offline
# - Add the following lines to offline.py in _plot_html after the line "config = {}":
#   config['modeBarButtonsToRemove'] = ['sendDataToCloud'] # Added manually to remove save icon
#   config['displaylogo'] = False # Added manually to remove plotly icon
#
# After doing this, the extra cleaning logic in "plotly_clean_html" is no longer necessary but it is still left
# here because it is harmless otherwise and at least strips save icons from serialized plots if the manual edits
# above have not been made.

# Initialize Jupyter notebook mode
cf.set_config_file(offline=True, theme='white', offline_link_text=None, offline_show_link=False)

def plotly_clean_html(plotly_filename, auto_open=True):
    """ Strips Save icon from plot.ly html visualizations """

    # Read in source html for viz
    with open(plotly_filename, 'r') as of:
        html = of.read()

    # Replace the target strings (removes "Save" icon)
    html = html.replace('displaylogo:!0', 'displaylogo:!1')
    html = html.replace('modeBarButtonsToRemove:[]', 'modeBarButtonsToRemove:["sendDataToCloud"]')

    # Re-write source html
    with open(plotly_filename, 'w') as of:
        of.write(html)

    # Return original file url
    return plotly_filename

def plotly_offline_plot(figure_or_data, filename='temp-plot.html', show_link=False,
                        auto_open=False, link_text='', **kwargs):
    """ Overload function for offline plotting to automatically strip out links to external services """
    plty.offline.plot(figure_or_data, filename=filename, show_link=show_link,
                             link_text=link_text, auto_open=False, **kwargs)
    filename = plotly_clean_html(filename)
    if auto_open:
        import webbrowser
        return webbrowser.open('file://' + filename)
    else:
        return filename
        

def plotly_offline_iframe(figure, filename, width=None, height=None, display_link=True, return_iframe=True, **kwargs):
    """
    Render plotly html as a file and return IFrame viewer
    
    :param fig: Plot.ly figure
    :param filename: Path to html file to contain figure
    :param width: Width of IFrame; if None will be set based on figure width
    :param height: Height of IFrame; if None will be set based on figure height
    :param display_link: Flag indicating that a link to the generated file should be shown
    :param return_iframe: Flag indicating whether IFrame should be returned; disabling this means only the path will be shown
        (assuming display_path=True, otherwise an html file will be generated and nothing else)
    """
    
    width = width if width is not None else figure['layout'].get('width', 800)
    height = height if height is not None else figure['layout'].get('height', 400)
    plty.offline.plot(figure, filename=filename, show_link=False,
                             link_text='', auto_open=False, **kwargs)
    filename = plotly_clean_html(filename)
    if display_link:
        from IPython.display import FileLink, display
        display(FileLink(filename, result_html_prefix='<strong>Figure Link</strong>: '))
    if return_iframe:
        from IPython.display import IFrame
        return IFrame(filename, width=width, height=height)
    

def plotly_offline_iplot(figure_or_data, show_link=False, link_text='', **kwargs):
    """ Overload function for inline, offline plotting to strip out links to external services by default"""
    config = {'modeBarButtonsToRemove': ['sendDataToCloud'], 'displaylogo': False }
    if 'config' in kwargs:
        config.update(kwargs.pop('config'))
    return plty.offline.iplot(figure_or_data, show_link=show_link, link_text=link_text, config=config, **kwargs)

plty.offline.plt = plotly_offline_plot
plty.offline.iplt = plotly_offline_iplot
plty.offline.ifrm = plotly_offline_iframe
