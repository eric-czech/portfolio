# Collection of utility functions for stripping annoying links from Plot.ly visualizations

import plotly as plty


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


def plot(figure_or_data, filename='temp-plot.html', show_link=False,
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


def iplot(figure_or_data, show_link=False, link_text='', **kwargs):
    """ Overload function for inline, offline plotting to strip out links to external services by default"""
    return plty.offline.iplot(figure_or_data, show_link=show_link, link_text=link_text, **kwargs)