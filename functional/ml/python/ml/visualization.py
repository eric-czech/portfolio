
from plotly import offline
import webbrowser


def clean_plotly_viz(plotly_filename, auto_open=True):

    with open(plotly_filename, 'r') as of:
        html = of.read()

    # Replace the target strings
    html = html.replace('displaylogo:!0', 'displaylogo:!1')
    html = html.replace('modeBarButtonsToRemove:[]', 'modeBarButtonsToRemove:["sendDataToCloud"]')
    with open(plotly_filename, 'w') as of:
        of.write(html)

    del html
    if auto_open:
        print('Opening Plotly visualization at path: {}'.format(plotly_filename))
        return webbrowser.open('file://' + plotly_filename)
    return plotly_filename


def render_plotly_viz(data, filename='/tmp/temp-viz.html', show_link=False, auto_open=True, **kwargs):
    offline.plot(
        data, filename=filename,
        show_link=False, auto_open=False,
        **kwargs
    )

    return clean_plotly_viz(filename, auto_open=auto_open)
