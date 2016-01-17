# Import common plotly modules
from plotly import offline as ptly
import plotly.graph_objs as go
import cufflinks as cf

# Initialize Jupyter notebook mode
ptly.init_notebook_mode()
cf.set_config_file(offline=True, theme='ggplot', offline_link_text=None, offline_show_link=False)
