import folium
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon as plt_polygon
import numpy as np
from aenum import MultiValueEnum


from IPython.core.display import display, HTML


def evalscript_template(bands, sample_type):
    evaluate_pixel = ", ".join(f"sample.{band}" for band in bands)
    return f"""
    //VERSION=3
    function setup() {{
      return {{
        input: {bands},
        output: [
            {{
              id: "bands",
              bands: {len(bands)},
              sampleType: "{sample_type}"
            }}
        ]
      }};
    }}
    function evaluatePixel(sample) {{
      return [{evaluate_pixel}];
    }}
    """


def plot_tiff_folium(data, bounds, variable_name):
    ymin, xmin, ymax, xmax = list(bounds)
    m = folium.Map(location=[(xmin+xmax)/2, (ymin+ymax)/2], zoom_start=5, control_scale=True)
    folium.raster_layers.ImageOverlay(
        #name=month,
        image=data,
        bounds=[[xmin, ymin], [xmax, ymax]],
        opacity=0.9,
        mercator_project=True,
        colormap=plt.cm.Reds if variable_name == 'TEMPERATURE' else plt.cm.Blues,
        control=True
        ).add_to(m)
    return m

def plot_multiple_foliums(maps, size_x, size_y):
    template = '<iframe srcdoc="{}" style="float:left; width: {}px; height: {}px; display:inline-block; width: 50%; margin: 0 auto; border: 2px solid black"></iframe>'
    html_map = ''
    for i, m in enumerate(maps):
        html_map = html_map + template.format(m.get_root().render().replace('"', '&quot;'),size_x,size_y)
    return HTML(html_map)


class EWC(MultiValueEnum):
    """Enum class containing basic LULC EWC types"""

    NO_DATA = "No data", 0, "black"
    TREE_COVER = "Tree cover", 10, "darkgreen"
    SHRUBLAND = "Shrubland", 20, "orange"
    GRASSLAND = "Grassland", 30, "yellow"
    CROPLAND = "Cropland", 40, "violet"
    BUILT_UP = "Built up", 50, "red"
    BARE_SPARSE_VEGETATION = "Bare /sparse vegetation", 60, "dimgrey"
    SNOW_ICE = "Snow and Ice", 70, "silver"
    PERMANENT_WATER_BODIES = "Permanent water bodies", 80, "blue"
    HERBACEOUS_WETLAND = "Herbaceous wetland", 90, "darkcyan"
    MANGROVES = "Mangroves", 95, "springgreen"
    MOSS_LICHEN = "Moss and lichen", 100, "khaki"

def construct_cmap(colors, data, name="cmap"):
    sub = []
    un = np.unique(data)
    for i, entry in enumerate(colors):
        if np.isin(entry.values[1], un):
            sub.append([entry.values[0], entry.values[1], entry.values[2]])
    bounds = [sub[0][1] - 0.5]
    [bounds.append((0.5 + entry[1])) for entry in sub]
    ticks = [np.mean([bounds[i], bounds[i - 1]]) for i in range(1, len(bounds))]
    cmap = ListedColormap([rgb_int(entry) for entry in sub], name=name)
    norm = BoundaryNorm(bounds, cmap.N)
    labels = [entry[0] for entry in sub]

    return cmap, norm, ticks, labels

def rgb_int(row):
    hex_val = row[2]
    rgb_val = matplotlib.colors.to_rgb(hex_val)
    return rgb_val