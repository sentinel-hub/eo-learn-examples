import matplotlib
import numpy as np
from aenum import MultiValueEnum
from matplotlib.colors import BoundaryNorm, ListedColormap


def rgb_int(row):
    hex_val = row[2]
    rgb_val = matplotlib.colors.to_rgb(hex_val)
    return rgb_val


def construct_cmap(colors, data, name="cmap"):
    sub = []
    un = np.unique(data)
    for entry in colors:
        if np.isin(entry.values[1], un):
            sub.append([entry.values[0], entry.values[1], entry.values[2]])
    bounds = [sub[0][1] - 0.5]
    [bounds.append((0.5 + entry[1])) for entry in sub]
    ticks = [np.mean([bounds[i], bounds[i - 1]]) for i in range(1, len(bounds))]
    cmap = ListedColormap([rgb_int(entry) for entry in sub], name=name)
    norm = BoundaryNorm(bounds, cmap.N)
    labels = [entry[0] for entry in sub]

    return cmap, norm, ticks, labels


class CLC_ACC(MultiValueEnum):
    """Enum class containing CORINE Land Cover types for accounting layers values"""

    CONTINUOUS_URBAN_FABRIC = "Continuous urban fabric", 111, "#E6004D"
    DISCONTINUOUS_URBAN_FABRIC = "Discontinuous urban fabric", 112, "#FF0000"
    INDUSTRIAL_OR_COMMERCIAL_UNITS = "Industrial or commercial units", 121, "#CC4DF2"
    ROAD_AND_RAIL_NETWORKS_AND_ASSOCIATED_LAND = "Road and rail networks and associated land", 122, "#CC0000"
    PORT_AREAS = "Port areas", 123, "#E6CCCC"
    AIRPORTS = "Airports", 124, "#E6CCE6"
    MINERAL_EXTRACTION_SITES = "Mineral extraction sites", 131, "#A600CC"
    DUMP_SITES = "Dump sites", 132, "#A64D00"
    CONSTRUCTION_SITES = "Construction sites", 133, "#FF4DFF"
    GREEN_URBAN_AREAS = "Green urban areas", 141, "#FFA6FF"
    SPORT_AND_LEISURE_FACILITIES = "Sport and leisure facilities", 142, "#FFE6FF"
    NON_IRRIGATED_ARABLE_LAND = "Non-irrigated arable land", 211, "#FFFFA8"
    PERMANENTLY_IRRIGATED_LAND = "Permanently irrigated land", 212, "#FFFF00"
    RICE_FIELDS = "Rice fields", 213, "#E6E600"
    VINEYARDS = "Vineyards", 221, "#E68000"
    FRUIT_TREES_AND_BERRY_PLANTATIONS = "Fruit trees and berry plantations", 222, "#F2A64D"
    OLIVE_GROVES = "Olive groves", 223, "#E6A600"
    PASTURES = "Pastures", 231, "#E6E64D"
    ANNUAL_CROPS_ASSOCIATED_WITH_PERMANENT_CROPS = "Annual crops associated with permanent crops", 241, "#FFE6A6"
    COMPLEX_CULTIVATION_PATTERNS = "Complex cultivation patterns", 242, "#FFE64D"
    LAND_PRINCIPALLY_OCCUPIED_BY_AGRICULTURE_WITH_SIGNIFICANT_AREAS_OF_NATURAL_VEGETATION = (
        "Land principally occupied by agriculture with significant areas of natural vegetation",
        243,
        "#E6CC4D",
    )
    AGRO_FORESTRY_AREAS = "Agro-forestry areas", 244, "#F2CCA6"
    BROAD_LEAVED_FOREST = "Broad-leaved forest", 311, "#80FF00"
    CONIFEROUS_FOREST = "Coniferous forest", 312, "#00A600"
    MIXED_FOREST = "Mixed forest", 313, "#4DFF00"
    NATURAL_GRASSLANDS = "Natural grasslands", 321, "#CCF24D"
    MOORS_AND_HEATHLAND = "Moors and heathland", 322, "#A6FF80"
    SCLEROPHYLLOUS_VEGETATION = "Sclerophyllous vegetation", 323, "#A6E64D"
    TRANSITIONAL_WOODLAND_SHRUB = "Transitional woodland-shrub", 324, "#A6F200"
    BEACHES__DUNES__SANDS = "Beaches - dunes - sands", 331, "#E6E6E6"
    BARE_ROCKS = "Bare rocks", 332, "#CCCCCC"
    SPARSELY_VEGETATED_AREAS = "Sparsely vegetated areas", 333, "#CCFFCC"
    BURNT_AREAS = "Burnt areas", 334, "#000000"
    GLACIERS_AND_PERPETUAL_SNOW = "Glaciers and perpetual snow", 335, "#A6E6CC"
    INLAND_MARSHES = "Inland marshes", 411, "#A6A6FF"
    PEAT_BOGS = "Peat bogs", 412, "#4D4DFF"
    SALT_MARSHES = "Salt marshes", 421, "#CCCCFF"
    SALINES = "Salines", 422, "#E6E6FF"
    INTERTIDAL_FLATS = "Intertidal flats", 423, "#A6A6E6"
    WATER_COURSES = "Water courses", 511, "#00CCF2"
    WATER_BODIES = "Water bodies", 512, "#80F2E6"
    COASTAL_LAGOONS = "Coastal lagoons", 521, "#00FFA6"
    ESTUARIES = "Estuaries", 522, "#A6FFE6"
    SEA_AND_OCEAN = "Sea and ocean", 523, "#E6F2FF"
    NODATA = "NODATA", 999, "#FFFFFF"


class CLC(MultiValueEnum):
    """Enum class containing CORINE Land Cover types"""

    CONTINUOUS_URBAN_FABRIC = "Continuous urban fabric", 1, "#E6004D"
    DISCONTINUOUS_URBAN_FABRIC = "Discontinuous urban fabric", 2, "#FF0000"
    INDUSTRIAL_OR_COMMERCIAL_UNITS = "Industrial or commercial units", 3, "#CC4DF2"
    ROAD_AND_RAIL_NETWORKS_AND_ASSOCIATED_LAND = "Road and rail networks and associated land", 4, "#CC0000"
    PORT_AREAS = "Port areas", 5, "#E6CCCC"
    AIRPORTS = "Airports", 6, "#E6CCE6"
    MINERAL_EXTRACTION_SITES = "Mineral extraction sites", 7, "#A600CC"
    DUMP_SITES = "Dump sites", 8, "#A64D00"
    CONSTRUCTION_SITES = "Construction sites", 9, "#FF4DFF"
    GREEN_URBAN_AREAS = "Green urban areas", 10, "#FFA6FF"
    SPORT_AND_LEISURE_FACILITIES = "Sport and leisure facilities", 11, "#FFE6FF"
    NON_IRRIGATED_ARABLE_LAND = "Non-irrigated arable land", 12, "#FFFFA8"
    PERMANENTLY_IRRIGATED_LAND = "Permanently irrigated land", 13, "#FFFF00"
    RICE_FIELDS = "Rice fields", 14, "#E6E600"
    VINEYARDS = "Vineyards", 15, "#E68000"
    FRUIT_TREES_AND_BERRY_PLANTATIONS = "Fruit trees and berry plantations", 16, "#F2A64D"
    OLIVE_GROVES = "Olive groves", 17, "#E6A600"
    PASTURES = "Pastures", 18, "#E6E64D"
    ANNUAL_CROPS_ASSOCIATED_WITH_PERMANENT_CROPS = "Annual crops associated with permanent crops", 19, "#FFE6A6"
    COMPLEX_CULTIVATION_PATTERNS = "Complex cultivation patterns", 20, "#FFE64D"
    LAND_PRINCIPALLY_OCCUPIED_BY_AGRICULTURE_WITH_SIGNIFICANT_AREAS_OF_NATURAL_VEGETATION = (
        "Land principally occupied by agriculture with significant areas of natural vegetation",
        21,
        "#E6CC4D",
    )
    AGRO_FORESTRY_AREAS = "Agro-forestry areas", 22, "#F2CCA6"
    BROAD_LEAVED_FOREST = "Broad-leaved forest", 23, "#80FF00"
    CONIFEROUS_FOREST = "Coniferous forest", 24, "#00A600"
    MIXED_FOREST = "Mixed forest", 25, "#4DFF00"
    NATURAL_GRASSLANDS = "Natural grasslands", 26, "#CCF24D"
    MOORS_AND_HEATHLAND = "Moors and heathland", 27, "#A6FF80"
    SCLEROPHYLLOUS_VEGETATION = "Sclerophyllous vegetation", 28, "#A6E64D"
    TRANSITIONAL_WOODLAND_SHRUB = "Transitional woodland-shrub", 29, "#A6F200"
    BEACHES__DUNES__SANDS = "Beaches - dunes - sands", 30, "#E6E6E6"
    BARE_ROCKS = "Bare rocks", 31, "#CCCCCC"
    SPARSELY_VEGETATED_AREAS = "Sparsely vegetated areas", 32, "#CCFFCC"
    BURNT_AREAS = "Burnt areas", 33, "#000000"
    GLACIERS_AND_PERPETUAL_SNOW = "Glaciers and perpetual snow", 34, "#A6E6CC"
    INLAND_MARSHES = "Inland marshes", 35, "#A6A6FF"
    PEAT_BOGS = "Peat bogs", 36, "#4D4DFF"
    SALT_MARSHES = "Salt marshes", 37, "#CCCCFF"
    SALINES = "Salines", 38, "#E6E6FF"
    INTERTIDAL_FLATS = "Intertidal flats", 39, "#A6A6E6"
    WATER_COURSES = "Water courses", 40, "#00CCF2"
    WATER_BODIES = "Water bodies", 41, "#80F2E6"
    COASTAL_LAGOONS = "Coastal lagoons", 42, "#00FFA6"
    ESTUARIES = "Estuaries", 43, "#A6FFE6"
    SEA_AND_OCEAN = "Sea and ocean", 44, "#E6F2FF"
    NODATA = "NODATA", 48, "#FFFFFF"


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


class GLC(MultiValueEnum):
    """Enum class containing basic LULC types"""

    NO_INPUT_DATA = "No input data available", 0, "#282828"
    SHRUBS = "Shrubs", 20, "#ffbb22"
    HERBCEOUS_VEGETATION = "Herbaceous vegetation", 30, "#ffff4c"
    CULTIVATED_AND_MANAGED_VEGETATION = "Cultivated and managed vegetation/agriculture (cropland)", 40, "#f096ff"
    URBAN_BUILT_UP = "Urban / built up", 50, "#fa0000"
    BARE = "Bare / sparse vegetation", 60, "#b4b4b4"
    SNOW_ICE = "Snow and Ice", 70, "#f0f0f0"
    PERMANENT_WATER_BODIES = "Permanent water bodies", 80, "#0032c8"
    HERBCEOUS_WETLAND = "Herbaceous wetland", 90, "#0096a0"
    MOSS_LICHEN = "Moss and lichen", 100, "#fae6a0"
    CF1 = "Closed forest, evergreen needle leaf", 111, "#58481f"
    CF2 = "Closed forest, evergreen, broad leaf", 112, "#009900"
    CF3 = "Closed forest, deciduous needle leaf", 113, "#70663e"
    CF4 = "Closed forest, deciduous broad leaf", 114, "#00cc00"
    CF5 = "Closed forest, mixed", 115, "#4e751f"
    CF6 = "Closed forest, unknown", 116, "#007800"
    OF1 = "Open forest, evergreen needle leaf", 121, "#666000"
    OF2 = "Open forest, evergreen broad leaf", 122, "#8db400"
    OF3 = "Open forest, deciduous needle leaf", 123, "#8d7400"
    OF4 = "Open forest, deciduous broad leaf", 124, "#a0dc00"
    OF5 = "Open forest, mixed", 125, "#929900"
    OF6 = "Open forest, unknown", 126, "#648c00"
    OPEN_SEA = "Open sea", 200, "#000080"
    NO_DATA = (
        "No data",
        255,
    )


class TLCM(MultiValueEnum):
    """Enum class containing basic LULC types"""

    E1 = "Dense built-up area", 1, "#ff00ff"
    E2 = "Diffuse built-up area", 2, "#ff55ff"
    E3 = "Industrial and commercial areas", 3, "#ffaaff"
    E4 = "Roads", 4, "#00ffff"
    E5 = "Oilseeds (Rapeseed)", 5, "#ffff00"
    E6 = "Straw cereals (Wheat, Triticale, Barley)", 6, "#d0ff00"
    E7 = "Protein crops (Beans / Peas)", 7, "#a1d600"
    E8 = "Soy", 8, "#ffab44"
    E9 = "Sunflower", 9, "#d6d600"
    E10 = "Corn", 10, "#ff5500"
    E11 = "Rice", 11, "#c5ffff"
    E12 = "Tubers/roots", 12, "#aaaa61"
    E13 = "Grasslands", 13, "#aaaa00"
    E14 = "Orchards and fruit growing", 14, "#aaaaff"
    E15 = "Vineyards", 15, "#550000"
    E16 = "Hardwood forest", 16, "#009c00"
    E17 = "Softwood forest", 17, "#003200"
    E18 = "Natural grasslands and pastures", 18, "#aaff00"
    E19 = "Woody moorlands", 19, "#55aa7f"
    E20 = "Natural mineral surfaces", 20, "#ff0000"
    E21 = "Beaches and dunes", 21, "#ffb802"
    E22 = "Glaciers and eternal snows", 22, "#bebebe"
    E23 = "Water", 23, "#0000ff"


class SCL(MultiValueEnum):
    """Enum class containing basic LULC types"""

    NO_DATA = "no data", 0, "#000000"
    SATURATED_DEFECTIVE = "saturated / defective", 1, "#ff0004"
    DARK_AREA_PIXELS = "dark area pixels", 2, "#868686"
    CLOUD_SHADOWS = "cloud shadows", 3, "#774c0b"
    VEGETATION = "vegetation", 4, "#10d32d"
    BARE_SOILS = "bare soils", 5, "#ffff53"
    WATER = "water", 6, "#0000ff"
    CLOUDS_LOW_PROBA = "clouds low proba.", 7, "#818181"
    CLOUDS_MEDIUM_PROBA = "clouds medium proba.", 8, "#c0c0c0"
    CLOUDS_HIGH_PROBA = "clouds high proba.", 9, "#f2f2f2"
    CIRRUS = "cirrus", 10, "#bbc5ec"
    SNOW_ICE = "snow / ice", 11, "#53fffa"

    @property
    def rgb(self):
        return [c / 255.0 for c in self.rgb_int]

    @property
    def rgb_int(self):
        hex_val = self.values[2][1:]
        return [int(hex_val[i : i + 2], 16) for i in (0, 2, 4)]
