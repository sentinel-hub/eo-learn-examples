import math
from eolearn.core import   FeatureType, EOTask
import numpy as np



class VegetationIndicesVHRS(EOTask):
    def __init__(self, feature_name) :
        self.feature_name = feature_name

    def calcul_ratio_vegetation_indices(self):
        self.NDVI = (self.B4 - self.B3)/(self.B4 + self.B3)
        self.NDWI = (self.B2 - self.B4)/(self.B2 + self.B4)
        #self.MSAVI2 = (2*self.B4 + 1 - ((2*self.B4 +1)^2)**0.5 - 8*(self.B4 - self.B3))/2
        self.VARI = (self.B2 - self.B3)/(self.B2 + self.B3 - self.B1)

    def execute(self, eopatch, **kwargs):
        arr0 = eopatch.data[self.feature_name]

        # Raw data
        self.B1 = arr0[..., 0]
        self.B2 = arr0[..., 1]
        self.B3 = arr0[..., 2]
        self.B4 = arr0[..., 3]
        #VIS
        self.calcul_ratio_vegetation_indices()
        eopatch.add_feature(FeatureType.DATA, "NDVI", self.NDVI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "NDWI", self.NDWI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "VARI", self.VARI[..., np.newaxis])

        return eopatch



class BiophysicalIndices:
    def __init__(self,B03, B04,B05, B06, B07,B8A,B11, B12, viewZenithMean, sunZenithAngles, viewAzimuthMean,sunAzimuthAngles):
        '''EOPatch should contains only 10 and 20m bands + illumination properties, as in eocrops.input.sentinel2'''
        self.B03 = B03
        self.B04 = B04
        self.B05 = B05
        self.B06 = B06
        self.B07 = B07
        self.B8A = B8A
        self.B11 = B11
        self.B12 = B12
        self.viewZenithMean = viewZenithMean
        self.sunZenithAngles = sunZenithAngles
        self.viewAzimuthMean = viewAzimuthMean
        self.sunAzimuthAngles = sunAzimuthAngles

    @staticmethod
    def _normalize(unnormalized, mini, maxi) :
        '''Normalize input Neural Network'''
        return 2*(unnormalized-mini)/(maxi-mini)-1

    @staticmethod
    def _denormalize(normalized, mini, maxi) :
        '''Denormalize output Neural Network'''
        return 0.5*(normalized+1)*(maxi-mini)+mini

    @staticmethod
    def _funccos(t):
        return np.vectorize(lambda t : math.cos(t))

    @staticmethod
    def _neuron(cste, w_b03, w_b04, w_b05, w_b06, w_b07, w_b8A, w_b11, w_b12, w_viewZen_norm, w_sunZen_norm,
               w_relAzim_norm, b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
               viewZen_norm, sunZen_norm, relAzim_norm) :

        def tansig(input_neuron) :
            '''Activation function neural network'''
            funcexp = np.vectorize(lambda t : math.exp(t))
            return 2/(1+funcexp(-2*input_neuron))-1

        '''Define the neuron, which is a linear combination of the metadata of sentinel2 images'''

        var_sum = cste+w_b03*b03_norm+w_b04*b04_norm+w_b05*b05_norm+w_b06*b06_norm
        var_sum += w_b07*b07_norm+w_b8A*b8a_norm+w_b11*b11_norm+w_b12*b12_norm
        var_sum += w_viewZen_norm*viewZen_norm+w_sunZen_norm*sunZen_norm+w_relAzim_norm*relAzim_norm

        return tansig(var_sum)

    @staticmethod
    def _layer2(cste, w_n1, neuron1, w_n2, neuron2, w_n3, neuron3, w_n4, neuron4, w_n5, neuron5) :
        '''Define the second layer, which is a linear combination of the neurons => input of the activation function'''
        return cste+w_n1*neuron1+w_n2*neuron2+w_n3*neuron3+w_n4*neuron4+w_n5*neuron5

    def apply_normalize(self) :
        self.b03_norm = self._normalize(self.B03, 0, 0.253061520471542)
        self.b04_norm = self._normalize(self.B04, 0, 0.290393577911328)
        self.b05_norm = self._normalize(self.B05, 0, 0.305398915248555)
        self.b06_norm = self._normalize(self.B06, 0.006637972542253, 0.608900395797889)
        self.b07_norm = self._normalize(self.B07, 0.013972727018939, 0.753827384322927)
        self.b8a_norm = self._normalize(self.B8A, 0.026690138082061, 0.782011770669178)
        self.b11_norm = self._normalize(self.B11, 0.016388074192258, 0.493761397883092)
        self.b12_norm = self._normalize(self.B12, 0, 0.493025984460231)

        degToRad = math.pi/180
        funccos = np.vectorize(lambda t : math.cos(t))

        self.viewZen_norm = self._normalize(funccos(self.viewZenithMean*degToRad), 0.918595400582046, 1)
        self.sunZen_norm = self._normalize(funccos(self.sunZenithAngles*degToRad), 0.342022871159208, 0.936206429175402)
        self.relAzim_norm = funccos((self.sunAzimuthAngles-self.viewAzimuthMean)*degToRad)


    def get_LAI(self) :
        '''Define biophysical vegetation index Leaf Area Index, computed from a trained neural network which has as input the metadata of sentinel2 images'''

        n1 = self._neuron(4.96238030555279, - 0.023406878966470, + 0.921655164636366,
                    0.135576544080099, - 1.938331472397950, - 3.342495816122680,
                    0.902277648009576, 0.205363538258614, - 0.040607844721716,
                    -0.083196409727092, 0.260029270773809, 0.284761567218845,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n2 = self._neuron(1.416008443981500, - 0.132555480856684, - 0.139574837333540,
                    - 1.014606016898920, - 1.330890038649270, 0.031730624503341,
                    - 1.433583541317050, - 0.959637898574699, + 1.133115706551000,
                    0.216603876541632, 0.410652303762839, 0.064760155543506,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n3 = self._neuron(1.075897047213310, 0.086015977724868, 0.616648776881434,
                    0.678003876446556, 0.141102398644968, - 0.096682206883546,
                    - 1.128832638862200, 0.302189102741375, 0.434494937299725,
                    - 0.021903699490589, - 0.228492476802263, - 0.039460537589826,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n4 = self._neuron(1.533988264655420, - 0.109366593670404, - 0.071046262972729,
                    + 0.064582411478320, 2.906325236823160, - 0.673873108979163,
                    - 3.838051868280840, 1.695979344531530, 0.046950296081713,
                    - 0.049709652688365, 0.021829545430994, 0.057483827104091,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n5 = self._neuron(3.024115930757230, - 0.089939416159969, 0.175395483106147,
                    - 0.081847329172620, 2.219895367487790, 1.713873975136850,
                    0.713069186099534, 0.138970813499201, - 0.060771761518025,
                    0.124263341255473, 0.210086140404351, - 0.183878138700341,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        l2 = self._layer2(1.096963107077220, -1.500135489728730, n1, -0.096283269121503, n2, -  0.194935930577094, n3,
                    - 0.352305895755591, n4, 0.075107415847473, n5)

        return self._denormalize(l2, 0.000319182538301, 14.4675094548151)

    def get_Cab(self) :
        '''Define biochemical vegetation index Chloro a+b, computed from a trained neural network which has as input the metadata of sentinel2 images'''

        n1 = self._neuron(4.242299670155190, 0.400396555256580, 0.607936279259404,
                    0.137468650780226, - 2.955866573461640, - 3.186746687729570,
                    2.206800751246430, - 0.313784336139636, + 0.256063547510639,
                    -0.071613219805105, 0.510113504210111, 0.142813982138661,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n2 = self._neuron(- 0.259569088225796, - 0.250781102414872, 0.439086302920381,
                    - 1.160590937522300, - 1.861935250269610, 0.981359868451638,
                    1.634230834254840, - 0.872527934645577, 0.448240475035072,
                    0.037078083501217, 0.030044189670404, 0.005956686619403,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n3 = self._neuron(3.130392627338360, 0.552080132568747, - 0.502919673166901,
                    6.105041924966230, - 1.294386119140800, - 1.059956388352800,
                    - 1.394092902418820, 0.324752732710706, - 1.758871822827680,
                    - 0.036663679860328, - 0.183105291400739, - 0.038145312117381,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n4 = self._neuron(0.774423577181620, 0.211591184882422, - 0.248788896074327,
                    0.887151598039092, 1.143675895571410, - 0.753968830338323,
                    - 1.185456953076760, 0.541897860471577, - 0.252685834607768,
                    - 0.023414901078143, - 0.046022503549557, - 0.006570284080657,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n5 = self._neuron(2.584276648534610, 0.254790234231378, - 0.724968611431065,
                    0.731872806026834, 2.303453821021270, - 0.849907966921912,
                    - 6.425315500537270, 2.238844558459030, - 0.199937574297990,
                    0.097303331714567, 0.334528254938326, 0.113075306591838,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        l2 = self._layer2(0.463426463933822, - 0.352760040599190, n1, - 0.603407399151276, n2, 0.135099379384275, n3,
                    - 1.735673123851930, n4, - 0.147546813318256, n5)

        return self._denormalize(l2, 0.007426692959872, 873.908222110306)/10

    def get_FAPAR(self) :
        '''Define biophysical vegetation index Fraction of Absorbed Photosynthetically Active Radiation, computed from a trained neural network which has as input the metadata of sentinel2 images'''
        n1 = self._neuron(- 0.887068364040280, 0.268714454733421, - 0.205473108029835,
                    0.281765694196018, 1.337443412255980, 0.390319212938497,
                    - 3.612714342203350, 0.222530960987244, 0.821790549667255,
                    - 0.093664567310731, 0.019290146147447, 0.037364446377188,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n2 = self._neuron(0.320126471197199, - 0.248998054599707, - 0.571461305473124,
                    - 0.369957603466673, 0.246031694650909, 0.332536215252841,
                    0.438269896208887, 0.819000551890450, - 0.934931499059310,
                    0.082716247651866, - 0.286978634108328, - 0.035890968351662,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n3 = self._neuron(0.610523702500117, - 0.164063575315880, - 0.126303285737763,
                    - 0.253670784366822, - 0.321162835049381, 0.067082287973580,
                    2.029832288655260, - 0.023141228827722, - 0.553176625657559,
                    0.059285451897783, - 0.034334454541432, - 0.031776704097009,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n4 = self._neuron(- 0.379156190833946, 0.130240753003835, 0.236781035723321,
                    0.131811664093253, - 0.250181799267664, - 0.011364149953286,
                    - 1.857573214633520, - 0.146860751013916, 0.528008831372352,
                    - 0.046230769098303, - 0.034509608392235, 0.031884395036004,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        n5 = self._neuron(1.353023396690570, - 0.029929946166941, 0.795804414040809,
                    0.348025317624568, 0.943567007518504, - 0.276341670431501,
                    - 2.946594180142590, 0.289483073507500, 1.044006950440180,
                    - 0.000413031960419, 0.403331114840215, 0.068427130526696,
                    self.b03_norm, self.b04_norm, self.b05_norm, self.b06_norm,
                    self.b07_norm, self.b8a_norm, self.b11_norm, self.b12_norm,
                    self.viewZen_norm, self.sunZen_norm, self.relAzim_norm)

        l2 = self._layer2(- 0.336431283973339, 2.126038811064490, n1, - 0.632044932794919, n2, 5.598995787206250, n3,
                    1.770444140578970, n4, - 0.267879583604849, n5)

        return self._denormalize(l2, 0.000153013463222, 0.977135096979553)




class VegetationIndicesS2(EOTask) :
    '''Define a class of vegetation indices, which are computed from the metadata of sentinel2 images extracted'''

    def __init__(self, feature_name, mask_data=True) :
        self.feature_name = feature_name
        self.mask_data = mask_data

    def get_vegetation_indices(self) :
        '''Define vegetation indices which are simply ratio of spectral bands'''
        self.NDVI = (self.B8A-self.B04)/(self.B8A+self.B04)
        self.NDWI = (self.B8A-self.B11)/(self.B8A+self.B11)
        self.GNDVI = (self.B8A-self.B03)/(self.B8A+self.B03)

        biopysicial_parameters = BiophysicalIndices(self.B03, self.B04, self.B05, self.B06, self.B07, self.B8A, self.B11, self.B12,
                                                    self.viewZenithMean, self.sunZenithAngles, self.viewAzimuthMean, self.sunAzimuthAngles)
        # Normalized bands
        biopysicial_parameters.apply_normalize()

        self.fapar = biopysicial_parameters.get_FAPAR()
        self.LAI = biopysicial_parameters.get_LAI()
        self.Cab = biopysicial_parameters.get_Cab()


    def execute(self, eopatch) :
        '''Add those vegeation indices to the eo-patch for futur use'''

        bands_array = eopatch.data[self.feature_name]
        illumination_array = eopatch.data['ILLUMINATION']

        valid_data_mask = eopatch.mask['VALID_DATA'] if self.mask_data else eopatch.mask['IS_DATA']

        if 'polygon_mask' in list(eopatch.mask_timeless.keys()) :

            bands_array = np.ma.array(bands_array,
                                      dtype=np.float32,
                                      mask=np.logical_or(~valid_data_mask.astype(np.bool), np.isnan(bands_array)),
                                      fill_value=np.nan)
            bands_array = bands_array.filled()

        # Raw data
        self.B02 = bands_array[..., 0]
        self.B03 = bands_array[..., 1]
        self.B04 = bands_array[..., 2]
        self.B05 = bands_array[..., 3]
        self.B06 = bands_array[..., 4]
        self.B07 = bands_array[..., 5]
        self.B08 = bands_array[..., 6]
        self.B8A = bands_array[..., 7]
        self.B11 = bands_array[..., 8]
        self.B12 = bands_array[..., 9]
        self.viewZenithMean = illumination_array[..., 0]
        self.sunZenithAngles = illumination_array[..., 1]
        self.viewAzimuthMean = illumination_array[..., 2]
        self.sunAzimuthAngles = illumination_array[..., 3]

        self.get_vegetation_indices()
        eopatch.add_feature(FeatureType.DATA, "fapar", self.fapar[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "LAI", self.LAI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "Cab", self.Cab[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "NDVI", self.NDVI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "NDWI", self.EVI2[..., np.newaxis])
        eopatch.remove_feature(FeatureType.DATA, "ILLUMINATION")

        return eopatch




class EuclideanNorm(EOTask) :
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """

    def __init__(self, feature_name, in_feature_name) :
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name

    def execute(self, eopatch) :
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr**2, axis=-1))

        eopatch.add_feature(FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch




