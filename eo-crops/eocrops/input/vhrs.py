
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

from datetime import datetime
import json

import time


from sentinelhub import (
    DataCollection,
)

import eocrops.tasks.vegetation_indices as vegetation_indices

from eolearn.io import  SentinelHubEvalscriptTask

import datetime as dt
import geopandas as gpd
import eolearn
import os

import eocrops.utils.utils as utils
import eocrops.tasks.preprocessing as preprocessing
import eocrops.tasks.cmd_otb as cmd_otb
import eocrops.input.utils_sh as utils_sh
from eolearn.core import SaveTask, FeatureType, OverwritePermission,SaveTask



class DownloadVHRSSentinelHub :
    '''
    Initialize query parameters and ingest data into SH
    Parameters
    ----------
    shapefile (GeoDataFrame) : shapefile of your field to extract data
    time_stamp (tuple) : time period to extract data
    provider (tuple) : choice between ('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'),('PLANET', 'PSScene4Band')
    maxCloudCoverage : maximum cloud cover that may be present in one image

    -------
    '''

    def __init__(self, shapefile, time_stamp, config, maxCloudCoverage,
                 url="https://services.sentinel-hub.com/api/v1/dataimport/") :
        self.shapefile = shapefile
        self.time_stamp = time_stamp
        self.config = config
        self.maxCloudCoverage = maxCloudCoverage
        self.url = url
        self.request = False
        self.order = False

    def get_quotas(self) :
        '''Get quota from you SH account'''
        client = BackendApplicationClient(client_id=self.config.sh_client_id)
        oauth = OAuth2Session(client=client)
        response = oauth.get(url=self.url+"quotas")
        response.raise_for_status()

        return response.json()

    def _create_token(self) :
        '''
        Create a token for the HTTP API from Sentinelhub third party data import
        Parameters
        ----------
        url (str) : default url for 3rd party data collection

        Returns oauth and the fetch token
        -------

        '''
        client = BackendApplicationClient(client_id=self.config.sh_client_id)
        oauth = OAuth2Session(client=client)

        # Get token for the session
        token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                                  client_secret=self.config.sh_client_secret)

        # All requests using this session will have an access token automatically added
        resp = oauth.get("https://services.sentinel-hub.com/oauth/tokeninfo")
        print(resp.content)

        return oauth, token

    def _get_input_polygon(self) :
        '''
        Format the input polygon for the HTTP API request into json format
        Parameters
        ----------

        Returns JSON format of the polygon of the field with its corresponding UTM CRS
        -------

        '''
        shapefile_boundary = utils.check_crs(self.shapefile.copy())
        field_bbox = utils.get_bounding_box(shapefile_boundary)
        input_polygon_json = gpd.GeoSeries([field_bbox.geometry]).to_json()
        res = json.loads(input_polygon_json)
        input_polygon = res['features'][0]['geometry']['coordinates']
        return input_polygon, field_bbox, str(field_bbox.crs).split(':')[-1]

    def _init_param_query(self,
                          provider=('AIRBUS', 'PHR')) :
        '''
        Initialize query parameters
        Parameters
        ----------
        provider (tuple) : choice between ('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'),('PLANET', 'PSScene4Band')
        Returns dictionary with your query parameters
        -------

        '''

        global dict_query

        input_polygon, field_bbox, CRS = self._get_input_polygon()
        if provider not in [('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'), ('PLANET', 'PSScene4Band')] :
            raise ValueError('Available providers are only "PLANET" and "AIRBUS"')

        if provider[0]=='PLANET' :
            dict_query = dict(provider=provider[0],
                              axCloudCoverage=self.maxCloudCoverage,
                              data=[{'itemType' : provider[-1],
                                     "productBundle" : "analytic_sr_udm2",
                                     'dataFilter' : {'timeRange' : {'from' : '', 'to' : ''},
                                                     'self.maxCloudCoverage' : self.maxCloudCoverage,
                                                     'nativeFilter' : {'type' : 'StringInFilter',
                                                                       'field_name' : 'quality_category',
                                                                       'config' : ['standard']}}}])

        elif provider[0]=='AIRBUS' :
            dict_query = dict(provider=provider[0],
                              data=[{
                                  "constellation" : provider[1],
                                  "dataFilter" : {"timeRange" : {"from" : "",
                                                                 "to" : ""}}}],
                              CRS=CRS)

        dict_query['input_polygon'] = input_polygon
        dict_query['field_bbox'] = field_bbox
        dict_query['CRS'] = CRS

        try :
            date_object = datetime.strptime(self.time_stamp[0], "%Y-%m-%d")
        except :
            raise ValueError('You must provide a date into yyyy-mm-dd string format')

        dict_query['data'][0]['dataFilter']['timeRange']['from'] = self.time_stamp[0]+"T00:00:00.000Z"
        dict_query['data'][0]['dataFilter']['timeRange']['to'] = self.time_stamp[1]+"T00:00:00.000Z"

        return dict_query

    def _search_query(self, dict_query, oauth) :
        '''
        Search satellite images available with respect to your input query
        Parameters
        ----------
        dict_query (dictionary) : dictionary with the parameters to search the query
        oauth : oauth client
        Returns JSON with the results and the list of item ids available
        -------
        '''
        # Last error : need to specify as well product bundle in the search query
        query = {
            "provider" : dict_query['provider'],
            "bounds" : {
                "properties" : {
                    "crs" : "http://www.opengis.net/def/crs/EPSG/0/"+dict_query['CRS']
                },
                "geometry" : {
                    "type" : "Polygon",
                    "coordinates" : dict_query['input_polygon']
                }
            },
            "data" : dict_query['data']
        }

        response = oauth.post(self.url+"search", json=query)
        response.raise_for_status()

        results = response.json()
        # get product ids
        if dict_query['provider']=='PLANET' :
            item_ids = [feature["id"] for feature in results["features"]]
        elif dict_query['provider']=='AIRBUS' :
            item_ids = [feature["properties"]["id"] for feature in results["features"]]
        else :
            raise ValueError('Provider must be set to Planet or Airbus')

        return results, item_ids

    @staticmethod
    def _payload_data(item_ids, provider=('PLANET', 'PSScene4Band')) :
        '''
        Define the payload for your HTTP API
        Parameters
        ----------
        provider (tuple) : choice between ('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'),('PLANET', 'PSScene4Band')
        item_id (list) : list of the item ids

        Returns
        -------

        '''

        output = {}

        if provider[0]=='AIRBUS' :

            output["data"] = [{"constellation" : provider[1],
                               "products" : [{"id" : item_id} for item_id in item_ids]}]

        elif provider[0]=='PLANET' :

            output['data'] = [{"itemType" : provider[1],
                               "harmonizeTo" : "NONE",
                               "productBundle" : "analytic_sr_udm2",
                               "itemIds" : item_ids}]
        return output['data']

    def _order_query(self, dict_query,
                     provider, oauth, item_ids, name_query='') :
        '''
        Order the query
        Parameters
        ----------
        dict_query (dictionary) : dictionary from init_param_query() that allows to set all the parameters for your order
        provider (tuple) : choice between ('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'),('PLANET', 'PSScene4Band')
        oauth : oauth client
        item_ids (list) : list of the item ids returned by SH
        name_query (str) : name that you can specify to retrieve easily on init_param_query
        Returns
        -------

        '''
        # Order product
        payload = {
            "name" : dict_query["provider"].lower()+" products "+name_query,  # ,
            "input" : {
                "provider" : dict_query["provider"],
                "bounds" : {
                    "properties" : {
                        "crs" : "http://www.opengis.net/def/crs/EPSG/0/"+dict_query['CRS']
                    },
                    "geometry" : {
                        "type" : "Polygon",
                        "coordinates" : dict_query['input_polygon']
                    }
                },
                "data" : self._payload_data(item_ids, provider)
            }
        }
        if dict_query["provider"]=='PLANET' :
            payload["input"]["planetApiKey"] = self.config.planet_key

        response = oauth.post(self.url+"orders", json=payload)  #
        response.raise_for_status()

        order = response.json()

        # Confirm the order
        print(order['id'])
        print(order['sqkm'])
        print(order['status'])

        return order

    def execute_query(self, provider, name_query=''):
        '''
        Execute workflow to get commercial data
        Parameters
        ----------
        provider (tuple) : choice between ('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'),('PLANET', 'PSScene4Band')
        name_query (str) : name that you can specify to retrieve easily on init_param_query

        Returns order and collection id to download the data, accessible as well in https://apps.sentinel-hub.com/dashboard/#/tpdi
        -------

        '''
        if self.request:
            raise ValueError('You already have a search request running. Please confirm with confirm_order() method, or call the method reset_order() to launch a new order')
        self.oauth, _ = self._create_token()
        dict_query = self._init_param_query(provider)
        results, item_ids = self._search_query(dict_query, self.oauth)
        order = self._order_query(dict_query, provider, self.oauth, item_ids, name_query)
        self.request = True
        return order['id'], results

    def confirm_order(self, order_id):
        '''
        Confirm the order after the having ordered the query
        Parameters
        ----------
        order_id (str) : id of the order
        Returns collection_id for SH
        -------
        '''
        if self.order:
            raise ValueError(
                'You already have an order confirmed. Please wait a little for the order to be ingested in SH and/or reset the workflow to make a new order using reset_order() method')
        self.order = True
        # Confirm the order
        response = self.oauth.post(self.url+f'orders/{order_id}/confirm')
        response.raise_for_status()
        jobStatus = response.json()['status']
        if jobStatus=="RUNNING" :
            time.sleep(5)  # pause
            jobStatus = response.json()['status']
            print(jobStatus)

        ###########################################################
        response = self.oauth.get(self.url+f"orders/{order_id}")
        response.raise_for_status()
        order = response.json()

        if jobStatus=="RUNNING" :
            time.sleep(2)  # pause to let query ingested

        self.order = True

        return order_id, order['collectionId']

    def check_status(self, order_id) :
        oauth, _ = self._create_token()
        return oauth.get(self.url+f"orders/{order_id}").json()['status']

    def reset_workflow(self):
        ''' Reset workflow to make any new order'''
        self.request = False
        self.order = False

    def _workflow_vhrs(self, byoc, provider, resolution, bands_name='BANDS', pansharpen=False, otb_path = None, saving_path = None) :
        '''
        Workflow to process third party commercial data
        Parameters
        ----------
        eopatch
        shapefile
        resample_days

        Returns EOPatch
        -------
        '''
        evalscript_byoc = get_evalscript(provider)
        input_task = SentinelHubEvalscriptTask(
            features=[(FeatureType.DATA, 'BANDS'),
                      (FeatureType.MASK, 'IS_DATA'),
                      (FeatureType.MASK, 'CLM')],
            data_collection=byoc,
            resolution=resolution,
            config=self.config,
            time_difference=dt.timedelta(hours=12),
            evalscript=evalscript_byoc,
        )

        cloud_mask = utils_sh.CloudMaskFromCLM()
        add_polygon_mask = preprocessing.PolygonMask(self.shapefile)
        pixels_masking = preprocessing.MaskPixels([bands_name])

        if pansharpen :
            if otb_path is None :
                raise ValueError('You must provide path to the bin directory of OrfeoToolBox'
                                 ' https://www.orfeo-toolbox.org/CookBook/Installation.html')
            pansharpen = cmd_otb.PanSharpening(fname=bands_name,
                                               otb_path=otb_path)
            bands_name += '-PAN'
        else :
            pansharpen = utils_sh.EmptyTask()

        vis = vegetation_indices.VegetationIndicesVHRS(bands_name)

        if saving_path is None :
            save = utils_sh.EmptyTask()
        else :
            if not os.path.isdir(saving_path) :
                os.makedirs(saving_path)
            save = SaveTask(saving_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

        workflow = eolearn.core.LinearWorkflow(input_task,
                                               cloud_mask,
                                               add_polygon_mask,
                                               pixels_masking,
                                               pansharpen,
                                               vis,
                                               save)

        return input_task, workflow, save


    def get_data(self, order_id, collection_id, provider, resolution, pansharpen = False, otb_path = None) :
        '''
        Download the data from Sentinelhub using eo-learn package
        Parameters
        ----------
        order_id (str) : order id from the 3rd party import https://apps.sentinel-hub.com/dashboard/#/tpdi
        collection_id (str) : collection id
        pansharpen (bool) : perform pansharpenning using OTB
        otb_path (str) : path where you have your OTB bin installed
        Returns EOPatch
        -------

        '''
        field_bbox = utils.get_bounding_box(self.shapefile)
        ##########################################
        # Define the byoc
        byoc = DataCollection.define_byoc(
            collection_id=collection_id,
            name=str(order_id),
        )
        input_task, workflow, save = self._workflow_vhrs(byoc,
                                                         provider,
                                                         resolution,
                                                         bands_name='BANDS',
                                                         pansharpen=pansharpen,
                                                         otb_path = otb_path)

        result = workflow.execute({
            input_task : {'bbox' : field_bbox, 'time_interval': self.time_stamp}
        })

        return result.eopatch()




def get_evalscript(provider):
    '''
    Get evaluation script to download the data
    Parameters
    ----------
    provider (tuple) : choice between ('AIRBUS', 'PHR'), ('AIRBUS', 'SPOT'),('PLANET', 'PSScene4Band')
     w.r.t the execution order

    Returns    evaluation script
    -------
    '''
    global evalscript_byoc
    if provider[0]=='PLANET' :
        evalscript_byoc = """
        function setup() {
            return {
              input: ["B1", "B2", "B3", "B4", "UDM2_Cloud","dataMask"],
              output: [
                  {id: "BANDS", bands: 4, sampleType: SampleType.FLOAT32},
                  {id:"IS_DATA", bands:1, sampleType: SampleType.UINT8 },
                  {id:"CLM", bands:1, sampleType: SampleType.UINT8 }
              ],
            }
          }

          function evaluatePixel(sample) {
            return {
                BANDS: [2.5 * sample.B1 / 10000, 2.5 * sample.B2 / 10000, 2.5 * sample.B3 / 10000, 2.5 * sample.B4 / 10000],
                CLM : [sample.UDM2_Cloud],
                IS_DATA :  [sample.dataMask]
            }
          }
        """

    elif provider[0]=='AIRBUS' :

        evalscript_byoc = """
        function setup() {
            return {
              input:  ["B0", "B1", "B2", "B3","PAN", "dataMask", "dataMask"],
              output: [
                  {id: "BANDS", bands: 5, sampleType: SampleType.FLOAT32},
                  {id:"IS_DATA", bands:1, sampleType: SampleType.UINT8},
                  {id:"CLM", bands:1, sampleType: SampleType.UINT8}

              ],
            }
          }

          function evaluatePixel(sample) {
          v = 4000
            return {
                BANDS: [sample.B0/v, sample.B1/v, sample.B2/v, sample.B3/v,sample.PAN/v],
                IS_DATA :  [sample.dataMask],
                CLM :  [sample.dataMask]
            }
          }
        """

    return evalscript_byoc
