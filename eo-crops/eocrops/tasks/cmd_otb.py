import numpy as np

import eolearn

from eolearn.core import FeatureType, EOTask
from pathlib import Path
from osgeo import gdal
import os
import shutil

import subprocess
import rasterio

from eolearn.io.local_io import ExportToTiffTask, ImportFromTiffTask



class MultitempSpeckleFiltering(EOTask):

    def __init__(self,  otb_path, feature_name = "BANDS-S1-IW", path_in = './', window = 3):
        '''
        Multitemporal filtering ONLY for Sentinel-1 data using OTB
        Parameters:
            otb_path (str) : Path where bin from Orfeo Toolbox package is installed
            path_in (str) : Path to write the temporary files (removed at the end of the process)
            window (int) : window to apply for Quegan filter for SAR data
        '''
        self.feature_name = feature_name
        self.otb_path = otb_path
        self.path_in = path_in
        self.window = window

    @staticmethod
    def _refactor_dates(t):
        # Add dates as suffix
        year, d, m = str(t.year), str(t.day), str(t.month)
        if len(d)==1 :
            d = '0'+d
        if len(m)==1 :
            m = '0'+m
        return '{0}{1}{2}'.format(year, m, d)

    def _apply_OTB_cmd(self, pol, ram = 8):
        path_in = os.path.join(self.path_in, 'S1_'+pol)
        s1_images = os.listdir(path_in)
        infiles = [os.path.join(path_in, k) for k in s1_images]
        infiles.sort()
        cmd = [os.path.join(self.otb_path, "otbcli_MultitempFilteringOutcore"), "-inl"]
        cmd += infiles
        cmd += ['-wr', str(self.window), '-oc', os.path.join(path_in, 'outcore.tif'), '-ram', str(8)]
        outdir = Path(path_in+'_filtered')

        if not outdir.exists() :
            os.mkdir(outdir)
        subprocess.call(cmd, shell=False)

        cmd = [os.path.join(self.otb_path, "otbcli_MultitempFilteringFilter"), "-inl"]
        cmd += infiles

        cmd += ['-enl', os.path.join(outdir, 'enl.tif'),
                '-wr', str(self.window),
                '-filtpath', outdir,
                '-oc', os.path.join(path_in, 'outcore.tif'),
                '-ram', str(ram)]

        subprocess.call(cmd, shell=False)

        outfiles = [os.path.join(outdir, k.split('.')[0]+'_filtered.tif') for k in s1_images]
        outfiles.sort()
        return infiles, outdir, outfiles

    def _save_temporary_geotiff(self, i, date, eopatch):
        ## TODO : Find a way to write temporary file without writing on disk using ExportToTiffTask to make the process faster
        export = ExportToTiffTask(feature=self.feature_name,
                                  folder=os.path.join(self.path_in, 'S1_VV/S1_VV_' + date),
                                  band_indices=[0],
                                  date_indices=[i])
        export.execute(eopatch)

        export = ExportToTiffTask(feature=self.feature_name,
                                  folder=os.path.join(self.path_in, 'S1_VH/S1_VH_' + date),
                                  band_indices=[1],
                                  date_indices=[i])
        export.execute(eopatch)


    def execute(self, eopatch, ram = 8):

        if os.path.exists(os.path.join(self.path_in, 'S1_VV')):
            shutil.rmtree(os.path.join(self.path_in, 'S1_VV'))
            shutil.rmtree(os.path.join(self.path_in, 'S1_VH'))

        os.mkdir(os.path.join(self.path_in, 'S1_VV'))
        os.mkdir(os.path.join(self.path_in, 'S1_VH'))

        times = list(eopatch.timestamp)
        for i, t in enumerate(times):
            date = self._refactor_dates(t)
            self._save_temporary_geotiff(i,date,eopatch)

        ########################################################################################################
        for pol in ['VV', 'VH']:

            infiles, outdir, outfiles = self._apply_OTB_cmd(pol,ram)
            ##########################################################################
            reference_file = infiles[0]
            with rasterio.open(reference_file) as src0:
                meta = src0.meta
                meta['nodata'] = 0.0
                meta['dtype'] = 'float32'

            meta['count'] = len(times)
            year = str(eopatch.timestamp[0].year)
            path_tif = outfiles[0].split('_' + year)[0] + '.tif'
            if 'outcore_filtered.tif' in os.listdir(outdir):
                outfiles.remove(os.path.join(outdir, 'outcore_filtered.tif'))
            outfiles.sort()

            with rasterio.open(os.path.join(os.getcwd(), path_tif), 'w', **meta) as dst:
                for i in range(1, len(times) + 1):
                    img = gdal.Open(os.path.join(os.getcwd(), outfiles[i - 1])).ReadAsArray()
                    dst.write_band(i, img)

            import_tif = ImportFromTiffTask((FeatureType.DATA, pol + '_filtered'), path_tif)
            eopatch = import_tif.execute(eopatch)


        shutil.rmtree(os.path.join(self.path_in, 'S1_VV_filtered'))
        shutil.rmtree(os.path.join(self.path_in, 'S1_VH_filtered'))
        shutil.rmtree(os.path.join(self.path_in, 'S1_VV'))
        shutil.rmtree(os.path.join(self.path_in, 'S1_VH'))

        return eopatch



class PanSharpening(EOTask):
    def __init__(self, fname = 'BANDS',
                 otb_path =  '/home/s999379/git-repo/OTB-7.4.0-Linux64/bin',
                 path_temporary_files = './tempo'):
        '''
        Multitemporal filtering ONLY for Sentinel-1 data using OTB
        Parameters:
            fname (str) : Name of the feature stored in data that gathers the bands
            otb_path (str) : Path where bin from Orfeo Toolbox package is installed
            path_temporary_files (str) : path to save the temporary geotiff file to call OTB
        '''
        self.fname = fname
        self.otb_path = otb_path
        self.path_temporary_files = path_temporary_files

    @staticmethod
    def _refactor_dates(t):
        # Add dates as suffix
        year, d, m = str(t.year), str(t.day), str(t.month)
        if len(d)==1 :
            d = '0'+d
        if len(m)==1 :
            m = '0'+m
        return '{0}{1}{2}'.format(year, m, d)


    def _extracted_from__save_temporary_geotiff(self, date, i, eopatch, band_indices=None):

        if band_indices is None :
            band_indices = list(range(4))

        export = ExportToTiffTask(feature=self.fname,
                                  folder=os.path.join(self.path_temporary_files, 'PAN_' + date),
                                  band_indices=[-1],
                                  date_indices=[i])
        export.execute(eopatch)

        export = ExportToTiffTask(feature=self.fname,
                                  folder=os.path.join(self.path_temporary_files, 'BANDS_' + date),
                                  band_indices=band_indices,
                                  date_indices=[i])
        export.execute(eopatch)


    def _apply_OTB_cmd(self, date):

        cm = [os.path.join(self.otb_path, 'otbcli_Pansharpening'),
              '-inp',os.path.join(self.path_temporary_files,'PAN_' + date +'.tif'),
              '-inxs', os.path.join(self.path_temporary_files,'BANDS_' + date +'.tif'),
              '-method', 'lmvm',
              '-out', os.path.join(self.path_temporary_files, 'Pansharpened_' + date +'.tif'),
              'float']

        subprocess.call(cm, shell=False)


    def _clean_temporary_files(self):
        shutil.rmtree(self.path_temporary_files)

    def execute(self, eopatch, band_indices=None):

        times = list(eopatch.timestamp)

        pan_bands = []
        for i, t in enumerate(times):
            date = self._refactor_dates(t)
            self._extracted_from__save_temporary_geotiff(date, i, eopatch, band_indices)
            self._apply_OTB_cmd(date)
            img = gdal.Open(os.path.join(self.path_temporary_files, 'Pansharpened_' + date +'.tif')).ReadAsArray()
            img = np.moveaxis(img, 0, -1)
            pan_bands.append(img)

        pan_bands = np.stack(pan_bands, axis =0)
        self._clean_temporary_files()

        eopatch.add_feature(eolearn.core.FeatureType.DATA, 'BANDS-PAN', pan_bands)

        return eopatch