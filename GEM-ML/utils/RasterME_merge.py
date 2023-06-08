### Michael Engel ### 2021-05-22 ### RasterME_merge.py ###
import numpy as np
import uuid
import sys
import os
import subprocess
import psutil
import shutil

#%% auxiliary methods
def getME_RAM(key=None):
    if key==None:
        return dict(psutil.virtual_memory()._asdict())
    else:
        return psutil.virtual_memory()._asdict().get(key,-1)

def getME_SWAP(key=None):
    if key==None:
        return dict(psutil.swap_memory()._asdict())
    else:
        return psutil.swap_memory()._asdict().get(key,-1)

def getME_CPU(key=None,period=2):
    cpu_dict = {
            'percent':psutil.cpu_percent(period) if (key=='percent' or key==None) else -1,
            'threads':psutil.cpu_count(logical=True),
            'kernels':psutil.cpu_count(logical=False),
            'available':len(psutil.Process().cpu_affinity()),
            }
    if key==None:
        return cpu_dict
    else:
        return cpu_dict.get(key,-1)

def deleteME(file, bequiet=False):
    if type(file)==list:
        success = []
        for i in range(len(file)):
            success.append(deleteME(file[i]))
        return success
    else:
        try:
            os.remove(file)
            return True
        except Exception as e:
            if not bequiet:
                print(e)
                print("deleteME: removing did not work! Either it is not existing or you don't have permission for that, e.g. if it is still open in another application!")
            return False

def getME_uniquename(ending='.txt'):
        uniquename = str(uuid.uuid4())+ending
        return uniquename
    
def vrt2format(inputfile,outputfile,format_out=None,format_option=None,dtype_out=None,bequiet=False):
    #%%% format_out_string: output format
    if format_out==None:
        format_out_string = ''
    else:
        format_out_string = ' -of %s'%format_out
    
    #%%% format_option_string: output format creation option depending on the chosen driver (e.g. 'COMPRESS=LZW')
    if format_option==None:
        format_option_string = ''
    else:
        format_option_string = ' -co %s'%format_option
    
    #%%% dtype_out_string: datatype for output file (e.g. for reducing size)
    if dtype_out==None:
        dtype_out_string = ''
    else:
        dtype_out_string = ' -ot %s'%dtype_out
    
    #%%% quiet_string: suppress function output
    if bequiet:
        quiet_string = ' -q'
    else:
        quiet_string = ''
    
    #%%% query
    cmd= 'gdal_translate'+format_out_string+format_option_string+dtype_out_string+quiet_string+' '+inputfile+' '+outputfile
#    print(cmd)
    success = subprocess.call(cmd,shell=True)
    return success

#%% main method
def raster_merge(
    inputfiles, outputfile,
    coords=None, coords_crs=None,
    format_out=None, format_option=None, dtype_out=None, spacing_out=None, nodata_out=None, nodata_in=None, resampling=None,
    RAMmax=None, threads=1, bequiet=False, overwrite=True, nmax_files=None, sparse=False
):
    '''
    Parameters
    ----------
    inputfiles : list of strings
        Path to the files to be merged. Mind the order!
    outputfile : string
        File to write into.
    
    coords : ndarray with shape [2x2] or list of lists of numbers, optional
        desired bbox for the extend (in epsg:4326 each point should provide lon-coordinate first and lat-coordinate second). The default is None.
    coords_crs : string, optional ### NOT sure if it works for the cases of gdalbuildvrt or gdal_merge ###
        coordinate system the coordinates are given in (does NOT reproject the input or outputfile), allows to give epsg:4326 coordinates to extract from files which live in another coordinate system. The default is None.
        
    format_out : string, optional
        Desired format of the outputfile. Insert "VRT" for sparse output. The default is None.
    format_option : string, optional
        Desired format option of the outputfile (e.g. 'COMPRESS=DEFLATE'). The default is None.
    dtype_out : string, optional
        Desired data type of the outputfile. The default is None.
    spacing_out : int, float, list of two numbers for x and y, optional, optional
        Desired spacing of the outputfile. The default is None.
    nodata_out : number, optional
        Desired NoDataValue of the outputfile. The default is None.
    nodata_in : number, optional
        Chosen NoDataValue of the inputfiles. The default is None.
    resampling : string, optional
        Desired resampling method for the interpolation such that the desired spacing is achieved. The default is None.
     
    RAMmax : int, optional ### No influence on the calculation of VRTs! ###
        Desired RAM used in MB. The default is None.
    threads : int, optional ### No influence on the calculation of VRTs! ###
        Desired number of threads. The default is 1.
        
    bequiet : boolean, optional
        Decider whether the function speaks to you or not. The default is False.
    overwrite : boolean, optional
        Decider whether outputfile is overwritten if already existing. The default is True.
    nmax_files : number, optional
        Maximum number of files merged together since large number of large files results in unstable merging. The default is None.
    sparse: boolean, optional
        Decider whether the intermediate merging is done using VRTs. The default is False.

    Returns
    -------
    int
        This function merges the given files together. The last file in the list has highest priority. That is, if two files provide values for the same coordinate, the value of the file which comes later in the list of inputfiles is used.

    '''
    #%% check overwrite
    if not overwrite and os.path.isfile(outputfile):
        print('raster_merge: outputfile already exists!')
        return 0
    
    #%% recursive call for nmax_files
    if nmax_files==None:
        pass
    elif type(nmax_files)!=int or nmax_files<2:
        TypeError('raster_merge: nmax_files has to be a positive integer larger than 1!')   
    elif len(inputfiles)<=nmax_files:
        pass
    else:
        inputfiles_intermediate = inputfiles
        
        #%%% intermediate loop
        proceed = True
        counter = 0
        todelete = []
        while proceed:
            #%%%% grouping of files
            print(f'\nraster_merge: nested merging {counter}')
            intermediatefiles = [inputfiles_intermediate[i*nmax_files:(i+1)*nmax_files] for i in range(len(inputfiles_intermediate)//nmax_files)]
            if len(inputfiles_intermediate)//nmax_files*nmax_files!=len(inputfiles_intermediate):
                intermediatefiles.append(inputfiles_intermediate[len(inputfiles_intermediate)//nmax_files*nmax_files:])
            
            print('length of intermediate input\t',len(inputfiles_intermediate))  
            print('length of intermediate output\t',len(intermediatefiles))
            
            #%%%% merge intermediates
            intermediateoutputfiles = [f'{counter}_{i}_'+getME_uniquename(ending='') for i in range(len(intermediatefiles))]
            for i in range(len(intermediatefiles)):
                print(f'raster_merge: merge {len(intermediatefiles[i])} files to intermediate {i+1}/{len(intermediatefiles)}')
                raster_merge(intermediatefiles[i], intermediateoutputfiles[i], coords=coords, coords_crs=coords_crs, format_out='VRT' if sparse else format_out, format_option=format_option, dtype_out=dtype_out, spacing_out=spacing_out, nodata_out=nodata_out, nodata_in=nodata_in, resampling=resampling, RAMmax=RAMmax, threads=threads, bequiet=bequiet, overwrite=overwrite, nmax_files=None)
            
            #%%%% delete unnecessary temporary files
            if counter>0:
                if sparse and format_out!='VRT':
                    [todelete.extend(intermediatefiles[i]) for i in range(len(intermediatefiles))]
                else:
                    [deleteME(intermediatefiles[i]) for i in range(len(intermediatefiles))]
                intermediatefiles = []
                
            #%%%% decide whether to proceed splitted merging
            if len(intermediateoutputfiles)<=nmax_files:
                proceed = False
            else:
                inputfiles_intermediate = intermediateoutputfiles
            
            counter = counter+1
                
        #%%% final merge
        print('\nraster_merge: merge intermediates to final file')            
        success = raster_merge(intermediateoutputfiles, outputfile, coords=coords, coords_crs=coords_crs, format_out='VRT' if sparse else format_out, format_option=format_option, dtype_out=dtype_out, spacing_out=spacing_out, nodata_out=nodata_out, nodata_in=nodata_in, resampling=resampling, RAMmax=RAMmax, threads=threads, bequiet=bequiet, overwrite=overwrite, nmax_files=None)
        
        if format_out!='VRT':
            if sparse:
                tmp = getME_uniquename(ending='')
                success = vrt2format(outputfile,tmp,format_out=format_out,format_option=format_option,dtype_out=dtype_out,bequiet=bequiet)
                deleteME(outputfile)
                shutil.copy(tmp,outputfile)
                deleteME(tmp)
                deleteME(todelete)
                
            deleteME(intermediateoutputfiles)
            
        if success==0:
            print('\nraster_merge: DONE!')
        else:
            print('\nraster_merge: FAILED!')
        return success
    
    #%% standard case without nmax_files
    #%%% build cmd depending on function
    if format_out=='VRT':
        case = 'gdalbuildvrt'
    elif RAMmax==None and threads==None and resampling==None:
        case = 'gdal_merge'
    else:
        case = 'gdal_warp'
    
    #%%%% gdalbuildvrt
    if case=='gdalbuildvrt':
        #%%%%% input&output files
        #%%%%%% inputfiles_string: input
        if len(inputfiles)<2:
            print('raster_merge: one file cannot be merged... Continues anyway...')
        tmpfile = getME_uniquename(ending='.txt')
        np.savetxt(tmpfile,np.array(inputfiles),fmt='%s')
        inputfiles_string = ' -input_file_list %s'%tmpfile

        #%%%%%% outputfile_string: output
        outputfile_string = ' %s'%outputfile
        
        #%%%%% coords_string: choose coordinates of merged file
        if coords==None:
            coords_string = ''
        else:
            coords = np.array(coords)
            
            minlat = np.min(coords[:,1])
            maxlat = np.max(coords[:,1])
            minlon = np.min(coords[:,0])
            maxlon = np.max(coords[:,0])
            
            coords_string = " -te '%f %f %f %f'"%(minlon,minlat,maxlon,maxlat)
        
        #%%%%% coords_crs_string: reference system of desired coordinates (can be different from source file crs but does not reproject it)
        if coords_crs==None:
            coords_crs_string = ''
        else:
            print("raster_merge: Actually, I don't know if this works for gdalbuildvrt - you should probably insert the coords in the same crs as the input files!")
            coords_crs_string = f' -te_srs {coords_crs}'
        
        #%%%%% format_out_string: output format (e.g. Gtiff)
        if format_out==None:
            format_out_string = ''
        elif format_out=='VRT':
            format_out_string = ''
        else:
            print('raster_merge: format_out not implemented for gdalbuildvrt!')
            format_out_string = ''
            
        #%%%%% format_option_string: dataset open option
        if format_option==None:
            format_option_string = ''
        else:
            print('raster_merge: format_option not implemented for gdalbuildvrt!')
            format_option_string = ''
            
        #%%%%% dtype_out_string: datatype for output file (e.g. for reducing size)
        if dtype_out==None:
            dtype_out_string = ''
        else:
            print('raster_merge: dtype_out not implemented for gdalbuildvrt!')
            dtype_out_string = ''
        
        #%%%%% spacing_out_string: spacing of the output file (joint use with resampling recommended!)
        if spacing_out==None:
            spacing_out_string = ''
        else:
            if (type(spacing_out)==list or type(spacing_out)==np.ndarray) and len(spacing_out)==2:
                spacing_out_string = ' -tr %f %f'%(spacing_out[0],spacing_out[1])
            else:
                spacing_out_string = ' -tr %f %f'%(spacing_out, spacing_out)
                
        #%%%%% nodata_out_string: no data value for the output file
        if nodata_out==None:
            nodata_out_string = ''
        else:
            nodata_out_string = " -vrtnodata  %s"%str(nodata_out)

        #%%%%% nodata_in_string: no data value of the input files
        if nodata_in==None:
            nodata_in_string = ''
        else:
            nodata_in_string = " -srcnodata %s"%str(nodata_in)
        
        #%%%%% resampling_string: resampling method (e.g. near, bilinear, cubic, cubicspline, min, max, med...)
        if resampling==None:
            resampling_string = ''
        else:
            resampling_string = ' -r %s'%resampling
            
        #%%%%% RAMmax_string: maximum RAM used by method
        if RAMmax==None:
            RAMmax_string = ''
        else:
            print("raster_merge: RAMmax not implemented for gdalbuildvrt!")
            RAMmax_string = ''

        #%%%%% threads_string: number of threads used
        if threads==None or threads==1:
            threads_string = ''
        else:
            print("raster_merge: Multithreading not implemented for gdalbuildvrt!")
            threads_string = ''
        
        #%%%%% quiet_string: toggle function output
        if bequiet:
            quiet_string = ' -q'
        else:
            quiet_string = ''
            
        #%%%%% overwrite_string: force overwrite of output file (use carefully!)
        if overwrite:
            overwrite_string = ' -overwrite'
        else:
            overwrite_string = ''
        
        #%%%%% build cmd
        cmd = 'gdalbuildvrt'+coords_string+coords_crs_string+spacing_out_string+nodata_out_string+nodata_in_string+resampling_string+format_option_string+quiet_string+overwrite_string+inputfiles_string+outputfile_string
    
    #%%%% gdal_merge
    elif case=='gdal_merge':
        #%%%%% input&output files
        #%%%%%% inputfiles_string: input
        if len(inputfiles)<2:
            print('raster_merge: one file cannot be merged... Continues anyway...')
        tmpfile = getME_uniquename(ending='.txt')
        np.savetxt(tmpfile,np.array(inputfiles),fmt='%s')
        inputfiles_string = ' --optfile %s'%tmpfile

        #%%%%%% outputfile_string: output
        outputfile_string = ' -o %s'%outputfile
        
        #%%%%% coords_string: choose coordinates of merged file
        if coords==None:
            coords_string = ''
        else:
            coords = np.array(coords)
            
            minlat = np.min(coords[:,1])
            maxlat = np.max(coords[:,1])
            minlon = np.min(coords[:,0])
            maxlon = np.max(coords[:,0])
            
            coords_string = " -ul_lr %f %f %f %f"%(minlon,maxlat,maxlon,minlat)
        
        #%%%%% coords_crs_string: reference system of desired coordinates (can be different from source file crs but does not reproject it)
        if coords_crs==None:
            coords_crs_string = ''
        else:
            print("raster_merge: Actually, I don't know if this works for gdal_merge.py - you should probably insert the coords in the same crs as the input files!")
            coords_crs_string = f' -ul_lr_srs {coords_crs}'
            
        #%%%%% format_out_string: output format (e.g. Gtiff)
        if format_out==None:
            format_out_string = ''
        else:
            format_out_string = ' -of %s'%format_out
        
        format_option_string
        
        #%%%%% format_option_string: output format creation option depending on the chosen driver (e.g. 'COMPRESS=LZW')
        if format_option==None:
            format_option_string = ''
        else:
            format_option_string = ' -co %s'%format_option

        #%%%%% dtype_out_string: datatype for output file (e.g. for reducing size)
        if dtype_out==None:
            dtype_out_string = ''
        else:
            dtype_out_string = ' -ot %s'%dtype_out
            
        #%%%%% spacing_out_string: spacing of the output file (joint use with resampling recommended!)
        if spacing_out==None:
            spacing_out_string = ''
        else:
            if (type(spacing_out)==list or type(spacing_out)==np.ndarray) and len(spacing_out)==2:
                spacing_out_string = ' -ps %f %f'%(spacing_out[0],spacing_out[1])
            else:
                spacing_out_string = ' -ps %f %f'%(spacing_out, spacing_out)
                
        #%%%%% nodata_out_string: no data value for the output file
        if nodata_out==None:
            nodata_out_string = ''
        else:
            nodata_out_string = f" -a_nodata  '{nodata_out}'"

        #%%%%% nodata_in_string: no data value of the input files
        if nodata_in==None:
            nodata_in_string = ''
        else:
            nodata_in_string = f" -n '{nodata_in}'"
        
        #%%%%% resampling_string: resampling method (e.g. near, bilinear, cubic, cubicspline, min, max, med...)
        if resampling==None:
            resampling_string = ''
        else:
            print("raster_merge: resampling not implemented for gdal_merge.py!")
            resampling_string = ''
            
        #%%%%% RAMmax_string: maximum RAM used by method
        if RAMmax==None:
            RAMmax_string = ''
        else:
            print("raster_merge: RAMmax not implemented for gdal_merge.py!")
            RAMmax_string = ''

        #%%%%% threads_string: number of threads used
        if threads==None or threads==1:
            threads_string = ''
        else:
            print("raster_merge: Multithreading not implemented for gdal_merge.py!")
            threads_string = ''
        
        #%%%%% quiet_string: toggle function output
        if bequiet:
            quiet_string = ' -q'
        else:
            quiet_string = ''
        
        #%%%%% overwrite_string: force overwrite of output file (use carefully!)
        overwrite_string = ''
        
        #%%%%% build cmd
        cmd = sys.executable+' gdal_merge.py'+coords_string+coords_crs_string+format_out_string+format_option_string+dtype_out_string+spacing_out_string+nodata_out_string+nodata_in_string+resampling_string+RAMmax_string+threads_string+quiet_string+overwrite_string+inputfiles_string+outputfile_string
    
    #%%%% gdal_warp
    elif case=='gdal_warp':
        #%%%%% input&output files
        #%%%%%% inputfiles_string: input
        if len(inputfiles)<2:
            print('raster_merge: one file cannot be merged... Continues anyway...')
        tmpfile = getME_uniquename(ending='.txt')
        np.savetxt(tmpfile,np.array(inputfiles),fmt='"%s"')
        inputfiles_string = ' --optfile %s'%tmpfile

        #%%%%%% outputfile_string: output
        outputfile_string = ' "%s"'%outputfile
        
        #%%%%% coords_string: choose coordinates of merged file
        if coords==None:
            coords_string = ''
        else:
            coords = np.array(coords)
            
            minlat = np.min(coords[:,1])
            maxlat = np.max(coords[:,1])
            minlon = np.min(coords[:,0])
            maxlon = np.max(coords[:,0])
            
            coords_string = ' -te "%f %f %f %f"'%(minlon,minlat,maxlon,maxlat)
        
        #%%%%% coords_crs_string: reference system of desired coordinates (can be different from source file crs but does not reproject it)
        if coords_crs==None:
            coords_crs_string = ''
        else:
            coords_crs_string = f' -te_srs {coords_crs}'
            
        #%%%%% format_out_string: output format (e.g. Gtiff)
        if format_out==None:
            format_out_string = ''
        else:
            format_out_string = ' -of %s'%format_out
            
        #%%%%% format_option_string: output format creation option depending on the chosen driver (e.g. 'COMPRESS=LZW')
        if format_option==None:
            format_option_string = ''
        else:
            format_option_string = ' -co %s'%format_option

        #%%%%% dtype_out_string: datatype for output file (e.g. for reducing size)
        if dtype_out==None:
            dtype_out_string = ''
        else:
            dtype_out_string = ' -ot %s'%dtype_out
            
        #%%%%% spacing_out_string: spacing of the output file (joint use with resampling recommended!)
        if spacing_out==None:
            spacing_out_string = ''
        else:
            if (type(spacing_out)==list or type(spacing_out)==np.ndarray) and len(spacing_out)==2:
                spacing_out_string = ' -tr %f %f'%(spacing_out[0],spacing_out[1])
            else:
                spacing_out_string = ' -tr %f %f'%(spacing_out, spacing_out)
                
        #%%%%% nodata_out_string: no data value for the output file
        if nodata_out==None:
            nodata_out_string = ''
        else:
            nodata_out_string = f" -dstnodata  '{nodata_out}'"

        #%%%%% nodata_in_string: no data value of the input files
        if nodata_in==None:
            nodata_in_string = ''
        else:
            nodata_in_string = f" -srcnodata '{nodata_in}'"
        
        #%%%%% resampling_string: resampling method (e.g. near, bilinear, cubic, cubicspline, min, max, med...)
        if resampling==None:
            resampling_string = ''
        else:
            resampling_string = ' -r %s'%resampling
        
        #%%%%% RAMmax_string: maximum RAM used by method
        if RAMmax==None:
            RAMmax_string = ''
        else:
            maxRAM = getME_RAM(key='available')//10**6
            if RAMmax>maxRAM:
                RAMmax=maxRAM
            if RAMmax<10000:
                RAM = int(RAMmax)
            else:
                RAM = int(10**6*RAMmax) # since gdalwarp thinks of RAM in bytes if a number larger than 10000 is given
            RAMmax_string = ' -wm %i'%RAM

        #%%%%% threads_string: number of threads used
        if threads==None or threads==1:
            threads_string = ''
        else:
            maxthreads = getME_CPU(key='available')
            if threads>maxthreads:
                threads = maxthreads
            else:
                threads = int(threads)
            threads_string = ' -multi -wo NUM_THREADS=%i'%threads
            
        #%%%%% quiet_string: toggle function output
        if bequiet:
            quiet_string = ' -q'
        else:
            quiet_string = ''
            
        #%%%%% overwrite_string: force overwrite of output file (use carefully!)
        if overwrite:
            overwrite_string = ' -overwrite'
        else:
            overwrite_string = ''
            
        #%%%%% build cmd
        cmd = 'gdalwarp'+coords_string+coords_crs_string+format_out_string+format_option_string+dtype_out_string+spacing_out_string+nodata_out_string+nodata_in_string+resampling_string+RAMmax_string+threads_string+quiet_string+overwrite_string+inputfiles_string+outputfile_string
    
    #%%%% 
    else:
        print('raster_merge: case not defined!')
        return 1
    
    #%%% query
    success = subprocess.call(cmd, shell=True)
    deleteME(tmpfile)
    if success!=0:
        print('raster_merge: something went wrong -> try to run in another IDE or shell!')
        return 1
    else:
        return 0
    
#%% test
if __name__=='__main__':
    raise NotImplementedError("RasterME_merge: No test implemented yet!")