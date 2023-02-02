### Michael Engel ### 2021-04-16 ### ConfigME.py ###
'''
Configuration Class [by Michael Engel, m.engel@tum.de]
------------------------------------------------------
This class shall ease programming any type of scientific code.
To get familiar with this style of programming, the user can use the config-class as a dictionary.
I highly recommend taking advantage of the importME-method as it can import anything from anywhere!
By doing this, every module used is listed in the config which could help at freezing the project using tools like pyinstaller.
Another key feature of this class are the parsing methods which go through your scripts and look after config-callers.
By using this, writing config-files is extremely easy as even arguments of imported modules are queried and automatically written into your files!
Have fun!
'''

import os
import platform
import datetime as dt
import dill as pickle
import json
import importlib as il
import sys
import subprocess
import uuid
import urllib.request as urlr
import re
import inspect
import numpy as np

class Config():
    #%% initialization
    def __init__(self,name,savename=None,savename_config=None,envname=None):
        '''
        Parameters
        ----------
        name : string
            Desired name of the project the config shall be used for XOR the desired config-file if config.load() without any argument is called directly afterwards.
        savename : string, optional
            Desired savename as a basis for all things to be saved in your project. The default is None.
        savename_config : TYPE, optional
            Desired name for the config-file which can be forwarded and loaded afterwards to ease replicability of your project. The default is None.

        Returns
        -------
        None.

        '''
        
        self.name = name
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

        if savename==None:
            self.savename = self.name+'_'+self.timestamp
        else:
            self.savename = savename

        if savename_config==None:
            self.savename_config = 'config_'+self.savename+'.dill'
        else:
            self.savename_config = savename_config
    
        if envname==None:
            self.envname = self.savename
        else:
            self.envname = envname
        pass
    
    #%% let it run
    def run(self,script=None,savename_config=None):
        if script==None:
            if hasattr(self,'main'):
                script = self.main
            elif hasattr(self,'file_main'):
                script = self.file_main
            else:
                raise RuntimeError('Neither main or file_main defined in config nor other script given!')
        script = os.path.abspath(script)
        
        if savename_config==None:
            savename_config = self.savename_config
            
        cmd = [sys.executable, fr'{script}', '--config_file', fr'{savename_config}']
        success = runME(cmd,shell=True)
        if success != 0:
            print(self.name+': something went wrong with config.run!')
        else:
            pass
        
        return success

    #%% directories
    def checkdir(self):
        return self._checker('dir',self._checkdir,buzzword='DIRECTORY')
    
    def get_dirs(self):
        return self._get_dict('dir')
    
    def linuxify(self,bequiet=True):
        dictl = self.get_dirs()
        dictl.update(self.get_modules())
        dictl.update(self.get_files())
        for key,value in dictl.items():
            if type(value)==str:     
                if not bequiet and '\\' in value:
                    print(f'linuxify {key} : {value}')
                value = value.replace('\\','/')
            elif type(value)==list:
                value = [val.replace('\\','/') for val in value]
                
            self.__dict__.update({key:value})            
        pass
    
    #%% files
    def checkfile(self):
        return self._checker('file',self._checkfile,buzzword='FILE')
    
    def get_files(self):
        return self._get_dict('file')

    #%% modules
    def checkmodule(self):
        return self._checker('module',self._checkmodule,buzzword='MODULE')    
    
    def get_modules(self):
        return self._get_dict('module')
    
    #%% printing functions
    def print(self):
        print(self.__str__())
        
    def print_dirs(self):
        print(f'\nDirectories Configuration File {self.savename}')
        print(self._dict2str(self.get_dirs()))
        
    def print_files(self):
        print(f'\nFiles Configuration File {self.savename}')
        print(self._dict2str(self.get_files()))
        
    def print_modules(self):
        print(f'\nModules Configuration File {self.savename}')
        print(self._dict2str(self.get_modules()))
     
    #%% importME
    def importME(self,modules=None,bequiet=True,setsyspath=False): # supports absolute imports
        #%%% case if no modules given - import all modules listed in class
        if modules==None: # Please, don't do this!
            if not bequiet:
                print('\nIMPORT ALL MODULES')
            keys = list(self.__dict__.keys()).copy()
            values = list(self.__dict__.values()).copy()
        
            modules = []
            for i in range(len(keys)):
                if 'module' in keys[i].split('_') and values[i]!=None:
                    modules.append(values[i])
        
        return importME(modules,bequiet=bequiet,setsyspath=setsyspath)

    #%% loading, saving...
    def load(self,file=None,linuxify=True):
        if file==None:
            file = self.name

        name,ext = os.path.splitext(file)
        if ext==".py":
            config_ = importME(name+".config")
        else:
            with open(file, 'rb') as file_:
                config_ = pickle.load(file_)
        self.__class__ = config_.__class__
        self.__dict__ = config_.__dict__
        
        if linuxify and platform.system()!='Windows':
            self.linuxify(bequiet=True)
        pass
    
    @classmethod
    def LOAD(cls,file,linuxify=True):
        name,ext = os.path.splitext(file)
        if ext==".py":
            config_ = importME(name+".config")
        else:
            with open(file, 'rb') as file_:
                config_ = pickle.load(file_)
        
        if linuxify and platform.system()!='Windows':
            config_.linuxify(bequiet=True)
            
        return config_
    
    def save(self, savename=None):
        if savename==None:
            file = self.savename_config
        else:
            file = savename

        with open(file, 'wb') as outfile:
            pickle.dump(self,outfile,pickle.HIGHEST_PROTOCOL)
        return file
    
    def saveGIT(self, savename=None):
        ### TODO
        pass
    
    #%% parsing files for config['xxx'] entries and write them into a file or the file from which this method was called
    def parse(self,files=None,mode=1,doublewrite=False):
        if type(files)==list:
            pass
        elif files==None:
            dictl = self.get_files()
            files = list(dictl.values())
        else:
            files = [files]
            
        attributes_ = []
        for file in files:
            with open(file,'r') as file_:
                # attributes_.extend(re.findall(r"config\[[\'\"]{1}([\w\'\"]*?)[\'\"]{1}\]",file_.read()))
                attributes_.extend(re.findall(r"config\[[fr]?[\\]?[\'\"]{1}([\w\{\}\'\"]*?)[\\]?[\'\"]{1}\]",file_.read())) # for jupyter notebooks
                
        attributes = self.makeME_uniquelist(attributes_)
        if not doublewrite:
            attributes = list(set(attributes)-set(self.__dict__))
        attributes.sort()
        
        if mode==None:
            return attributes
        else:
            if mode==0 or mode==1:
                caller = inspect.getframeinfo(inspect.stack()[1][0])
                outfile = caller.filename
                index = caller.lineno
            elif type(mode)==str:
                outfile = mode
                index = 0
            else:
                print('Config.parse: wrong mode chosen!')
                return False
                
            if os.path.exists(outfile):
                with open(outfile,'r') as file_:
                    contents = file_.readlines()
                    file_.close()
            else:
                contents = []
            
            if mode==1 and len(contents)>0:
                index=index-1
                contents.pop(index) # get rid of caller in chosen file
                
            for idx,attribute in enumerate(attributes):
                idx = idx+index
                contents.insert(idx, f'config.{attribute} = \n')
                
            if type(mode)==str:
                contents.insert(0,'### Made with ConfigME by Michael Engel! ###\n')
            
            with open(outfile,'w') as file_:
                contents = "".join(contents)
                file_.write(contents)
                file_.close()
        return True
    
    def parse_args(self,files=None,mode=1): ### TODO: expand towards args; currently kwargs only
        '''
        Note that this method has to be called AFTER the modules in the config have already been assigned!
        '''
        if type(files)==list:
            pass
        elif files==None:
            dictl = self.get_files()
            files = list(dictl.values())
        else:
            files = [files]

        # query for all **config['some_params']
        attributes_ = []
        for file in files:
            with open(file,'r') as file_:
                # attributes_.extend(re.findall(r"\*{2}config\[[\'\"]{1}([\w\'\"]*?)[\'\"]{1}\]",file_.read()))
                attributes_.extend(re.findall(r"\*{2}config\[[fr]?[\\]?[\'\"]{1}([\w\{\}\'\"]*?)[\\]?[\'\"]{1}\]",file_.read())) # for jupyter notebooks
        
        # make unique params list and define corresponding module names due to convention module_<modulename>
        attributes = self.makeME_uniquelist(attributes_)
        modules = ['_'.join(['module']+param.split('_')[1:]) for param in attributes]
        
        # get arguments and their default values. if nothing is returned or an error happens, an empty list is given
        keys = []
        values = []
        for module in modules:
            # query arguments and default values
            try:
                args,defaults = getME_args(self.importME(self[module]),key=['args','defaults'])
            except Exception as e:
                print(e)
                print(f"Config.parse_args: {module} made some problems! Maybe it's not existing or something not covered by inspect.getfullargspec yet. Still, we continue!")
                args = []
                defaults = []
            
            # if None make empty list
            if defaults==None:
                defaults = []
            if args==None:
                args = []
            
            # take care of strings and the length of the list
            defaults = [f'"{default}"' if type(default)==str else default for default in defaults] # strings should be strings in the file
            defaults = ['']*(len(args)-len(defaults))+defaults # if there is no default, there still has to be something to put into the file
            
            keys.append(args)
            values.append(defaults)
        
        # return
        if mode==None:
            # return list
            return attributes,keys,values
        else:
            # check mode
            if mode==0 or mode==1:
                # get name and line of file this function was called
                caller = inspect.getframeinfo(inspect.stack()[1][0])
                outfile = caller.filename
                index = caller.lineno
            elif type(mode)==str:
                # set name
                print("Config.parse_args: this mode will very likely give an error but let's see! :D")
                outfile = mode
                index = 0
            else:
                print('Config.parse_args: wrong mode chosen!')
                return False
            
            # get list of contents if file already exists or make some dummy list for type(mode)=str
            if os.path.exists(outfile):
                with open(outfile,'r') as file_:
                    contents = file_.readlines()
                    file_.close()
            else:
                contents = ['dummy\n']
            
            # get rid of call in calling file for mode=1
            if mode==1 and len(contents)>0:
                index=index-1
                contents.pop(index) 
            
            # change contents
            for idx,attributekeyvalue in enumerate(zip(attributes,keys,values)):
                attribute, key, value = attributekeyvalue
                for i, content in enumerate(contents):
                    # looking for the line in file to insert the args and default values
                    if ((f'config.{attribute}' in content or f'config["{attribute}"]' in content or f"config['{attribute}']" in content) and not content.strip().startswith('#')) or type(mode)==str:
                        # replace original string by dictionary definition relying on the queried arguments and their default values
                        params_string = ''.join(['{\n']+[f"""\t"{key_}":config['{'_'.join([attribute,key_])}'],\n""" for key_ in key]+['}'])
                        contents[i] = f"\nconfig.{attribute} = {params_string}\n"
                        
                        # insert arguments and their default values if available
                        contents.insert(i+1, "\n")
                        for key_,value_ in zip(key[::-1],value[::-1]):
                            contents.insert(i, f"config.{'_'.join([attribute,key_])} = {value_ if value_=='' else str(value_)+' # think of changing this default value!'}\n")
                        contents.insert(i,f"# params for {attribute}\n")
                        
                        if type(mode)==str:
                            contents.insert(i,'### Made with ConfigME by Michael Engel! ###\n')
                            
                        break # necessary as going on forever otherwise
            
            # write changed contents to file
            with open(outfile,'w') as file_:
                contents = "".join(contents)
                file_.write(contents)
                file_.close()
            
            return True
    
    def parse_modules(self,files=None,mode=1): ### TODO
        if type(files)==list:
            pass
        elif files==None:
            dictl = self.get_files()
            files = list(dictl.values())
        else:
            files = [files]
            
        attributes_ = []
        for file in files:
            with open(file,'r') as file_:
                # attributes_.extend(re.findall(r"importME\(([\w]{1}[\'\"]{1}[\w\'\"\.\\/]*[\'\"]{1}?)[\w\'\"\.\,\=\\/]*\)",file_.read())) ### TODO: implement if format strings or raw strings are given!
                # file_.seek(0)
                attributes_.extend(re.findall(r"importME\([\'\"]{1}([\w\'\"\.\\/]*?)[\'\"]{1}[\w\'\"\.\,\=\\/]*\)",file_.read()))
                file_.seek(0)
                attributes_.extend(re.findall(r"importME\((config\[[\'\"]{1}[\w\'\"\.\\/]*[\'\"]{1}\]?)[\w\'\"\.\,\=\\/]*\)",file_.read()))
                
        attributes = self.makeME_uniquelist(attributes_)
        attributes.sort()
        
        if mode==None:
            return attributes
        else:
            if mode==0 or mode==1:
                caller = inspect.getframeinfo(inspect.stack()[1][0])
                outfile = caller.filename
                index = caller.lineno
            elif type(mode)==str:
                outfile = mode
                index = 0
            else:
                print('Config.parse_modules: wrong mode chosen!')
                return False
                
            if os.path.exists(outfile):
                with open(outfile,'r') as file_:
                    contents = file_.readlines()
                    file_.close()
            else:
                contents = []
            
            if mode==1 and len(contents)>0:
                index=index-1
                contents.pop(index) # get rid of caller in chosen file
            
            modulestring = ','.join([f"{attribute_}" if "config" in attribute_ else f"""'{attribute_}'""" for attribute_ in attributes])
                
            if not hasattr(self,'modules'):
                contents.insert(index, f"""config.modules = [{modulestring}]\n""")
            else:
                contents.insert(index, f"""config.modules_{uuid.uuid4().hex} = [{modulestring}]\n""")
                
            if type(mode)==str:
                contents.insert(0,'### Made with ConfigME by Michael Engel! ###\n')
            
            with open(outfile,'w') as file_:
                contents = "".join(contents)
                file_.write(contents)
                file_.close()
        return True
    
    #%% environment stuff
    def env_getEnv(self,mode=None): ### TODO: include pip packages!!!
        if hasattr(self,'environment') and mode==None:
            return self.environment
        
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            self.environment = json.loads(subprocess.check_output("conda env export --from-history --json",shell=True))
            return self.environment
        else:
            raise RuntimeWarning('Config.env_getEnv: mode not implemented!')
            return False
    
    def env_saveEnv(self,savename=None,mode=None):
        if savename==None:
            savename = self['envname']
        
        if os.path.splitext(savename)[0]=='.yml':
            pass
        else:
           savename = os.path.splitext(savename)[0]+'.yml' 
        
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            try:
                subprocess.check_output(fr"conda env export > {savename} --from-history",shell=True)
                return True
            except Exception as e:
                print(e)
                return False
        else:
            raise RuntimeWarning('Config.env_saveEnv: mode not implemented!')
            return False
    
    def env_getEnvs(self,mode=None,paths=False): # get list of existing envs on machine
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            envlist = json.loads(subprocess.check_output("conda env list --json",shell=True))["envs"]
            if paths:
                return envlist
            else:
                envlist = [os.path.basename(path) for path in envlist]
                return envlist
        else:
            raise RuntimeWarning('Config.env_getEnvs: mode not implemented!')
            return False
        
    def env_getChannels(self,mode=None):
        if hasattr(self,'channels') and mode==None:
            return self.channels
        
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            self.channels = json.loads(subprocess.check_output("conda env export --json",shell=True))["channels"] # this gets all channels used (compared to --from-history)
            return self.channels
        else:
            raise RuntimeWarning('Config.env_getChannels: mode not implemented!')
            return False
            
    def env_setChannel(self,channel,mode=None,overwrite=False):
        if type(channel)==list:
            success = []
            for channel_ in channel[::-1]: # if overwrite=True, the order is maintained if channels are overwritten
                success.append(self.env_setChannel(channel_, mode=mode, overwrite=overwrite))
            return success
        else:
            if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
                if channel in self.env_getChannels(mode=mode) and not overwrite:
                    print(f'Config.env_setChannel: channel {channel} already in environment! If you want to continue anyway, use overwrite=True!')
                    return True
                else:
                    print(subprocess.check_output(f"conda config --env --add channels {channel}",shell=True))
                    return True
            else:
                raise RuntimeWarning('Config.env_setChannel: mode not implemented!')
                return False
    
    def env_installEnv(self,name=None,mode=None,overwrite=False): ### TODO: include pip packages!!!
        if name==None:
            name = self['envname']
        environment = self.env_getEnv(mode=mode)
    
        if not overwrite and name in self.env_getEnvs(mode=mode,paths=False):
            print(f'Config.env_installEnv: {name} already existing on the machine!')
            return True
                
        if mode==None or mode=='ForceME' or mode=='eat this!':
            cmd = [f"conda create -n {name}",f"echo Created {name}!",f"conda activate {name}",f"echo Activated {name}!"]
            
            channels = environment['channels']
            channelstring = ' '.join(['-c '+channel for channel in channels])
            packages = environment['dependencies']
            for i, package in enumerate(packages):
                cmd.append(f"echo conda install {package} --yes {channelstring}")
                cmd.append(f"conda install {package} --yes {channelstring}")
                cmd.append(f"echo Done with {package}!")
            
            if platform.system()=='Windows':
                batFile = self.getME_uniquename('.bat')
                with open(batFile, 'w') as file:
                    file.write('call '+'\ncall '.join(cmd))
            else:
                batFile = os.path.abspath(self.getME_uniquename('.sh'))
                with open(batFile, 'w') as file:
                    file.write("#!/bin/bash -i\necho Initialized Shell!\n"+'\n'.join(cmd))
                runME(f"chmod +x {batFile}",shell=True)
            
            try:
                print(f'Config.env_installEnv: START installing {name}!')
                runME(batFile,shell=True)
                print(f'Config.env_installEnv: DONE installing {name}!')
                return True
            except Exception as e:
                print(e)
            finally:
                deleteME(batFile)

        elif mode=='conda':
            tmp = self.getME_uniquename('.yml')
            with open(tmp,'wb') as file:
                file.write(json.dumps(environment))
            try:
                if overwrite:
                    cmd = f"conda env create -f {tmp} -n {name} --force"
                else:
                    cmd = f"conda env create -f {tmp} -n {name}"
                print('Config.env_installEnv:',subprocess.check_output(cmd,shell=True))
                return True
            except Exception as e:
                print(e)
                return False
            finally:
                deleteME(tmp)
            
        else:
            raise RuntimeWarning('Config.env_installEnv: mode not implemented!')
            return False
        
    def env_adaptEnv(self,name=None,mode=None,overwrite=False):
        pass ### TODO: adapt/extend existing env
    
    def env_extendEnv(self,envlist,mode=None):
        pass ### TODO: extend own environment. maybe shouldn't do this...
    
    def env_deleteEnv(self,name=None,mode=None):
        if name==None:
            name = self['savename']
        
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            try:
                cmd = f"conda env remove --name {name}"
                print(subprocess.check_output(cmd,shell=True))
                return True
            except Exception as e:
                print(e)
                return False
        else:
            raise RuntimeWarning('Config.env_delete: mode not implemented!')
            return False
    
    def env_listenvs(self,mode=None):
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            print(subprocess.check_output("conda env list",shell=True).decode())
            return True
        else:
            raise RuntimeWarning('Config.env_listenvs: mode not implemented!')
            return False
    
    def env_list(self,mode=None):
        if mode==None or mode=='conda' or mode=='ForceME' or mode=='eat this!':
            print(subprocess.check_output("conda list",shell=True).decode())
            return True
        else:
            raise RuntimeWarning('Config.env_list: mode not implemented!')
            return False
            
    #%% dictionary like methods
    def update(self,*args,**kwargs): # one positional argument for loading a file and overwriting everything which is provided in the json-file given, keyword arguments for normal update as known for dictionaries
        if len(args)==1 and len(kwargs)==0:
            with open(*args, 'rb') as file_:
                model_ = pickle.load(file_)
            configdict = model_.__dict__
            keys = list(configdict.keys())
            values = list(configdict.values())
            for i in range(len(keys)):
                self.__setattr__(keys[i],values[i])
            pass
        elif len(args)==0:
            for key, value in kwargs.items():
                self.__setattr__(key, value)
            pass
        else:
            raise RuntimeError('One positional argument XOR multiple keywordarguments!')
    
    def extend(self,file): # same as self.update but WITHOUT overwriting name, savename, timestamp and savename_config, could be used to overwrite hardware-specific settings using a predefined config_hardware-file e.g. ### TODO: find a better name!
        with open(file, 'rb') as file_:
            model_ = pickle.load(file_)
        configdict = model_.__dict__
        keys = list(configdict.keys())
        values = list(configdict.values())
        for i in range(len(keys)):
            if keys[i]!='name' and keys[i]!='savename' and keys[i]!='timestamp' and keys[i]!='savename_config':
                self.__setattr__(keys[i],values[i])
        pass
    
    def join(self,files,priority='old'):
        if type(files)==list:
            pass
        elif type(files)==str:
            files = [files]
        
        dictlist = []
        for file in files:
            with open(file, 'rb') as file_:
                model_ = pickle.load(file_)
            dictlist.append(model_.__dict__)
          
        if priority=='new':
            dictlist.insert(0,self.__dict__)
        elif priority=='old':
            dictlist = dictlist[::-1]
            dictlist.append(self.__dict__)
        elif (type(priority)==list or type(priority)==np.ndarray) and len(priority)==len(files)+1:
            dictlist.insert(0,self.__dict__)
            dictlist = list(np.array(dictlist)[priority])
        else:
            print('Config.join: wrong priority given! If a list is given, note that the length has to be len(files)+1 as the joining config has to be taken into account as well!')
        
        configdict = {}
        for dictl in dictlist:
            configdict.update(dictl)
        self.__dict__ = configdict
        pass
    
    #%% utils ### TODO: Keep or remove?
    def getME_uniquename(self,ending=''):
        uniquename = str(uuid.uuid4())+ending
        return uniquename
        
    def makeME_uniquelist(self,listl):
        return list(dict.fromkeys(listl))
    
    def countME_uniquelist(self,listl):
        return len(dict.fromkeys(listl))
    
    #%% python builtins
    def __getitem__(self,key):
        return getattr(self,key,{})
    
    def __setitem__(self,key,value):
        return setattr(self,key,value)

    def __call__(self,file):
        self.load(file)
        pass

    def __str__(self):
        return f'\nConfiguration File {self.savename}\n'+self._dict2str(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)
    
    #%% hidden functions
    def _get_dict(self,buzzword):
        outdict = {}
        for key,value in self.__dict__.items():
            if buzzword in key.split('_'):
                outdict.update({key:value})
        return outdict
    
    def _dict2str(self, dicti):
        keys = list(dicti.keys())
        values = list(dicti.values())
        stringlist = []
        for i in range(len(keys)):
            stringlist.append('{}\t\t{} : {}'.format(i,keys[i],values[i]))
        return '\n'.join(stringlist)
    
    def _checker(self,key,fun,buzzword=''):
        print('\nCHECK {}'.format(buzzword))
        keys = list(self.__dict__.keys()).copy()
        values = list(self.__dict__.values()).copy()

        successlist = []
        for i in range(len(keys)):
            if key in keys[i].split('_'):
                success = fun(values[i])
                successlist.append([keys[i],success])
                if bool(success):
                    print('check key {}\t\t succeeded : {} : {}'.format(i,keys[i],values[i]))
                else:
                    print('check key {}\t\t FAILED : {} : {}'.format(i,keys[i],values[i]))
        return successlist

    def _checkdir(self,directory):
        if type(directory)==str:
            try:
                os.makedirs(directory, exist_ok=True)
                return True
            except:
                pass
            
        return False

    def _checkfile(self,file):
        if type(file)==str and os.path.isfile(file):
            return True
        else:
            return False
    
    def _checkmodule(self,module):
        return self.importME(module)

#%% context management
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodata', nargs='*',type=int, default=-1)
    parser.add_argument('--dir', nargs='+',required=True)
    parser.add_argument('--savename',nargs='*',default=None)
    args = parser.parse_args()
    return args

def works_for_all(func):
    def inner(*args, **kwargs):
        print("I can decorate any function")
        return func(*args, **kwargs)
    return inner

#%% auxiliary
def importME(modules,bequiet=True,setsyspath=False): # supports absolute imports
    if type(modules)==list:
        out = []
        for module in modules:
            out.append(importME(module,bequiet=bequiet))
    else:
        if "http" in modules.split("://") or "https" in modules.split("://") or "view-source" in modules.split(":"):
            url = modules
            ext = '.py'
            exts = []
            while(ext!=''):
                url, ext = os.path.splitext(url)
                exts.append(ext)
                
            tmp = os.path.join(os.getcwd(),str(uuid.uuid4()))
            
            try:
                urlr.urlretrieve(url,filename=tmp+'.py') # .py extension for tmp necessary as importlib does not recognize otherwise!
            except Exception as e:
                if not bequiet:
                    print(e)
                    print("trying url with .py-extension")
                urlr.urlretrieve(url+'.py',filename=tmp+'.py')
            
            out = importME(tmp+''.join(exts),bequiet=bequiet)
            if out==False: # "view-source" for gitlab (somehow not working otherwise)
                if not bequiet:
                    print('trying url with querying source code')
                out = importME("view-source:"+modules,bequiet=bequiet)
            
            try:
                os.remove(tmp+'.py')
            except Exception as e:
                if not bequiet:
                    print(e)
            
        else:
            try:
                out = il.import_module(f"{modules}")
            except Exception as e:
                if not bequiet:
                    print(e)
                    print('trying something else...')
                
                modules = modules.replace('\\','/') # make method os agnostic
                modules_ = os.path.abspath(modules)
                path = os.path.dirname(modules_)
                
                module_, ext = os.path.splitext(os.path.basename(modules_))
                module = module_ if ext=='.py' else os.path.basename(modules_)
                attributes = module.split('.')
                
                sys.path.append(path)
                try:
                    out = il.import_module(attributes[0])
                    for attr_ in attributes[1:]:
                        out = out.__getattribute__(attr_)
                except Exception as e:
                    if not bequiet:
                        print(e)
                        print('giving up...')
                        
                    raise ImportError(f'importME: Module {modules} not found!')
                    out = False
                finally:
                    if setsyspath==False or setsyspath==None:
                        sys.path.remove(path)
                    elif setsyspath=='permanent':
                        print('importME: Permanent sys path not implemented yet!')
                        pass
            
    return out

def getME_args(fun,key='args',ignoreself=True):
    # inspect function
    if inspect.isclass(fun):
        args,varargs,varkw,defaults,kwonlyargs,kwonlydefaults,annotations = inspect.getfullargspec(fun.__init__)
        # remove if desired
        if ignoreself:
            try:
                args.remove('self')
            except:
                pass
    else:
        args,varargs,varkw,defaults,kwonlyargs,kwonlydefaults,annotations = inspect.getfullargspec(fun)
    
    # build dictionary
    dict_args = {
        'args':args if args==None else list(args),
        'varargs':varargs if varargs==None else list(varargs),
        'varkw':varkw if varkw==None else list(varkw),
        'defaults':defaults if defaults==None else list(defaults),
        'kwonlyargs':kwonlyargs if kwonlyargs==None else list(kwonlyargs),
        'kwonlydefaults':kwonlydefaults if kwonlydefaults==None else list(kwonlydefaults),
        'annotations':annotations if annotations==None else list(annotations)
    }
    
    # return result
    if key==None:
        return dict_args
    elif type(key)==list:
        return list(map(dict_args.get,key,[False]*len(key)))
    else:
        return dict_args.get(key,False)

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

def runME(cmd,shell=False):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell)
    while True:
        output = process.stdout.readline().decode()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def get_most_recent_config(config_dir, pattern="config_.*[.]dill", mode="m"):
    method = os.path.getmtime if mode == "m" else os.path.getctime

    all_files = os.listdir(config_dir)
    configs = [file for file in all_files if re.search(pattern, file)]
    if not len(configs): raise ValueError(f"No configs found for given pattern '{pattern}'")

    times = [method(config_file) for config_file in configs]
    max_index = np.argmax(times)

    return configs[max_index]
