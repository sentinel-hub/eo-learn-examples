### Michael Engel ### 2022-10-21 ### ExecuteME.py ###
import multiprocessing as mp_dill
import multiprocessing as mp_orig
from queue import Empty
import sys

#%% execute with multiprocessing-type-kwargs
def _execute(
        kwargsqueue,
        mpkwargs,
        kwargsmode,
        resultsqueue,
        NoReturn,
        ID,
        threadcounter,
        counter,
        threshold,
        locky,
        fun,
        timeout,
        bequiet
    ):
    while True: 
        with locky:
            if not threadcounter.value<=(threshold-counter.value): # condition absolutely crucial at this particular place!!! Maybe switch to end of while loop?
                threadcounter.value = threadcounter.value-1
                if not bequiet:
                    print()
                break

        try:
            kwargs = kwargsqueue.get(timeout=timeout) # timeout very important (see merging not locked)
        except Empty:
            continue
        
        kwargs.update(mpkwargs)
        if kwargsmode=="kwargs":    
            result = fun(**kwargs)
        elif kwargsmode=="args":
            result = fun(*kwargs.values())
        else:
            result = fun(kwargs) 
        
        if not NoReturn:
            resultsqueue.put(result)
        
        with locky:
            counter.value = counter.value+1 # Maybe put in front of merging? That could avoid a few calls to the queue but might be less robust if something unexpected happens while merging.
            if not bequiet:
                print(f"\r{counter.value}/{threshold} calculations [{counter.value/threshold*100:.2f}%] with {threadcounter.value} thread(s)\t",end="\r",flush=True)
    return 0

def execute(
        fun,
        kwargslist,
        mpkwargs = None,
        kwargsmode = "kwargs",
        resultsqueue = None,
        NoReturn = False, # e.g. if the queue size would get too large!
        timeout = 1,
        threads = 0,
        checkthreads = True,
        multiprocessing_context = None,
        multiprocessing_mode = None,
        bequiet = False,
    ):
    if multiprocessing_mode==None or multiprocessing_mode=="std" or multiprocessing_mode=="standard":
        mp = mp_orig
    elif multiprocessing_mode=="dill":
        mp = mp_dill
    else:
        raise NotImplementedError(f"ExecuteME.execute: multiprocessing mode {multiprocessing_mode} not implemented!")
        
    if multiprocessing_context==None:
        context = mp.get_context()
    elif type(multiprocessing_context)==str:
        context = mp.get_context(multiprocessing_context)
    else:
        context = multiprocessing_context
        
    if isinstance(kwargslist,list):
        kwargsqueue = context.Queue()
        [kwargsqueue.put(kwargs) for kwargs in kwargslist]
    else:
        kwargsqueue = kwargslist
    
    if mpkwargs==None:
        mpkwargs = {}
    else:
        mpkwargs = mpkwargs
    
    if resultsqueue==None:
        resultsqueue = context.Queue()
    else:
        resultsqueue = resultsqueue
        
    threshold = kwargsqueue.qsize()
    if threads!=0:
        if checkthreads:
            threads = min(context.cpu_count(),max(1,min(threshold,threads)))
            
        mainer = sys._getframe(1).f_globals["__name__"]
        if mainer=="__main__":
            pass
        else:
            print("ExecuteME.execute: You are not calling this function from your main! Be aware that parallelization from within a spawned subprocess may cause severe resource issues!")
    
    counter = context.Value("i",0)
    locky = context.Lock()
    if threads>0:
        threadcounter = context.Value("i",threads)
        try:
            processes = []
            for ID in range(threads):
                processes.append(
                    context.Process(
                        target = _execute,
                        kwargs = {
                            "kwargsqueue":kwargsqueue,
                            "mpkwargs":mpkwargs,
                            "kwargsmode":kwargsmode,
                            "resultsqueue":resultsqueue,
                            "NoReturn":NoReturn,
                            "ID":ID,
                            "threadcounter":threadcounter,
                            "counter":counter,
                            "threshold":threshold,
                            "locky":locky,
                            "fun":fun,
                            "timeout":timeout,
                            "bequiet":bequiet
                        }
                    )
                )
            for process in processes:
                process.start()
        finally:
            [process.join() for process in processes]
    elif threads==0:
        threadcounter = context.Value("i",1)
        _execute(
            **{
                "kwargsqueue":kwargsqueue,
                "mpkwargs":mpkwargs,
                "kwargsmode":kwargsmode,
                "resultsqueue":resultsqueue,
                "NoReturn":NoReturn,
                "ID":0,
                "threadcounter":threadcounter,
                "counter":counter,
                "threshold":threshold,
                "locky":locky,
                "fun":fun,
                "timeout":timeout,
                "bequiet":bequiet
            }
        )
    else:
        raise RuntimeError(f"ExecuteME.execute: {threads} threads not supported!")
    
    # For progressbar
    print()
    
    if NoReturn:
        return 0
    elif resultsqueue.qsize()!=threshold:
        print(f"ExecuteME.execute: something unexpected happened! Queue has {resultsqueue.qsize()} elements but should have {threshold}!")
        return resultsqueue
    else:
        return [resultsqueue.get() for i in range(threshold)]
    
def Devices(devices,multiprocessing_context="spawn"):
    if multiprocessing_context==None:
        context = mp_orig.get_context()
    elif type(multiprocessing_context)==str:
        context = mp_orig.get_context(multiprocessing_context)
    else:
        context = multiprocessing_context
        
    queue = context.Queue()
    [queue.put(device) for device in devices]
    return queue

def _dummyfun(a):
    import time
    time.sleep(0.1)
    funny = lambda x: x**3
    return funny(a**2)

def _dummyfun2(a,queue):
    import time
    time.sleep(0.1)
    funny = lambda x: x**3
    result = funny(a**2)
    queue.put(result)
    return result/2

#%% main
if __name__=="__main__":
############################################################################### CASE STANDARD
    print("STANDARD CASE")
    import time
    
    #%%% choose
    Nsamples = 50
    kwargslist = [{"a":i} for i in range(Nsamples)]
    threads = 6
    
    #%%% query
    #%%%% multiprocessed
    start_multi = time.time()
    result_multi = execute(
        fun = _dummyfun,
        kwargslist = kwargslist,
        mpkwargs = None,
        kwargsmode = "kwargs",
        resultsqueue = None,
        NoReturn = False,
        timeout = 1,
        threads = threads,
        checkthreads = True,
        multiprocessing_context = None,
        multiprocessing_mode = None,
        bequiet = False
    )
    time_multi = time.time()-start_multi
    
    #%%%% single process
    start_single = time.time()
    result_single = execute(
        fun = _dummyfun,
        kwargslist = kwargslist,
        mpkwargs = None,
        kwargsmode = "kwargs",
        resultsqueue = None,
        NoReturn = False,
        timeout = 1,
        threads = 0,
        checkthreads = True,
        multiprocessing_context = None,
        multiprocessing_mode = None,
        bequiet = False
    )
    time_single = time.time()-start_single
    
    #%%% results
    print(f"Result Multi:\t\t{result_multi}")
    print(f"Result Single:\t\t{result_single}")
    print()
    print(f"Time Multi:\t\t{time_multi}s")
    print(f"Time Single:\t\t{time_single}s")
    print()
    
############################################################################### CASE MPKWARGS
    print("MPKWARGS CASE")
    import time
    
    #%%% choose
    Nsamples = 50
    queue = mp_orig.Queue()
    kwargslist = [{"a":i} for i in range(Nsamples)]
    mpkwargs = {"queue":queue}
    threads = 6
    
    #%%% query
    #%%%% multiprocessed
    start_multi = time.time()
    result_multi = execute(
        fun = _dummyfun2,
        kwargslist = kwargslist,
        mpkwargs = mpkwargs,
        kwargsmode = "kwargs",
        resultsqueue = None,
        NoReturn = False,
        timeout = 1,
        threads = threads,
        checkthreads = True,
        multiprocessing_context = "spawn",
        multiprocessing_mode = None,
        bequiet = False
    )
    time_multi = time.time()-start_multi
    
    #%%%% single process
    start_single = time.time()
    result_single = execute(
        fun = _dummyfun2,
        kwargslist = kwargslist,
        mpkwargs = mpkwargs,
        kwargsmode = "kwargs",
        resultsqueue = None,
        NoReturn = False,
        timeout = 1,
        threads = 0,
        checkthreads = True,
        multiprocessing_context = "spawn",
        multiprocessing_mode = None,
        bequiet = False
    )
    time_single = time.time()-start_single
    
    #%%% results
    print(f"Result Multi:\t\t{result_multi}")
    print(f"Result Single:\t\t{result_single}")
    print()
    print(f"Time Multi:\t\t{time_multi}s")
    print(f"Time Single:\t\t{time_single}s")
    print()
    print(f"Queue Size:\t\t{queue.qsize()}")
    print()