### Michael Engel ### 2022-09-17 ### ReduceME.py ###
import multiprocessing as mp
from queue import Empty
import sys
from contextlib import nullcontext

#%% process functions
#%%% Binary Tree
def _reducefun_BinaryTree(queue,ID,
                        threadcounter,
                        counter,threshold,
                        locky,
                        reducer=None, ordered=False,
                        timeout=1,bequiet=False):
    while True: 
        with locky:
            if not threadcounter.value<=(threshold+1-counter.value)//2: # condition absolutely crucial at this particular place!!! Maybe switch to end of while loop?
                threadcounter.value = threadcounter.value-1
                if threadcounter.value==0 and not bequiet:
                    print()
                break
            
        with nullcontext() if not ordered else locky: # use own lock here instead of locky -> implement soon!
            try:
                item1 = queue.get(timeout=timeout) # timeout very important (see merging not locked)
            except Empty:
                continue
                
            try:
                item2 = queue.get(timeout=timeout)
            except Empty:
                queue.put(item1)
                continue
        
        queue.put(item1+item2 if reducer==None else reducer(item1,item2))
        
        with locky:
            counter.value = counter.value+1 # Maybe put in front of merging? That could avoid a few calls to the queue but might be less robust if something unexpected happens while merging.
            if not bequiet:
                print(f"\r{counter.value}/{threshold} calculations [{counter.value/threshold*100:.2f}%] with {threadcounter.value} threads\t",end="\r",flush=True)

    return 0

#%%% M-ary Tree
def _reducefun_MaryTree(): # choose number of children
    raise NotImplementedError("ReduceME._reducefun_MaryTree not implemented yet!")

#%% main methods
#%%% Binary Tree
def reduce_BinaryTree(samples, reducer=None, ordered=False,
                     queue=None, timeout=1,
                     threads=0, checkthreads=True,
                     bequiet=False):
    #%%%% check input arguments
    if type(samples)==list:
        Nsamples = len(samples)
    elif type(samples)==int and queue!=None:
        Nsamples = samples
    else:
        raise RuntimeError(f"ReduceME.reduce_BinaryTree: Samples of type {type(samples)} not accepted! Samples given have to be a list or an integer denoting the expected number of samples given in a queue (i.e. which could be fed by a producer node)!")
    
    if queue==None:
        queue = mp.Queue()
        [queue.put(i) for i in samples]
    else:
        queue = queue
    
    if threads!=0:
        if checkthreads:
            threads = min(mp.cpu_count(),max(1,min(Nsamples,threads)))
            
        mainer = sys._getframe(1).f_globals["__name__"]
        if mainer=="__main__":
            pass
        else:
            print("ReduceME.reduce_BinaryTree: You are not calling this function from your main! Be aware that parallelization from within a spawned subprocess may cause severe resource issues!")
    
    #%%%% query
    counter = mp.Value("i",0)
    threshold = Nsamples-1
    locky = mp.Lock()
    if threads>0:
        threadcounter = mp.Value("i",threads)
        try:
            processes = []
            for ID in range(threads):
                processes.append(
                    mp.Process(
                        target = _reducefun_BinaryTree,
                        kwargs = {
                            "queue":queue,
                            "ID":ID,
                            "threadcounter":threadcounter,
                            "counter":counter,
                            "threshold":threshold,
                            "locky":locky,
                            "reducer":reducer,
                            "ordered":ordered,
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
        threadcounter = mp.Value("i",1)
        _reducefun_BinaryTree(
            **{
                "queue":queue,
                "ID":0,
                "threadcounter":threadcounter,
                "counter":counter,
                "threshold":threshold,
                "locky":locky,
                "reducer":reducer,
                "ordered":ordered,
                "timeout":timeout,
                "bequiet":bequiet
            }
        )
    else:
        raise RuntimeError(f"ReduceME.reduce_BinaryTree: {threads} threads not supported!")
    
    #%%%% return
    if queue.qsize()==1:
        return queue.get(timeout=timeout)
    else:
        print(f"ReduceME.reduce_BinaryTree: something unexpected happened! Queue has still {queue.qsize()} elements but should have one!")
        return queue

#%%% M-ary Tree
def reduce_MaryTree(): # choose number of children
    raise NotImplementedError("ReduceME.reduce_MaryTree not implemented yet!")

#%% dummy methods
def dummymerger(item1, item2):
    return item1+item2

#%% main
if __name__=="__main__":
    import time
    import numpy as np
    
    #%%% choose
    Nsamples = 12345
    samples = [1]*Nsamples
    threads = 5
    ordered = False # increases coordination overhead of processes if True
    
    #%%% query
    #%%%% multiprocessed
    start_multi = time.time()
    result_multi = reduce_BinaryTree(
        samples, reducer=dummymerger, ordered=ordered,
        queue=None, timeout=1,
        threads=threads, checkthreads=True,
        bequiet=False
    )
    time_multi = time.time()-start_multi
    
    #%%%% single process
    start_single = time.time()
    result_single = reduce_BinaryTree(
        samples, reducer=dummymerger, ordered=ordered,
        queue=None, timeout=1,
        threads=0, checkthreads=True,
        bequiet=False
    )
    time_single = time.time()-start_single
    
    #%%% results
    print(f"Reference:\t\t\t{np.sum(samples)}")
    print(f"Result Multi:\t\t{result_multi}")
    print(f"Result Single:\t\t{result_single}")
    print()
    print(f"Time Multi:\t\t{time_multi}s")
    print(f"Time Single:\t\t{time_single}s")