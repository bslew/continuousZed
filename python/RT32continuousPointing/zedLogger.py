'''
Created on Feb 4, 2022

@author: blew
'''
import logging
import os,sys
import datetime



def get_logging(name,fname,mode='a', lvl=logging.INFO):
    r'''
    Create and return a new logger to be used.
    
    Creates a logger object with name `name` that will log to file `fname`.
    The log file can be appended depending on the `mode` parameter.
    
    Parameters
    ----------

    name : str
        Logger name
        
    fname : str
        Log file name
        None to disable logging to file
        
    mode : str {'a', 'w'}, optional
        Log file open mode
        
    lvl : ``log level as defined in logging module`` {logging.DEBUG}
        
    Returns
    -------
    logger 
        logger object
    
    Raises
    ------
    
    Notes
    -----
    Notes here
    
    References
    ----------
    refs
    
    
    '''
    print("Starting logger {} with level {} (log file: {})".format(name,lvl,fname))
    if os.path.dirname(fname)!='':
        os.makedirs(os.path.dirname(fname),exist_ok=True)
    
    # create logger with 'spam_application'
    logger = logging.getLogger(name)
    logger.setLevel(lvl)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")

    
    # create file handler which logs even debug messages
    if fname!=None:
        fh = logging.FileHandler(fname,mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(ch)

    
    return logger
