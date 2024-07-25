#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import obspy
import numpy as np
import mc_lay as mcl
from obspy import read, read_events, read_inventory
import json
from obspy.geodetics import gps2dist_azimuth
import datetime
import logging.config
from copy import copy, deepcopy
import multiprocessing  as mp
from mpi4py.futures import MPIPoolExecutor
import os
import matplotlib.pyplot as plt
import pygad
from scipy.optimize import least_squares

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 



LOGLEVELS = {0: 'CRITICAL', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}

logger = logging.getLogger('att1d')
logger.addHandler(logging.NullHandler())

LOGGING_DEFAULT_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'capture_warnings': True,
    'formatters': {
        'file': {
            'format': ('%(asctime)s %(module)-10s%(process)-6d%(levelname)-8s'
                       '%(message)s')
        },
        'console': {
            'format': '%(levelname)-8s%(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'level': None,
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'level': None,
            'filename': None,
        },
    },
    'loggers': {
        'att1d': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'py.warnings': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        }

    }
}


def configure_logging(loggingc, verbose=0, loglevel=3, logfile=None):
    if loggingc is None:
        loggingc = deepcopy(LOGGING_DEFAULT_CONFIG)
        if verbose > 3:
            verbose = 3
        loggingc['handlers']['console']['level'] = LOGLEVELS[verbose]
        if logfile is None or loglevel == 0:
            del loggingc['handlers']['file']
            loggingc['loggers']['handlers'] = ['console']
            loggingc['loggers']['py.warnings']['handlers'] = ['console']
        else:
            loggingc['handlers']['file']['level'] = LOGLEVELS[loglevel]
            loggingc['handlers']['file']['filename'] = logfile
    logging.config.dictConfig(loggingc)
    logging.captureWarnings(loggingc.get('capture_warnings', False))
    
    
class ConfigJSONDecoder(json.JSONDecoder):
    """Decode JSON config with comments stripped"""
    def decode(self, s):
        s = '\n'.join(l.split('#', 1)[0] for l in s.split('\n'))
        return super(ConfigJSONDecoder, self).decode(s)


def get_station(seedid):
    """Station name from seed id"""
    st = seedid.rsplit('.', 2)[0]
    if st.startswith('.'):
        st = st[1:]
    return st


def get_freqs(max=None, min=None, step=None, width=None, cfreqs=None,
              fbands=None):
    """Determine frequency bands"""
    if cfreqs is None and fbands is None:
        max_exp = int(np.log(max / min) / step / np.log(2))
        exponents = step * np.arange(max_exp + 1)[::-1]
        cfreqs = max / 2 ** exponents
    if fbands is None:
        df = np.array(cfreqs) * (2 ** width - 1) / (2 ** width + 1)
        fbands = [(f, (f - d, f + d)) for d, f in zip(df, cfreqs)]
    else:
        fbands = sorted(fbands)
        cfreqs = [0.5 * (f1 + f2) for f1, f2 in fbands]
        fbands = [(0.5 * (f1 + f2), (f1, f2)) for f1, f2 in fbands]
    return fbands


def get_eventid(event):
    """Event id from event"""
    return str(event.resource_id).split('/')[-1]


def get_origin(event):
    """Preferred or first origin from event"""
    return event.preferred_origin() or event.origins[0]


def get_distance(event, stat_coord):
    """"Distance calculation"""
    ori = get_origin(event)
    hdist = gps2dist_azimuth(ori.latitude, ori.longitude, 
                    stat_coord.get('latitude'), stat_coord.get('longitude'))[0]
    vdist = ori.depth + stat_coord.get('elevation') \
            - stat_coord.get('local_depth')
    return np.sqrt(hdist ** 2 + vdist ** 2)


def get_picks(arrivals, station):
    """Picks for specific station from arrivals"""
    picks = {}
    for arrival in arrivals:
        phase = arrival.phase.upper()
        if phase in ('PG', 'SG'):
            phase = phase[0]
        if phase not in 'PS':
            continue
        pick = arrival.pick_id.get_referred_object()
        seedid = pick.waveform_id.get_seed_string()
        if station == get_station(seedid):
            picks[phase] = pick.time
    return picks


def energy_env(data, rho, df, fs=4):
    """Spectral energy density of one channel"""
    hilb = scipy.fftpack.hilbert(data)
    return rho * (data ** 2 + hilb ** 2) / 2 / df / fs


def smooth_(x, window_len=None, window='flat', method='zeros'):
    """Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an
        odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    :param method: handling of border effects\n
        'zeros': zero padding on both ends (len(smooth(x)) = len(x))\n
        'reflect': pad reflected signal on both ends (same)\n
        'clip': pad signal on both ends with the last valid value (same)\n
        None: no handling of border effects
        (len(smooth(x)) = len(x) - len(window_len) + 1)
    See also:
    www.scipy.org/Cookbook/SignalSmooth
    """
    if window_len is None:
        return x
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if method == 'zeros':
        s = np.r_[np.zeros((window_len - 1) // 2), x,
                  np.zeros(window_len // 2)]
    elif method == 'reflect':
        s = np.r_[x[(window_len - 1) // 2:0:-1], x,
                  x[-1:-(window_len + 1) // 2:-1]]
    elif method == 'clip':
        s = np.r_[x[0] * np.ones((window_len - 1) // 2), x,
                  x[-1] * np.ones(window_len // 2)]
    else:
        s = x
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    return np.convolve(w / w.sum(), s, mode='valid')


def time2utc(time, trace, starttime=None):
    """Convert string with time information to UTCDateTime object
    :param time: can be one of:\n
         "OT+-???s" seconds relative to origin time\n
         "P+-???s" seconds relative to P-onset\n
         "S+-???s" seconds relative to S-onset\n
         "???Ptt" travel time relative to P-onset travel time\n
         "???Stt" travel time relative to S-onset travel time\n
         "???SNR" time after which SNR falls below this value
                  (after time given in starttime)\n
         "time>???SNR" time after which SNR falls below this value
                  (after time given in front of expression)
    :param trace: Trace object with stats entries
    :param starttime: UTCDatetime object for SNR case.
    """
    ot = trace.stats.origintime
    p = trace.stats.ponset
    s = trace.stats.sonset
    time = time.lower()
    if time.endswith('snr'):
        if '>' in time:
            time1, time = time.split('>')
            if time1 != '':
                st = time2utc(time1, trace)
                starttime = st
        assert starttime is not None
        tr = trace.slice(starttime=starttime)
        snr = float(time[:-3])
        noise_level = tr.stats.noise_level
        try:
            index = np.where(tr.data < snr * noise_level)[0][0]
        except IndexError:
            index = len(tr.data)
        t = starttime + index * tr.stats.delta
    elif time.endswith('stt') or time.endswith('ptt'):
        rel = p if time[-3] == 'p' else s
        t = ot + float(time[:-3]) * (rel - ot)
    elif ((time.startswith('s') or time.startswith('p') or
           time.startswith('ot')) and time.endswith('s')):
        rel = p if time.startswith('p') else s if time.startswith('s') else ot
        t = rel + float(time[2:-1] if time.startswith('ot') else time[1:-1])
    else:
        raise ValueError('Unexpected value for time window')
    return t


def tw2utc(tw, trace):
    """Convert time window to UTC time window
    :param tw: tuple of two values, both can be a string (see :func:`time2utc`)
        or a list of strings in which case the latest starttime and earliest
        endtime is taken.
    :param trace: Trace object with stats entries
    """
    starttime = None
    for val in tw:
        if isinstance(val, (list, tuple)):
            times = [time2utc(v, trace, starttime=starttime) for v in val]
            t = max(times) if starttime is None else min(times)
        else:
            t = time2utc(val, trace, starttime=starttime)
        if starttime is None:
            starttime = t
    return starttime, t


def _get_local_minimum(tr, smooth=None, ratio=5, smooth_window='flat', 
                       bulk_win_len = None):
    """Get local minimum of the coda"""
    data = tr.data
    if smooth:
        window_len = int(round(smooth * tr.stats.sampling_rate))
        try:
            data = smooth_(tr.data, window_len=window_len, method='clip',
                           window=smooth_window)
        except ValueError:
            pass
    mins = scipy.signal.argrelmin(data)[0]
    maxs = scipy.signal.argrelmax(data)[0]
    if len(mins) == 0 or len(maxs) == 0:
        return
    mins2 = [mins[0]]
    for mi in mins[1:]:
        if data[mi] < data[mins2[-1]]:
            mins2.append(mi)
    mins = np.array(mins2)
    for ma in maxs:
        try:
            mi = np.nonzero(mins < ma)[0][-1]
            mi = mins[mi]
        except IndexError:
            mi = 0
        if data[ma] / data[mi] > ratio:
            return tr.stats.starttime + mi * tr.stats.delta


def get_T(freq, filter_, st, noise_windows, coda_window, twin_min, smooth, 
          smooth_window, cut_coda, rho, df, fs, dt):
    """Determine coda window length"""
    sr = st[0].stats.sampling_rate  
    
    filter__=copy(filter_)
    if freq[1] > 0.495 * sr:
        fu = {'freq': freq[0], 'type': 'highpass'}
    else:
        fu = {'freqmin': freq[0], 'freqmax': freq[1], 'type': 'bandpass'}       
    filter__.update(fu) 
    
    st.detrend('linear') 
    st.filter(**filter__) 
    
    data = [energy_env(tr.data, rho, df, fs=fs) for tr in st]

    Ns = [len(d) for d in data]
    if max(Ns) - min(Ns) > 0:
        T, acp, twin_len = 0, 0, 0
        return T, acp, twin_len

    data = np.sum(data, axis = 0)
    tr = obspy.Trace(data = data, header = st[0].stats)
     
    if smooth:
        tr.data = smooth_(tr.data, int(round(sr * smooth)),
                          window=smooth_window, method='zeros')    
    noise_levels = []
    for nw in noise_windows:
        noisew = tw2utc(nw, tr)
        tr_noisew = tr.slice(*noisew)
        noise_levels.append(np.mean(tr_noisew.data))
        
    tr.stats.noise_level = np.min(noise_levels)
    time_window = tw2utc(coda_window, tr)   
    if cut_coda:
        bulk_win = cut_coda.get('bulk_win_len')
        if cut_coda.get('smooth'):
            seam = 0.5 * cut_coda['smooth']
            cw = (time_window[0] - seam + bulk_win, time_window[1] + seam)
            if cw[0] >= cw[1]:
                esl = tr.slice(cw[1]-1, cw[1])
                time_window = (cw[1]-1, cw[1])
            else:
                esl = tr.slice(cw[0], cw[1])
                time_window = (cw[0]+ seam, cw[1]- seam)
        else:
            esl = tr.slice(time_window[0] + bulk_win, time_window[1])
            time_window = (time_window[0] + bulk_win, time_window[1])
        cut_coda.setdefault('smooth_window', smooth_window)
        tmin = _get_local_minimum(esl, **cut_coda)
        time_window=(time_window[0] - bulk_win,time_window[1])
    if time_window[1] - time_window[0] < twin_min:
        T, acp = time_window[1] - tr.stats.origintime, 0
        twin_len = time_window[1] - time_window[0]
    elif tmin: 
        if tmin - time_window[0] < twin_min:
            T, acp = tmin - tr.stats.origintime, 0
            twin_len = tmin - time_window[0]
        else:
            T, acp = tmin - tr.stats.origintime, 1
            twin_len = tmin - time_window[0]
    else:
        T, acp = time_window[1] - tr.stats.origintime, 1
        twin_len = time_window[1] - time_window[0]
    tr.trim(tr.stats.origintime, time_window[1])   
    
    if acp == 1:
        tr=tr.resample(100)
        Obs=tr.data
        deci = int(dt /(1/100))
        Obs = Obs[::deci]
        Obs=smooth_(Obs,window_len=window_len,window='bartlett',method='clip') 
    else:
        Obs=0

    return T, acp, twin_len, Obs


def find_eps(l_star, k):
   return np.log10(np.interp(1/l_star, [mcl.gtr_3d_karman(k_depth[k],a_depth[k],ii,kappa_depth[k]) for ii in np.logspace(-5,1,100000)], np.logspace(-5,1,100000)))


def mc(k_depth,a_depth,eps_depth,kappa_depth,b_depth,s_dp, particles_n, r_dist, r_depth,nrec, dt, rec_radius, v_depth,v_vel, rho_depth, att_depth,rdm_sc_l,ntime):
                e_dp, tt, rec_vol = mcl.mc_lay_3d_niso(k_depth,a_depth,eps_depth,kappa_depth,b_depth,s_dp, particles_n, r_dist, r_depth,nrec, dt, rec_radius, v_depth,v_vel, rho_depth,att_depth,rdm_sc_l,ntime)
                return e_dp, tt, rec_vol
   

def shift_error(shift,c1,c2):
    return c1 - (c2+shift)


            
def error(evi, att):
    k=evid_.index(evi)
    nrec=len(stations[k])
    s_dp=source_depth[k]
    r_dist=rec_dist[k]
    r_depth=rec_depth[k]
    ntime=ntimes[k]
    T_stat=maxtimes[k]
    spicks_=spicks[k]
    env_obs=energies[k]
    
    eps=att[0]
    b=att[1]

    result_list = []
    def log_result(result):
        result_list.append(result)
        
    n_cpu=ncpu

    particles_n=int(particles/n_cpu)
    
    def para_async(eps,b):
        pool = mp.Pool()
        for i in range(n_cpu):
            res=pool.apply_async(mc, args = (k_depth,a_depth,eps,kappa_depth,b,s_dp, particles_n, r_dist, r_depth,nrec, dt, rec_radius, v_depth,v_vel, rho_depth,att_depth,int(len(rdm_sc)-1),ntime), callback = log_result)
        pool.close()
        pool.join()
        return np.array(result_list)
    result = para_async(eps,b)
    e_dp = np.sum(result[:,0], axis=0)
    tt = result[0,1]
    rec_vol = result[0,2]


    W_, Obs_, Syn_ = [], [], []
    for i in range(nrec):
        
        env_syn=e_dp[i]/(particles*rec_vol[i]) / 1e9
        
        try:
            ttsyn=tt[np.where(np.diff(env_syn) == 0)[0][-1]+2]
        except:
            ttsyn=1000
                
        if spicks_[i]+1<ttsyn:
            ttwin=spicks_[i]+1
        else: 
            ttwin=max(ttsyn, spicks_[i])
            
        ind = int(round(((ttwin))/dt))
        ind_max = int(round(T_stat[i]/dt))

        Syn=smooth_(env_syn,window_len=window_len,window='bartlett',method='clip')
        
        for j_ in range(1, int(round(((ttwin+3))/dt))):
            if abs(np.log10(Syn[j_]) - np.log10(Syn[j_-1])) > 30:
                Syn[:j_] = [Syn[j_]] * j_
        
        Syn=Syn[ind:ind_max]
        Syn[Syn<1e-30] = 1e-30
        
        Obs=env_obs[i][ind:ind_max]

        if len(Obs) < len(Syn):
            Syn = Syn[:len(Obs)]
            
        if len(Obs) > len(Syn):
            Obs = Obs[:len(Syn)]
            
        res = least_squares(shift_error, x0=1, args=(np.log10(Obs),np.log10(Syn)))
        
        W_.append(10**res.x[0])
        Obs_.append(Obs)
        Syn_.append(Syn)

    return k, W_, Obs_, Syn_
  



def sitesandsource(z):
    ii = int(len(z)/2)
        
    eps=[10**float(i) for i in z[:ii]]  
    b=[10**float(i) for i in z[ii:]]
       
    results=[]
    for evid in evid_:
        val = error(evid, [eps,b])
        results.append(val)


    #estimate sites and source    
    sta_unq=[]
    sta_unq=stations.copy()
    sta_unq={x for l in sta_unq for x in l}
    sta_unq=[x for x in sta_unq]
    sta_unq.sort()

    eqs=[]
    for j, stas in enumerate(stations):
        indices = [sta_unq.index(i) for i in stas]
        for i in indices:
            arr=np.zeros(len(evid_)+len(sta_unq))
            arr[j] = 1
            arr[i + len(evid_)] = 1
            eqs.append(arr)
    
    arr=np.zeros(len(evid_)+len(sta_unq))
    arr[len(evid_):] = 1
    eqs.append(arr)
    
    W_mod=[i[1] for i in results].copy()
    W_mod.append([len(sta_unq)])
    
    res, _, _, _ = np.linalg.lstsq(eqs, np.log10(np.array(np.concatenate(W_mod))), rcond=None)
    
    sitsds = {'events': {k: None for k in [get_eventid(i) for i in events_]}, 
               'stations': {k: None for k in stations_}}
    for i in sitsds['events']:
            sitsds['events'][i]=[]
    for i in sitsds['stations']:
            sitsds['stations'][i]=[]
    
    for i in [get_eventid(i) for i in events_]:
        if i in evid_:
            sitsds['events'][i].append(10**res[evid_.index(i)])
        else:
            sitsds['events'][i].append(None)
    
    for i in stations_:
        if i in sta_unq:
            sitsds['stations'][i].append(10**res[sta_unq.index(i)+len(evid_)])
        else:
            sitsds['stations'][i].append(None)
            
    return sitsds
    
    


def invert(z,sitsds):  
    ii = int(len(z)/2)
        
    eps=[10**float(i) for i in z[:ii]]  
    b=[10**float(i) for i in z[ii:]]
        
    results=[]
    for evid in evid_:
        val = error(evid, [eps,b])
        results.append(val)     
            
    #calc resi
    resi, npts=0,0
    for i, resu in enumerate(results): 
        W=sitsds['events'][evid_[i]][0]
        if not W:
            continue
        for j, env in enumerate(resu[2]):
            Obs=resu[2][j]
            Syn=resu[3][j]
            R=sitsds['stations'][stations[i][j]][0]
            if not R:
                continue
            Esyn=Syn*W*R    
            resi += np.sum((np.log10(Obs) - np.log10(Esyn))**2)
            npts+=len(Obs)
    
 
    gstar=[]
    for i in range(len(k_depth)):
        gstar.append(mcl.gtr_3d_karman(k_depth[i],a_depth[i],eps[i],kappa_depth[i]))
        
    qsc=[(i_*j_)/(2*np.pi*freq_band[0][0]) for i_,j_ in zip(gstar,att_vel)] 
    qi=[i_/(2*np.pi*freq_band[0][0]) for i_ in b]
    
    lamberr=np.linalg.norm(np.diff(np.log10(qsc)))+np.linalg.norm(np.diff(np.log10(qi))) 
    val = (resi / npts) + (lamb**2 * lamberr)

    msg= ('%s %s %s %s')
    logger.debug(msg, val, np.log10(gstar), np.log10(b), np.log10(eps))
    print(val,np.log10(gstar),np.log10(b), np.log10(eps))

    return val
        



def fitness_func(solution, solution_idx):
    global sitsds
    if ga_instance.generations_completed == 0:
        #Create empty dict
        sitsds = {'events': {kk: None for kk in [get_eventid(ii) for ii in events_]}, 
                   'stations': {kk: None for kk in stations_}}
        for kk in sitsds['events']:
                sitsds['events'][kk]=[]
        for kk in sitsds['stations']:
                sitsds['stations'][kk]=[]
        
        #get source density spectrum
        for kk in [get_eventid(ii) for ii in events_]:
            if kk in evid_:
                sitsds['events'][kk].append(startval['events'][kk]['W'][ind_s])
            else:
                sitsds['events'][kk].append(None)
            
        #get site amplification
        for kk in stations_:
            if kk in startval['R'].keys():
                sitsds['stations'][kk].append(startval['R'][kk][ind_s])
            else:
                try:
                    sitsds['events'][kk].append(None) 
                except:
                    continue
    
    if ga_instance.generations_completed % 10 == 0 and ga_instance.generations_completed !=0:
        try:
           best_ind=np.where(ga_instance.best_solutions_fitness == max(ga_instance.best_solutions_fitness))[0][1]
           z = ga_instance.best_solutions[best_ind]
           sitsds = sitesandsource(z)  
        except:
           pass
    
    fitness = 1 / (invert(solution,sitsds))
    return fitness



def mutation_func(offspring, self):
    for offspring_idx in range(offspring.shape[0]):
        mutation_ind=np.array(np.random.choice(range(0, self.num_genes), size=self.mutation_num_genes))
        for gene_idx in mutation_ind:
            offspring[offspring_idx,gene_idx]=np.random.uniform(low=self.gene_space[gene_idx][0],high=self.gene_space[gene_idx][1])
    return offspring



def wrapper():
    past=datetime.datetime.now()
    print('Start inversion')
    
    
    # Set the callback function to the GA instance
    ga_instance.run()
    
    # Get the best solution and its fitness value
    best_solution = ga_instance.best_solution()
    print(best_solution)
    

    ga_instance.plot_fitness(save_dir='fitness.png')
    plt.close()
    ga_instance.plot_new_solution_rate(save_dir='newsol.png')
    plt.close()
    ga_instance.plot_genes(save_dir='genes.png')
    plt.close()
    

    msg= ('\n %s')
    logger.debug(msg, best_solution)
    
    now = datetime.datetime.now()
    print(now-past)
    
    msg= ('\n Time: %s')
    logger.debug(msg, now-past)
    


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    




#Load config file
conf = None
if conf is None:
    conf = 'config.json'
if conf:
    try:
        with open(conf) as f:
            conf = json.load(f, cls=ConfigJSONDecoder)
    except:
        logger.critical(' Error while loading config file')
        exit()


logfile = conf['logfile']
loglevel = conf['loglevel']
log = conf['log']

configure_logging(log, 0, loglevel, logfile)

inv = read_inventory(conf['inv'])
events_ = read_events(conf['events_'])
stream = read(conf['stream'], inventory=inv)
startvals= conf['startvals']


for tr in stream:
    tr.attach_response(inv)
    sens = tr.stats.response.instrument_sensitivity
    tr.data = tr.data / sens.value


with open(startvals) as json_file:
    startval = json.load(json_file)
    

channels = inv.get_contents()['channels']
one_channel = {get_station(ch): ch for ch in channels}
stations_ = list(set(get_station(ch) for ch in channels))


freqs = conf['freqs']
freq_band = get_freqs(**freqs) 

ind_s=np.where(np.array(startval['freq']) == freq_band[0][0])[0][0]
g_start=startval['g0'][ind_s]*1000
b_start=startval['b'][ind_s]

filter_= conf['filter_']
noise_windows=conf['noise_windows']
coda_window=conf['coda_window']
skip=conf['skip']
twin_min=skip.get('coda_window')
smooth=conf['smooth']
smooth_window=conf['smooth_window']
cut_coda=conf['cut_coda']
use_picks=conf['use_picks']
vs=conf['vs']
request_window= conf['request_window']
rho=conf['rho']
fs=conf['fs']
df=freq_band[0][1][1]-freq_band[0][1][0]

rec_radius = conf['rec_radius']
dt=conf['dt']
window_len=int(round(conf['window_len']*1/dt))

lamb = conf['damping']

particles=conf['particles']

v_vel=conf['v_vel']
v_depth=conf['v_depth']
rho_depth=conf['rho_depth']

att_depth=conf['att_depth']


itera=conf['iteration']
popsize=conf['popsize']
parents=conf['parents']
ncpu=conf['ncpu']



k_depth, a_depth, att_vel = [],[],[]
for i in range(len(att_depth)):
    if i == len(att_depth) - 1:
        end_idx = len(v_depth)
    else:
        end_idx = v_depth.index(att_depth[i+1])
    start_idx = v_depth.index(att_depth[i])
    
    if start_idx == end_idx:
        start_idx=start_idx-1
    
    indices_between = list(range(start_idx, end_idx))
    vel=np.mean([v_vel[j] for j in indices_between])
    att_vel.append(vel)
    k_depth.append((2*np.pi*freq_band[0][0])/vel)
    a_depth.append(6/((2*np.pi*freq_band[0][0])/vel))
    



kappa_depth=conf['kappa_depth']

ii_=0
rdm_sc=[]
while ii_ <= particles/int(os.environ['SLURM_CPUS_PER_TASK']) :
    theta_scat=np.arccos(1-2*np.random.random())
    if (np.random.random() < mcl.g_3d_karman(k_depth[0],a_depth[0],kappa_depth[0],theta_scat)):
        rdm_sc.append(theta_scat)
        ii_+=1
        
f = open("scat.dat", "w")
for ii in rdm_sc:
    f.write(f"{ii}\n")
f.close()
        
        
        

lstmin=conf['lstar_bounds'][0]
lstmax=conf['lstar_bounds'][1]


rranges=[(find_eps(lstmax,j), find_eps(lstmin,j)) for j, i in enumerate(att_depth)]
rranges+=[tuple(conf['b_bounds']) for i in att_depth]


z=[[np.random.uniform(low=i[0],high=i[1]) for i in rranges] for i in range(popsize-len(startval['g0']))]

for i, j in enumerate(startval['g0']):
    eps=[find_eps(1/(j*1000),ii) for ii in range(len(att_depth))]
    b=[np.log10(startval['b'][i]) for ii in range(len(att_depth))]
    z.insert(i,list(np.hstack(eps+b)))

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=itera,
                       num_parents_mating=parents,
                       fitness_func=fitness_func,
                       sol_per_pop=popsize,
                       num_genes=len(rranges),
                       initial_population=z,
                       mutation_type=mutation_func,
                       save_solutions=True,
                       save_best_solutions=True,
                       gene_space=rranges,
                       suppress_warnings=True,
                       gene_type=float)


source_depth, rec_depth, rec_dist, stations, evid_ = [], [], [], [], []
energies, maxtimes, ntimes, spicks = [], [], [], []
for freq in freq_band:
    for event in events_:
        evid = get_eventid(event)
        origin = get_origin(event)
        arrivals = origin.arrivals   
    
        station_list_ = []
        T_stat = []
        env_ = []
        station_dist = []
        station_depth = []
        samp_=[]
        R=[]
        spick_=[]
           
        if use_picks:
            station_picks = [i.waveform_id.network_code + str('.') + 
                            i.waveform_id.station_code for i in event.picks] 
        else:
            station_picks = stations_
            
        station_event = set(station_picks).intersection(stations_)
      
                    
        for station in station_event:
            try:
                stat_coord = inv.get_coordinates(one_channel[station], 
                                                       datetime=origin.time)
                dist = get_distance(event, stat_coord)
            except:
                continue
            
            if skip and 'distance' in skip:
                val = skip['distance']
                if val and dist / 1000 > val:
                    continue
                
            if use_picks:
                pick = get_picks(arrivals, station)
                if 'S' not in pick:
                    continue
                spick = pick['S'] - origin.time
            else:
                pick = {"S": origin.time + dist / vs}
                spick = dist / vs
    
               
            st = stream.select(network=station.split('.')[0], 
                               station=station.split('.')[1])
                    
            for tr in st:
                tr.stats.origintime = origin.time
                tr.stats.ponset = pick.get('P')
                tr.stats.sonset = pick.get('S')
            
            st_slice = st.slice(starttime = origin.time + request_window[0], 
                                endtime = origin.time + request_window[1])
    
            if len(st_slice) == 0:
                continue     
                           
            T_, acp, Twin, env = get_T(freq[1], filter_, st_slice, noise_windows, coda_window, 
                           twin_min, smooth, smooth_window, cut_coda, rho, df, fs, dt)
            
            if acp == 0:
                continue  
            else:
                T_stat.append(T_)
                env_.append(env)
                spick_.append(spick)
                
            station_list_.append(station)
            station_dist.append(dist / 1000)
            station_depth.append(inv.select(station=station.split('.')[1])[0][0][0].depth/1000)
                       
        if len(env_) == 0:
            continue    
        
        evid_.append(evid)
        source_depth.append(origin.depth / 1000)
        rec_depth.append(station_depth)
        rec_dist.append(station_dist)
        stations.append(station_list_)
        energies.append(env_)
        spicks.append(spick_)
        maxtimes.append(T_stat)
        ntimes.append(int(max(T_stat)/dt) + 2)
        


if __name__ == '__main__':
    wrapper()

