import os
import h5py
import numpy as np
from sklearn import mixture
from scipy import special
from scipy import optimize
import subprocess
import json
import ivar
import datetime


def default_exp(dp):
    """Function that generates the path to the default data folder, relative to dp"""
    exp_date = datetime.datetime.now().strftime("%Y_%m_%d")
    search_path = os.path.join(dp, exp_date)
    try:
        exp_name = os.listdir(search_path)[-1]
    except:
        # might be last experiment from yesterday
        print "no data for {}, search yesterday's data".format(exp_date)
        exp_date = (datetime.datetime.now() - datetime.timedelta(1)).strftime("%Y_%m_%d")
        search_path = os.path.join(dp, exp_date)
        try:
            exp_name = os.listdir(search_path)[-1]
        except Exception as e:
            print "I tried my best but there is no data in today or yesterday's directories"
    return os.path.join(exp_date, exp_name, 'results.hdf5')


def intersection(A1, m0, m1, s0, s1):
    return (m1*s0**2-m0*s1**2-np.sqrt(s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*np.log((1-A1)/A1)*(s1**2-s0**2))))/(s0**2-s1**2)


def area(A1, m0, m1, s0, s1):
    return np.sqrt(np.pi/2)*((1-A1)*s0+(1-A1)*s0*special.erf(m0/np.sqrt(2)/s0)+A1*s1+A1*s1*special.erf(m1/np.sqrt(2)/s1))


# Normed Overlap for arbitrary cut point
def overlap(xc, A1, m0, m1, s0, s1):
    err0 = (1-A1)*np.sqrt(np.pi/2)*s0*(1-special.erf((xc-m0)/np.sqrt(2)/s0))
    err1 = A1*np.sqrt(np.pi/2)*s1*(special.erf((xc-m1)/np.sqrt(2)/s1)+special.erf(m1/np.sqrt(2)/s1))
    return (err0+err1)/area(A0, A1, m0, m1, s0, s1)


# Relative Fraction in 1
def frac(A1, m0, m1, s0, s1):
    return A1


def dblgauss(x, A1, m0, m1, s0, s1):
    return (1-A1)*np.exp(-(x-m0)**2 / (2*s0**2))/np.sqrt(2*np.pi*s0**2) + A1*np.exp(-(x-m1)**2 / (2*s1**2))/np.sqrt(2*np.pi*s1**2)


def get_iteration_variables(h5file, iterations):
    """Returns a list of variables that evaluate to a list"""
    i_vars = []
    i_vars_desc = {}
    if iterations > 1:
        for i in h5file['settings/experiment/independentVariables/'].iteritems():
            # how to eval numpy functions withoout namespace
            if ivar.is_iterated(i[1]['function']):
                i_vars.append(i[0])
                i_vars_desc[i[0]] = {
                    'description': i[1]['description'][()],
                    'function': i[1]['function'][()],
                }
    return (i_vars, i_vars_desc)


def binomial_error(ns, n):
    """Normal approximation interval, see: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval"""
    alpha = 1-0.682
    z = 1-0.5*alpha
    errs = np.zeros_like(ns)
    # if ns is 0 or n, then use ns = 0.5 or n - 0.5 for the error calculation so we dont get error = 0
    ns[ns==0] = 0.5
    for r in range(len(n)):
        if np.any(ns[r]==n[r].astype('int')):
            ns[ns[r]==n[r].astype('int')] = n[r]-0.5
        if np.any(n[r] == 0):
            print "no loading observed"
            errs[r] = np.full_like(ns[r], np.nan)
        else:
            errs[r] = (z/n[r].astype('float'))*np.sqrt(ns[r].astype('float')*(1.0-ns[r].astype('float')/n[r].astype('float')))
    return errs


def jsonify(data):
    """Prep for serialization."""
    json_data = dict()
    for key, value in data.iteritems():
        if isinstance(value, list):  # for lists
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = jsonify(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data


class QDP:
    """A data processing class for quantizing data.

    cuts format is:
    [
        [0-1 threshold, 1-2 threshold, ... (n-1)-n threshold],  # shot 0
        ...  # shot 1
    ]

    The raw data format is the following:
    [ # list of experiments
        { # experiment dict
            experiment_name: `exp_name`,
            source_file: `source file path`
            # variable with lowest index is the innermost loop
            # other variables/settings can be included in the iteration object,
            # but these must be because they are the actual changed variables
            variable_list: [`var_0`, `var_1`],
            variable_desc: {
                `var_0`: {'description': `description`, 'function': `function`},
                `var_1`: {'description': `description`, 'function': `function`},
            }
            iterations: {
                `iter_key`: {
                    timeseries_data: [ # if camera data time series is length 1
                            [shot_0_timeseries], ... [shot_n-1_timeseries],
                            ...
                    ],
                    signal_data: [ # measurement list
                            [shot_0_signal, ... shot_n-1_signal],
                            ...
                    ],
                    quantized_data: [ # 0, 1, 2, ...
                            [shot_0_quant, ... shot_n-1_quant],
                            ...
                    ],
                    loading: `loading`,
                    retention: `retention`,
                    retention_err: `retention_err`,
                    loaded: `loaded`,
                    variables: {
                        `var_0`: var_0_iter_val,
                        `var_1`: var_1_iter_val,
                    }
                },
                `iter_key`: {
                    timeseries_data: [ # measurement list
                            [shot_0_timeseries], ... [shot_n-1_timeseries],
                            ...
                    ],
                    signal_data: [ # measurement list
                            [shot_0_signal, ... shot_n-1_signal],
                            ...
                    ],
                    quantized_data: [ # 0, 1, 2, ...
                            [shot_0_quant, ... shot_n-1_quant],
                            ...
                    ],
                    loading: `loading`,
                    retention: `retention`,
                    retention_err: `retention_err`,
                    loaded: `loaded`,
                    variables: {
                        `var_0`: var_0_iter_val,
                        `var_1`: var_1_iter_val,
                    }
                }
            }
        },
        ...
    ]
    """
    def __init__(self, base_data_path=""):
        self.experiments = []
        self.cuts = {}
        self.rload = {}
        # set a data path to search from
        self.base_data_path = base_data_path
        # save current git hash
        #self.version = subprocess.check_output(['git', 'describe', '--always']).strip()

    def apply_thresholds(self, cuts=None, exp='all', loading_shot=0):
        """Digitize data with existing thresholds (default) or with supplied thresholds.

        digitization bins are right open, i.e. the condition  for x in bin i b[i-1] <= x < b[i]
        """
        if cuts is None:
            cuts = self.cuts
        # apply cuts to one or all experiments
        if exp == 'all':
            exps = self.experiments
        else:
            exps = self.experiments[exp]
        for e in exps:
            #print e
            for i in e['iterations']:
                meas, shots, rois = np.squeeze(e['iterations'][i]['signal_data']).shape[:3]
                # digitize the data
                quant = np.empty((shots, rois, meas))
                for r in range(rois):
                    for s in range(shots):
                        # first bin is bin 1
                        try:
                            quant[s, r] = np.digitize(np.squeeze(e['iterations'][i]['signal_data'])[:, s, r, 0], cuts[r][s])
                        except:
                            quant[s, r] = np.digitize(np.squeeze(e['iterations'][i]['signal_data'])[:, s, r], cuts[r][s])
                e['iterations'][i]['quantized_data'] = quant.swapaxes(0, 2).swapaxes(1,2)  # to: (meas, shots, rois)
                # calculate loading and retention for each shot
                retention = np.empty((shots, rois))
                reloading = np.empty((shots, rois))
                for r in range(rois):
                    for s in range(shots):
                        retention[s, r] = np.sum(np.logical_and(
                            quant[loading_shot, r],
                            quant[s, r]
                        ), axis=0)
                        reloading[s, r] = np.sum(np.logical_and(
                            np.logical_not(quant[loading_shot, r]),
                            quant[s, r]
                        ), axis=0)
                loaded = np.copy(retention[loading_shot, :])
               
                retention[loading_shot, :] = 0.0
                reloading[loading_shot, :] = 0.0
                #print retention/loaded
                e['iterations'][i]['loading'] = loaded/meas
                e['iterations'][i]['retention'] = retention/loaded
                try:
                    e['iterations'][i]['retention_err'] = binomial_error(retention, loaded)
                except:
                    e['iterations'][i]['retention_err'] = binomial_error(retention[1], loaded)
                e['iterations'][i]['loaded'] = loaded
                e['iterations'][i]['reloading'] = reloading.astype('float')/(meas-loaded)

        return self.get_retention()

    def format_counter_data(self, array, shots, drops, bins):
        """Formats raw 2D counter data into the required 4D format.

        Formats raw 2D counter data with implicit stucture:
            [   # counter 0
                [ dropped_bins shot_time_series dropped_bins shot_time_series ... ],
                # counter 1
                [ dropped_bins shot_time_series dropped_bins shot_time_series ... ]
            ]
        into the 4D format expected by the subsequent analyses"
        [   # measurements, can have different lengths run-to-run
            [   # shots array, fixed size
                [   # roi list, shot 0
                    [ time_series_roi_0 ],
                    [ time_series_roi_1 ],
                    ...
                ],
                [   # roi list, shot 1
                    [ time_series_roi_0 ],
                    [ time_series_roi_1 ],
                    ...
                ],
                ...
            ],
            ...
        ]
        """
        rois, bins = array.shape[:2]
        bins_per_shot = drops + bins  # bins is data bins per shot
        # calculate the number of shots dynamically
        num_shots = int(bins/(bins_per_shot))
        # calculate the number of measurements contained in the raw data
        # there may be extra shots if we get branching implemented
        num_meas = num_shots//shots
        # build a mask for removing valid data
        shot_mask = ([False]*drops + [True]*bins)
        good_shots = shots*num_meas
        # mask for the roi
        ctr_mask = np.array(shot_mask*good_shots + 0*shot_mask*(num_shots-good_shots), dtype='bool')
        # apply mask a reshape partially
        array = array[:, ctr_mask].reshape((rois, num_meas, shots, bins))
        array = array.swapaxes(0, 1)  # swap rois and measurement axes
        array = array.swapaxes(1, 2)  # swap rois and shots axes
        return array

    def get_retention(self, shot=1, fmt='dict'):
        #print self.experiments[0]['iterations'][0]['signal_data'].shape[3]*self.experiments[0]['iterations'][0]['signal_data'].shape[2]
        try:    
            retention = np.empty((
            len(self.experiments),
            len(self.experiments[0]['iterations'].items()),
            self.experiments[0]['iterations'][0]['signal_data'].shape[3]*self.experiments[0]['iterations'][0]['signal_data'].shape[2]
            
        ))
        except:
            retention = np.empty((
            len(self.experiments),
            len(self.experiments[0]['iterations'].items()),
            self.experiments[0]['iterations'][0]['signal_data'].shape[2]  # rois
        ))
        #print self.experiments[0]['iterations'][0]['signal_data'].shape
        err = np.empty_like(retention)
        redX =  np.empty_like(retention)
        FORTX =  np.empty_like(retention)
        redY =  np.empty_like(retention)
        FORTY =  np.empty_like(retention)
        loading = np.empty_like(retention)
        for e, exp in enumerate(self.experiments):
            if len(exp['variable_list']) > 1:
                ivar =np.empty((len(self.experiments),len(exp['variable_list']),len(self.experiments[0]['iterations'].items()),self.experiments[0]['iterations'][0]['signal_data'].shape[3]*self.experiments[0]['iterations'][0]['signal_data'].shape[2]))
                for j in (0,1):
                    for i in exp['iterations']:
                        ivar_name = exp['variable_list']
                        try:
                            redY[e, i] = exp['iterations'][i]['Red_camera_dataY']
                            FORTY[e, i] = exp['iterations'][i]['FORT_camera_dataY']
                            redX[e, i] = exp['iterations'][i]['Red_camera_dataX']
                            FORTY[e, i] = exp['iterations'][i]['FORT_camera_dataX']
                            retention[e, i] = exp['iterations'][i]['retention'][shot]
                            loading[e, i] = exp['iterations'][i]['loading']
                            err[e, i] = exp['iterations'][i]['retention_err'][shot]
                            if ivar_name is not None:
                           
                                ivar[e,j, i] = exp['iterations'][i]['variables'][ivar_name[j]]
                            else:
                                ivar[e, i,j] = 0
                        except IndexError:
                            print "error reading (e,i): ({},{})".format(e, i)
                    # if numpy format is requested return it
                        #print retention
                if fmt == 'numpy' or fmt == 'np':
                    return np.array([ivar, retention, err, loading])
                   # print retention
                else:
                    # if unrecognized return dict format
                    return {
                        'retention': retention,
                        'loading': loading,
                        'error': err,
                        'ivar': ivar,
                        'redX': redX,
                        'redY': redY,
                        'FORTX': FORTX,
                        'FORTY': FORTY
                    }
            if len(exp['variable_list']) == 1:
                ivar_name = exp['variable_list'][0]
                ivar = np.empty_like(retention)
            else:
                ivar = np.empty_like(retention)
                ivar_name = None
            for i in exp['iterations']:
                try:
                    redY[e, i] = exp['iterations'][i]['Red_camera_dataY']
                    FORTY[e, i] = exp['iterations'][i]['FORT_camera_dataY']
                    redX[e, i] = exp['iterations'][i]['Red_camera_dataX']
                    FORTY[e, i] = exp['iterations'][i]['FORT_camera_dataX']
                    retention[e, i] = exp['iterations'][i]['retention'][shot]
                    loading[e, i] = exp['iterations'][i]['loading']
                    err[e, i] = exp['iterations'][i]['retention_err'][shot]
                    if ivar_name is not None:
                        ivar[e, i] = exp['iterations'][i]['variables'][ivar_name]
                    else:
                        ivar[e, i] = 0
                except IndexError:
                    print "error reading (e,i): ({},{})".format(e, i)
        # if numpy format is requested return it
        #print retention
        if fmt == 'numpy' or fmt == 'np':
            return np.array([ivar, retention, err, loading])
        else:
            # if unrecognized return dict format
            return {
                'retention': retention,
                'loading': loading,
                'error': err,
                'ivar': ivar,
                'redX': redX,
                'redY': redY,
                'FORTX': FORTX,
                'FORTY': FORTY
            }

    def get_thresholds(self):
        return self.cuts

    def fit_distribution(self, shot_data, max_atoms=1, hbins=0):
        cut = np.nan
        # use a gaussian mixture model to find initial guess at signal distributions
        gmix = mixture.GaussianMixture(n_components=max_atoms+1)
        gmix.fit(np.array([shot_data]).transpose())
        # order the components by the size of the signal
        indicies = np.argsort(gmix.means_.flatten())
        guess = []
        for n in range(max_atoms+1):
            idx = indicies[n]
            guess.append([
                gmix.weights_[idx],  # amplitudes
                gmix.means_.flatten()[idx],  # x0s
                np.sqrt(gmix.means_.flatten()[idx])  # sigmas
            ])
        # reorder the parameters, drop the 0 atom amplitude
        guess = np.transpose(guess).flatten()[1:]
        # bin the data, default binning is just range([0,max])
        if hbins < 1:
            hbins = range(int(np.max(shot_data))+1)
        hist, bin_edges = np.histogram(shot_data, bins=hbins, normed=True)
        try:
            popt, pcov = optimize.curve_fit(dblgauss, bin_edges[:-1], hist, p0=guess)
            cut = [intersection(*popt)]
            rload = frac(*popt)
        except RuntimeError:
            popt = np.array([])
            pcov = np.array([])
            cut = [np.nan]  # [intersection(*guess)]
            rload = np.nan  # frac(*guess)
        except TypeError:
            print("There may not be enough data for a fit. ( {} x {} )".format(len(bin_edges)-1, len(hist)))
            popt = np.array([])
            pcov = np.array([])
            cut = [np.nan]
            rload = np.nan
        return {
            'hist_x': bin_edges[:-1],
            'hist_y': hist,
            'max_atoms': max_atoms,
            'fit_params': popt,
            'fit_cov': pcov,
            'cuts': cut,
            'guess': guess,
            'rload': rload,
        }

    def generate_thresholds(self, save_cuts=True, exp=0, itr=0, **kwargs):
        """Find the optimal thresholds for quantization of the data."""
        # can drop this check when support is added
        if 'max_atoms' not in kwargs:
            kwargs['max_atoms'] = 1
        elif kwargs['max_atoms'] > 1:
            raise NotImplementedError
        meas, shots, rois = self.experiments[exp]['iterations'][itr]['signal_data'].shape[:3]
        ret_val = {}
        for r in range(rois):
            self.rload[r] = np.zeros(shots)
            ret_val[r] = []
            cuts = []
            for s in range(shots):
                # stored format is (sub_measurement, shot, roi, 1)
                shot_data = self.experiments[exp]['iterations'][itr]['signal_data'][:, s, r, 0]
                ret_val[r].append(self.fit_distribution(shot_data, **kwargs))
                cuts.append(ret_val[r][-1]['cuts'])
                self.rload[r][s] = ret_val[r][-1]['rload']
            self.set_thresholds(cuts, roi=r)
        return ret_val

    def get_settings_from_file(self, h5_settings):
        """Extract any necessary parameters from the file."""
        pass

    def load_data_file(self, filepath=''):
        """Load a data file into the class.

        Additional experiments will be appended to an existing list.
        """
        if not filepath:
            filepath = default_exp(self.base_data_path)
        full_path = os.path.join(self.base_data_path, filepath)
        new_experiments = self.load_hdf5_file(full_path)
        self.experiments += new_experiments

    def load_exp_list(self, exps):
        """manually load an experiment list into the class for analysis.

        Probably only useful for testing.
        """
        self.experiments = exps

    def load_hdf5_file(self, full_filepath, h5file=None):
        """Load experiments from data file into standard format.

        returns list of experiment objects
        """
        if h5file is None:
            h5file = h5py.File(full_filepath)
        # load necessary settings
        self.get_settings_from_file(h5file['settings/'])
        h5_exps = h5file['experiments/'].iteritems()
        exps = []
        # step through experiments
        for e in h5_exps:
            # iterations are the same for all experiments in the same file
            # iteration variables are the same for all experiments in a data file
            iterations = len(e[1]['iterations/'].items())
            ivars, ivar_desc = get_iteration_variables(h5file, iterations)
            exp_data = {
                'experiment_name': os.path.basename(full_filepath),
                'source_file': full_filepath,
                'source_filename': os.path.basename(full_filepath),
                'source_path': os.path.dirname(full_filepath),
                'variable_list': ivars,
                'variable_desc': ivar_desc,
                'iterations': {}
            }
            # step through iterations
            for i in e[1]['iterations/'].iteritems():
                exp_data['iterations'][int(i[0])] = self.process_iteration(i[1])
            exps.append(exp_data)
        h5file.close()
        return exps

    def process_iteration(self, h5_iter):
        iteration_obj = {
            'variables': {},
            'timeseries_data': [],  # cant be numpy array because of different measurement number
            'signal_data': [],  # cant be numpy array because of different measurement number
            'Red_camera_dataX': [],
            'FORT_camera_dataX': [],
            'Red_camera_dataY': [],
            'FORT_camera_dataY': []
        }
        # copy variable values over
        for v in h5_iter['variables'].iteritems():
            iteration_obj['variables'][v[0]] = v[1][()]
        # copy measurement values over
        for m in h5_iter['measurements/'].iteritems():
            data = self.process_measurement(m[1], iteration_obj['variables'])
            try:
                iteration_obj['timeseries_data'].append(data['timeseries_data'])
            except:
                pass
            iteration_obj['signal_data'].append(data['signal_data'])
            if np.isnan(data['Red_camera_dataX']):
                pass
            else:
                iteration_obj['Red_camera_dataX'].append(data['Red_camera_dataX'])
            if np.isnan(data['FORT_camera_dataX']):
                pass
            else:
                 iteration_obj['FORT_camera_dataX'].append(data['FORT_camera_dataX'])
            if np.isnan(data['Red_camera_dataY']):
                pass
            else:
                 iteration_obj['Red_camera_dataY'].append(data['Red_camera_dataY'])
            if np.isnan(data['FORT_camera_dataY']):
                pass
            else:
                 iteration_obj['FORT_camera_dataY'].append(data['FORT_camera_dataY'])
        # cast as numpy arrays, compress sub measurements
        iteration_obj['signal_data'] = np.concatenate(iteration_obj['signal_data'])
        iteration_obj['Red_camera_dataX'] = np.nanmean(iteration_obj['Red_camera_dataX'])
        iteration_obj['FORT_camera_dataX'] = np.nanmean(iteration_obj['FORT_camera_dataX'])
        iteration_obj['Red_camera_dataY'] = np.nanmean(iteration_obj['Red_camera_dataY'])
        iteration_obj['FORT_camera_dataY'] = np.nanmean(iteration_obj['FORT_camera_dataY'])
        iteration_obj['timeseries_data'] = np.mean(iteration_obj['timeseries_data'])
        return iteration_obj

    def process_measurement(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of timeseries_data for each shot.
        """
        try:
            sum_data = self.process_analyzed_counter_data(measurement, variables)
            ts_data = self.process_raw_counter_data(measurement, variables)
        except:
            sum_data = self.process_analyzed_camera_data(measurement, variables)
            Red_data = self.process_analyzed_camera_data_Red(measurement, variables)
            FORT_data = self.process_analyzed_camera_data_FORT(measurement, variables)
            ts_data = []
        return {'timeseries_data': ts_data, 'signal_data': sum_data, 'Red_camera_dataX': Red_data[0],'Red_camera_dataY': Red_data[1] ,'FORT_camera_dataX': FORT_data[0],'FORT_camera_dataY': FORT_data[1]}

    
    def process_raw_camera_data(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of camera_data for each shot.
        """
        total_shots = 0
        array = []
        for x in (0,1,2):
            array.append([measurement['data/Andor_4522/shots/'+str(x)].value])
            total_shots += 1
       # total_shots = array.shape[1]
        return self.format_counter_data(array, total_shots)

    def process_raw_counter_data(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of timeseries_data for each shot.
        """
        drop_bins = variables['throwaway_bins']
        meas_bins = variables['measurement_bins']
        array = measurement['data/counter/data'].value
        total_shots = array.shape[1]/(drop_bins + meas_bins)
        if total_shots > 2:
            print("Possibly too many shots, analysis might need to be updated")
        return self.format_counter_data(array, total_shots, drop_bins, meas_bins)

    def process_analyzed_camera_data(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of camera_data for each shot.
        """
        # stored format is (sub_measurement, shot, roi, 1)
        # last dimension is the "roi column", an artifact of the camera roi definition
        return np.array([measurement['analysis/squareROIsums'].value])
    def process_analyzed_camera_data_FORT(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of camera_data for each shot.
        """
        # stored format is (sub_measurement, shot, roi, 1)
        # last dimension is the "roi column", an artifact of the camera roi definition
        return np.array([measurement['data/camera_data/15102504/stats/X0'].value,measurement['data/camera_data/15102504/stats/Y0'].value])
    def process_analyzed_camera_data_Red(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of camera_data for each shot.
        """
        # stored format is (sub_measurement, shot, roi, 1)
        # last dimension is the "roi column", an artifact of the camera roi definition
        return np.array([measurement['data/camera_data/16483678/stats/X0'].value,measurement['data/camera_data/16483678/stats/Y0'].value])
 
 
    def process_analyzed_counter_data(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of timeseries_data for each shot.
        """
        # stored format is (sub_measurement, shot, roi, 1)
        # last dimension is the "roi column", an artifact of the camera roi definition
        return measurement['analysis/counter_data'].value
 
    def save_experiment_data(self, filename_prefix='data', path=None):
        """Saves data to files with the specified prefix."""
        if path is None:
            path = self.experiments[0]['source_path']
        self.save_json_data(filename_prefix=filename_prefix, path=path)
        self.save_retention_data(filename_prefix=filename_prefix, path=path)

    def save_json_data(self, filename_prefix='data', path=None):
        if path is None:
            path = self.experiments[0]['source_path']
        with open(os.path.join(path, filename_prefix + '.json'), 'w') as f:
            json.dump({
                    'data': map(jsonify, self.experiments),
                    'metadata': {'version': self.version},
                },
                f
            )

    def save_retention_data(self, filename_prefix='data', path=None, shot=1):
        if path is None:
            path = self.experiments[0]['source_path']
        try:
            np.save(
                os.path.join(path, filename_prefix + '.npy'),
                self.get_retention(shot=shot, fmt='numpy'),
                allow_pickle=False
            )
        except KeyError:
            print('Retention data has not been processed.  Not saving.')

    def set_thresholds(self, cuts, roi=0):
        self.cuts[roi] = cuts
