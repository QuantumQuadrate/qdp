import pytest
import os.path
import h5py
import numpy as np

from QDP import qdp

test_file_path = os.path.dirname(os.path.abspath(__file__))
filename = 'test_{}.hdf5'


def between(x, l, h):
    assert((x > l) and (x < h))


def base_hdf5_file(data_func, n_exps=1, n_iters=10, n_meas=200, n_shots=2, drop_bins=3, meas_bins=10, rois=(1, 1)):
    # use a memory mapped file for testing
    i = 0
    while i < 10:
        try:
            f = h5py.File(os.path.join(test_file_path, filename.format(i)), "w", driver='core')
        except:
            i += 1
        else:
            break

    i_var = 'dummy'
    i_var_func = 'arange({})'.format(n_iters)
    i_var_dset = 'settings/experiment/independentVariables/{}/{}'
    f.create_dataset(i_var_dset.format(i_var, 'function'), data=i_var_func)
    f.create_dataset(i_var_dset.format(i_var, 'description'), data='dummy iterator')
    e = f.create_group('experiments')
    for i in range(n_exps):
        e1 = e.create_group(str(i))
        iter = e1.create_group('iterations')
        for j in range(n_iters):
            i1 = iter.create_group(str(j))
            v = i1.create_group('variables')
            v.create_dataset(i_var, data=j)
            v.create_dataset('throwaway_bins', data=drop_bins)
            v.create_dataset('measurement_bins', data=meas_bins)
            m = i1.create_group('measurements')
            for k in range(n_meas):
                meas = m.create_group(str(k))
                data = meas.create_group('data')
                analysis = meas.create_group('analysis')
                ctr = data.create_group('counter')
                # raw data
                n_rois = rois[0]*rois[1]
                ts_data = np.empty((n_rois, (drop_bins+meas_bins)*n_shots))
                sig_data = np.empty((1, n_shots, rois[0]*rois[1], 1))
                for r in range(n_rois):
                    cnt_data = data_func(meas_bins, n_shots)
                    # sum up the raw data
                    sig_data[0, :, r, 0] = cnt_data.sum(axis=1)
                    cnt_data = np.insert(cnt_data, 0, [[0]*n_shots for b in range(drop_bins)], axis=1)
                    ts_data[r] = cnt_data.flatten()
                ctr.create_dataset('data', dtype='uint64', data=ts_data)
                analysis.create_dataset('counter_data', dtype='uint32', data=sig_data)
    return f


@pytest.fixture()
def test_hdf5_file():
    # use a constant signal function for logical testing
    def const_func(meas_bins, shots):
        return np.array([meas_bins*[1] for s in range(shots)])

    def teardown():
        base_hdf5_file.close()

    return base_hdf5_file(const_func)


@pytest.fixture()
def test_hdf5_file_binary():
    # use a random binary signal function for fit testing
    meas_bins = 10
    # total integrated signal
    signal = 20.0/meas_bins
    background = 5.0/meas_bins
    event_prob = 0.3

    def binary_func(meas_bins, shots):
        atom = np.random.random(1) < event_prob
        total_signal = (signal*atom) + background
        return np.random.poisson(lam=total_signal, size=(shots, meas_bins))

    def teardown():
        base_hdf5_file.close()

    return base_hdf5_file(binary_func)


def test_basic(test_hdf5_file):
    # test that the raw time series data is cut up correctly
    q = qdp.QDP(test_file_path)
    exps = q.load_hdf5_file(os.path.join(test_file_path, filename), h5file=test_hdf5_file)
    # check that the independent variable was correctly identified
    assert(exps[0]['variable_list'][0] == 'dummy')
    assert(len(exps[0]['variable_list']) == 1)
    # check that the data was copied correctly
    for e in exps:
        for i in e['iterations']:
            assert(e['iterations'][i]['variables']['dummy'][()] == int(i))
            for m in e['iterations'][i]['timeseries_data']:
                for t in m:
                    all(item == 1 for item in t)
                    assert(len(t) > 0)
            for m in e['iterations'][i]['signal_data']:
                assert(item == 10 for item in m)


def test_binary_cuts(test_hdf5_file_binary):
    # test that the automatic cuts function is ok
    q = qdp.QDP(test_file_path)
    exps = q.load_hdf5_file(os.path.join(test_file_path, filename), h5file=test_hdf5_file_binary)
    # manually load in an experiment list
    q.load_exp_list(exps)
    # find cuts
    ret_val = q.generate_thresholds()
    # retrieve cuts
    cuts = q.get_thresholds()
    between(cuts[0][0][0], 8, 15)  # check that the shot 0 0-1 atom threshold is reasonable (region 0)
    between(cuts[0][1][0], 8, 15)  # check that the shot 1 0-1 atom threshold is reasonable (region 0)

    # check that fits are reasonable
    # amplitudes
    between(ret_val[0][0]['fit_params'][0], 0.2, 0.4)  # 30% event
    between(ret_val[0][1]['fit_params'][0], 0.2, 0.4)  # 30% event
    # means
    between(ret_val[0][0]['fit_params'][1], 4, 6)  # 70% no event
    between(ret_val[0][0]['fit_params'][2], 23, 27)  # 30% event
    between(ret_val[0][1]['fit_params'][1], 4, 6)  # 70% no event
    between(ret_val[0][1]['fit_params'][2], 23, 27)  # 30% event

    # digitization
    q.apply_thresholds()
    ret_val = q.get_retention()
    retention = ret_val['retention']
    err = ret_val['error']
    # ivar = ret_val['ivar']
    # shape is (experiments, iterations, rois)
    assert(retention.shape == (1, 10, 1))
    assert(retention[0, 0] > 0.9)
    all(e > 0 for e in err[0])

    # try saving data
    q.save_experiment_data()
