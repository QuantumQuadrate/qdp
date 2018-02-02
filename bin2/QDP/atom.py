import numpy as np
from scipy import constants


def release_recapture(drop, T_uk, U_mk, wr_um, zr_um, fr_khz, fa_khz, gravity=True, n=100, m_au=86.909):
    """A monte carlo implementation of a release and recapture experiment.

    See: C. Tuchendler et al. Phys. Rev. A 78, 033425 (2008)

    drop: free expansion in us
    T: temperature in uK
    U: trap depth in mK
    wr: trap laser waist in um
    zr: Rayliegh length in um
    fr_khz: radial oscillation freq in khz
    fa_khz: axial oscillation freq in khz
    gravity: turn on/off gravity (along radial direction)
    n: number of trials
    m_au: mass in a.u. (default 87Rb)
    """
    kb = constants.Boltzmann
    m_kg = m_au*constants.atomic_mass

    # distance in um
    sigma_x = atom_distribution_sigma_um(T_uk, fr_khz, m_au)
    sigma_z = atom_distribution_sigma_um(T_uk, fa_khz, m_au)
    # velocities in um/us = m/s
    sigma_v = np.sqrt(kb*T_uk*1E-6/(m_kg))
    # print sigma_x
    # print sigma_z
    # print sigma_v
    g = 9.8*1E-6  # um/us^2

    # generate random positions and velocities
    x, y = np.random.normal(scale=sigma_x, size=(2, n))
    z = np.random.normal(scale=sigma_z, size=n)
    vx, vy, vz = np.random.normal(scale=sigma_v, size=(3, n))
    # calculate final position after time drop
    x_f = x + vx*drop
    y_f = y + vy*drop
    z_f = z + vz*drop
    if gravity:
        y_f = - 0.5*g*np.power(drop, 2)

    # calculate initial and final energies
    PE_i = -kb*U_mk*1E-3*gaussian_beam(x, y, z, wr_um, zr_um)
    PE_f = -kb*U_mk*1E-3*gaussian_beam(x_f, y_f, z_f, wr_um, zr_um)
    KE = 0.5*m_kg*(np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2))
    if gravity:
        KE_f = 0.5*m_kg*(np.power(vx, 2) + np.power(vy - g*drop, 2) + np.power(vz, 2))
    else:
        KE_f = KE

    # only consider initial states that were trapped
    trapped_i = (PE_i + KE) < 0.0
    trapped_f = (PE_f + KE_f) < 0.0

    # measure probability of remaining trapped
    n_act = np.sum(trapped_i)
    n_recaptured = np.sum(np.logical_and(trapped_i, trapped_f))
    return float(n_recaptured)/n_act


def gaussian_beam(x, y, z, wr, zr):
    """Gaussian beam with amplitude 1"""
    return np.exp(-2.*(np.power(x, 2) + np.power(x, 2))/(np.power(wr, 2)*(1+np.power(z/zr, 2))))/(1+np.power(z/zr, 2))


def atom_distribution_sigma_um(T_uk, f_khz, m_au):
    """Sigma for atom cloud in FORT."""
    return 1E3*np.sqrt(constants.Boltzmann*T_uk*1E-6/(m_au*constants.atomic_mass))/(2*np.pi*f_khz)


if __name__ == '__main__':
    drops = np.arange(0, 40, 10)
    T_uk = 50.0
    U_mk = 1.5
    wr_um = 2.5
    zr_um = 17.9
    fr_khz = 48.0
    fa_khz = 4.7
    for d in drops:
        print(release_recapture(d, T_uk, U_mk, wr_um, zr_um, fr_khz, fa_khz, n=10000))
