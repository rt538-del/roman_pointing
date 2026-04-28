import os
from pathlib import Path

import numpy as np
import scipy.optimize as optimize
import astropy.units as u

from roman_pointing.roman_pointing import calcRomanAngles, getL2Positions

SLEW_SETTLE_PATH = Path(__file__).resolve().parents[2] / "Notebooks" / "SlewSettle.ecsv"


def _linear(x, m, b):
    return m * x + b


def _fit_slew_model():
    """Fit a linear model to the GSFC SlewSettle data (angles > 6.5 deg)."""
    angles = np.loadtxt(SLEW_SETTLE_PATH, skiprows=11, usecols=0, delimiter=",")
    times  = np.loadtxt(SLEW_SETTLE_PATH, skiprows=11, usecols=1, delimiter=",")
    popt, _ = optimize.curve_fit(_linear, angles[angles > 6.5], times[angles > 6.5])
    return popt


# Fit once at import time so every call to compute_slew_time is fast.
_SLEW_POPT = _fit_slew_model()


def compute_slew_time(coord1, coord2, ts):
    """Estimate slew-and-settle time between two targets at each time step.

    Replicates the logic from Roman_SlewTime_Calculator.ipynb. Slew angle is
    driven by the larger of |Δpitch| and |Δyaw| (yaw wraps at ±180 deg).
    Time is estimated by interpolating the GSFC SlewSettle curve.

    Args:
        coord1 (astropy.coordinates.SkyCoord): First target in BarycentricMeanEcliptic.
        coord2 (astropy.coordinates.SkyCoord): Second target in BarycentricMeanEcliptic.
        ts (astropy.time.Time): Array of times over which to compute slew.

    Returns:
        tuple:
            - slew_angle (astropy.units.Quantity): Max slew angle (deg) at each time step.
            - slew_time_s (numpy.ndarray): Estimated slew+settle time (seconds) at each step.
    """
    roman_pos = getL2Positions(ts)

    _, yaw1, pitch1, _ = calcRomanAngles(coord1, ts, roman_pos)
    _, yaw2, pitch2, _ = calcRomanAngles(coord2, ts, roman_pos)

    yaw_diff   = np.abs(yaw1 - yaw2).to(u.deg)
    pitch_diff = np.abs(pitch1 - pitch2).to(u.deg)
    yaw_diff[yaw_diff > 180 * u.deg] = (360 * u.deg) - yaw_diff[yaw_diff > 180 * u.deg]

    slew_angle = np.max([yaw_diff.value, pitch_diff.value], axis=0) * u.deg
    slew_time_s = _linear(slew_angle.value, *_SLEW_POPT)

    return slew_angle, slew_time_s


def get_cache_dir():
    """
    Finds the home directory for the system and generates a cache dir as needed

    Returns:
        str:
            Path to cache directory
    """

    # POSIX system
    if os.name == "posix":
        if "HOME" in os.environ:
            homedir = os.environ["HOME"]
        else:
            raise OSError("Could not find POSIX home directory")
    # Windows system
    elif os.name == "nt":
        # msys shell
        if "MSYSTEM" in os.environ and os.environ.get("HOME"):
            homedir = os.environ["HOME"]
        # network home
        elif "HOMESHARE" in os.environ:
            homedir = os.environ["HOMESHARE"]
        # local home
        elif "HOMEDRIVE" in os.environ and "HOMEPATH" in os.environ:
            homedir = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"])
        # user profile?
        elif "USERPROFILE" in os.environ:
            homedir = os.path.join(os.environ["USERPROFILE"])
        # something else?
        else:
            try:
                import winreg as wreg

                shell_folders = (
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
                )
                key = wreg.OpenKey(wreg.HKEY_CURRENT_USER, shell_folders)
                homedir = wreg.QueryValueEx(key, "Personal")[0]
                key.Close()
            except Exception:
                # try home before giving up
                if "HOME" in os.environ:
                    homedir = os.environ["HOME"]
                else:
                    raise OSError("Could not find Windows home directory")
    else:
        # some other platform? try HOME to see if it works
        if "HOME" in os.environ:
            homedir = os.environ["HOME"]
        else:
            raise OSError("Could not find home directory on your platform")

    assert os.path.isdir(homedir) and os.access(homedir, os.R_OK | os.W_OK | os.X_OK), (
        f"Identified {homedir} as home directory, but it does not exist "
        "or is not accessible/writeable"
    )

    path = os.path.join(homedir, ".corgi", "cache")
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except PermissionError:
            print("Cannot create directory: {}".format(path))

    # ensure everything worked out
    assert os.access(path, os.F_OK), "Directory {} does not exist".format(path)
    assert os.access(path, os.R_OK), "Cannot read from directory {}".format(path)
    assert os.access(path, os.W_OK), "Cannot write to directory {}".format(path)
    assert os.access(path, os.X_OK), "Cannot execute directory {}".format(path)

    return path
