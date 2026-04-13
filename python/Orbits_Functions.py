""" Functions for orbital calculations - AERO351 """

from math import floor
import requests
import scipy
from scipy.integrate import solve_ivp
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import json
import os


def julian_date(year, month, day, hour, minute, second=0, use_usno=True):
    """ Calculate the Julian Date for a given date and time. """

    if use_usno:
        if second in (0, 0.0, None):
            time_str = f"{hour:02d}:{minute:02d}"
        else:
            # If you want seconds, keep them as two digits (or pass a float if needed)
            # For integer seconds:
            time_str = f"{hour:02d}:{minute:02d}:{int(second):02d}"

        BASE = "https://aa.usno.navy.mil/api/juliandate"

        params = {
            "date": f"{year}-{month}-{day}",  # leading zeros not required
            "time": time_str,                 # UT1/UTC
            "era": "AD",                       # "AD" or "BC"
        }

        r = requests.get(BASE, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()

        d = payload.get("data")
        # Normalize shape (list or dict)
        if isinstance(d, list):
            if not d:
                raise ValueError("USNO returned an empty data list.")
            d = d[0]
        elif not isinstance(d, dict):
            raise ValueError(f"Unexpected 'data' shape: {type(d)}. Full payload: {payload}")

        if "jd" not in d:
            raise KeyError(f"'jd' not found in data record. Full record: {d}")

        print(f"{d["jd"]} (from USNO)")

    if month <= 2:
        year -= 1
        month += 12

    J0 = 367 * year - floor((7 * (year + floor((month + 9) / 12))) / 4) + floor((275 * month) / 9) + day + 1721013.5
    JD = J0 + (hour / 24) + (minute / 1440) + (second / 86400)
    return JD

def sidereal_time(angle, year, month, day, hour, minute, second):
    """ Calculate the Local Sidereal Time in degrees, given the longitude and Julian Date.
    Longitude is in degrees, positive eastward from Greenwich and negative westward.
    jd is the julian date in days.
    """

    j0 = julian_date(year, month, day, 0, 0, 0, False)
    T0 = (j0 - 2451545.0) / 36525
    GWST = 100.46061837 + 36000.770053608 * T0 + 0.000387933 * T0**2 - (2.58 * 10**-8)*T0**3
    while GWST < 0:
        GWST += 360.0
    while GWST >= 360.0:
        GWST -= 360.0
    
    UT = hour + (minute / 60) + (second / 3600)
    GWST += 360.98564724 * (UT / 24)
    while GWST < 0:
        GWST += 360.0
    while GWST >= 360.0:
        GWST -= 360.0

    LST = GWST + angle
    if LST < 0:
        return LST + 360.0
    if LST >= 360.0:
        return LST - 360.0

    return LST

def ODEprimer(rvec, vvec, TSpan, teval=None, mu=398600):

    y0 = np.array([rvec[0], rvec[1], rvec[2], vvec[0], vvec[1], vvec[2]])

    def twobodyeq(t, y, mu):
        """"
        twobodyeq - Returns the derivative of the state space variables.
        INPUTS:
        t - time (not used in this function but required by solve_ivp)
        y - A 6x1 array of the position and velocity of a particle in 3D space.
        
        OUTPUTS:
        ydot - The derivative of y for the two-body problem.
        """

        _ = t

        r = np.sqrt(y[0]**2 + y[1]**2 + y[2]**2)

        y_dot = [0] * 6

        y_dot[0] = y[3]
        y_dot[1] = y[4]
        y_dot[2] = y[5]
        y_dot[3] = (-mu/r**3) * y[0]
        y_dot[4] = (-mu/r**3) * y[1]
        y_dot[5] = (-mu/r**3) * y[2]

        return y_dot
    if teval is not None:
        sol = solve_ivp(twobodyeq, TSpan, y0, rtol=1e-8, atol=1e-8,
                        t_eval=teval, args=(mu,))
    else:
        sol = solve_ivp(twobodyeq, TSpan, y0, rtol=1e-8, atol=1e-8,
                        args=(mu,))
    return sol

def burn_eq(t, y, T, Isp, mu, in_v_dir=True):
    """
    Equations of motion for a spacecraft burn
    
    State y = [rx, ry, rz, vx, vy, vz, m]
    """
    g0 = 9.80665
    
    rvec = y[0:3]
    vvec = y[3:6]
    m = y[6]
    
    r = np.linalg.norm(rvec)
    v = np.linalg.norm(vvec)
    
    a_g = -mu*rvec/r**3
    
    if m > 0.0 and v > 0.0 and T > 0.0:
        v_hat = vvec/v
        a_T = (T/m) * v_hat/1000
        mdot = -T/(Isp*g0)
    else:
        a_T = np.zeros(3)
        mdot = 0.0
    
    rdot = vvec
    if in_v_dir == True:
        vdot = a_g + a_T
    else:
        vdot = a_g - a_T
    return np.hstack((rdot, vdot, mdot))

def burn_propagate(t_span, y0, T, Isp, in_v_dir=True, mu=398600,
                   t_eval=None, rtol=1e-9, atol=1e-9):
    """
    -------------
    Inputs:
    t_span  = [initial time, final time]
    y0      = initial state vector hstack((r0, v0, m))
    T       = Thrust magnitude
    Isp     = Specific impules
    mu      = Grav Parameter
    t_eval  = Specification for rk45's evaluation
    -------------
    Outputs:
    sol     = Result of the ODE
                0:2 - position
                3:5 - velocity
                6   - mass
    """
    sol = solve_ivp(fun=lambda t, y: burn_eq(t, y, T, Isp, mu, in_v_dir),
                    t_span=t_span, y0=y0, method="RK45",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    return sol  

def plot_orbit(y0, label, color, show=True):
    """
    Plot the orbit of a satellite given the solution from ODEprimer

    Parameters
    ----------
    sol : solution from ODEprimer

    Returns
    -------
    None
    """
    
    if not isinstance(y0, (list, tuple)):
        y0 = [y0]
        
    light = pv.Light()
    light.set_direction_angle(30, -20)
    
    ps = pv.Plotter(lighting='none')
    cubemap = pv.examples.download_cubemap_space_4k()
    _ = ps.add_actor(cubemap.to_skybox())
    ps.set_environment_texture(cubemap, is_srgb=True, resample=1 / 64)
    ps.add_light(light)
    
    earth = pv.examples.load_globe()
    earth.points *= .000001
    tex = pv.examples.load_globe_texture()
    ps.add_mesh(earth, texture=tex, smooth_shading=True)
    
    label_points = []
    label_texts = []
    
    for y, label, color in zip(y0,label,color):
        R = y[:3, :].T
    
        curve = pv.Spline(R, n_points=len(R))
        ps.add_mesh(curve, color=color, line_width=3, render_lines_as_tubes=True)
        
        label_points.append(R[0])
        label_texts.append(label)
        
    try:
        ps.add_point_labels(np.array(label_points),
                           label_texts,
                           text_color="white",
                           point_size=12,
                           font_size=14,
                           shape=None,
                           always_visible=True
                           )
    except Exception:
        pass
    
    
    
    ps.add_axes()
    ps.camera.zoom(1.2)
    
    ps.open_gif("orbit_animation.gif")
    for angle in range(360):
        ps.camera.Azimuth(1)
        ps.render()
        ps.write_frame()
    if show:
        ps.show()
    else:
        ps.close()
    
def animate_orbits(y0, tot_time, labels=None, save=None, max_trail=None,
                   framerate=30, trail=False):
    if not isinstance(y0, (list, tuple)):
        y0 = [y0]
    n = len(y0)
    
    # Angular Velocity of the Earth
    w_earth = 0.004167  # deg/sec
    
    if labels is None:
        for i in range(1,n):
            labels[i] = [f"sat {i+1}"]
    
    positions = []
    for y in y0:
        R = y[:3, :].T
        positions.append(R)
        
    max_steps = max(len(R) for R in positions)
    dt = float(tot_time) / max(1, max_steps - 1)
        
    light = pv.Light()
    light.set_direction_angle(30, -20)
    
    p = pv.Plotter(lighting='none')
    cubemap = pv.examples.download_cubemap_space_4k()
    _ = p.add_actor(cubemap.to_skybox())
    p.set_environment_texture(cubemap, is_srgb=True, resample=1 / 64)
    p.add_light(light)
    p.add_axes()
    
    earth = pv.examples.load_globe()
    earth.points *= .000001
    tex = pv.examples.load_globe_texture()
    earth_mesh = p.add_mesh(earth, texture=tex, smooth_shading=True)
    
    colors = ["yellow", "deepskyblue", "tomato", "lime", "magenta", "orange"]
    sat_r = []
    trails = []
    
    for i, R in enumerate(positions):
        color = colors[i % len(colors)]

        orbit_line = pv.Spline(R, n_points=len(R))
        p.add_mesh(orbit_line, color="yellow", line_width=.5, render_lines_as_tubes=True)

        sat = pv.PolyData(R[0:1])
        p.add_mesh(sat, color=color, render_points_as_spheres=True, point_size=8)
        sat_r.append(sat)
        
        trail = pv.PolyData()
        trail.points = R
        trail.lines = np.array([1, 0], dtype=np.int64)
        p.add_mesh(trail, color=color, line_width=2, render_lines_as_tubes=True)
        trails.append(trail)
        
        try:
            p.add_point_labels(R, [labels[i]], always_visible=True, color='white', shape=None, font_size=14)
        except Exception:
            pass
    p.camera.zoom(.75)
    
    if save:
        p.open_movie(save, framerate=framerate)

    for k in range(max_steps):
        earth_mesh.rotate_z(w_earth*dt)
        for i in range(len(positions)):
            R = positions[i]
            idx = min(k, len(R) - 1)

            sat_r[i].points = R[idx:idx+1]
            sat_r[i].Modified()

            if max_trail is None:
                start = 0
            else:
                start = max(0, idx + 1 - max_trail)

            pts = R[start:idx+1]
            trails[i].points = pts
            line = np.concatenate([[len(pts)], np.arange(len(pts), dtype=np.int64)])
            trails[i].lines = line
            trails[i].Modified()
            
        p.render()
        if save:
            p.write_frame()

    if save:
        p.close()
    else:
        p.show()

def trueanom2pos(nu, ecc, h, mu=398600):
    """
    Description of function:
        Convert from true anomaly to position vector

    Args:
        nu (float): True anomaly [rad]
        ecc (float): Eccentricity
        h (float): Angular momentum [km^2/s]
        mu (int, optional): Gravitational parameter. Defaults to 398600.

    Returns:
        r - position vector [km]
    """
    r = (h**2 / mu) * (1 / (1 + ecc * np.cos(nu)))
    return r

def ECI2COEs(y0, mu=398600):
    """
    Description of function:
        Convert from ECI to COEs

    Args:
        y0 (array): An array carrying your state vector, including r0 and v0, should be in units of km and km/s
        mu (int, optional): Gravitational parameter. Defaults to 398600.

    Returns:
        h - angular momentum vector [km^2/s]
        ecc - eccentricity vector
        RAAN - Right Ascension of the Ascending Node [rad]
        argp - argument of perigee [rad]
        nu - true anomaly [rad]
        inc - inclination [rad]
    """
    r_vec = np.array(y0[:3], dtype=float)
    v_vec = np.array(y0[3:], dtype=float)
    distance = np.linalg.norm(r_vec)
    speed = np.linalg.norm(v_vec)
    v_r = np.dot(r_vec, v_vec) / distance
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    k = np.array([0, 0, 1])
    N_vec = np.cross(k, h_vec)
    N = np.linalg.norm(N_vec)
    
    eps = 1e-10
    
    if N < eps:
        RAAN = 0
    else:
        if N_vec[1] >= 0:
            RAAN = np.arccos(N_vec[0] / N)
        else:
            RAAN = 2 * np.pi - np.arccos(N_vec[0] / N)
    ecc_vec = (1/mu) * ((speed**2-(mu/distance))*r_vec - v_r * distance * v_vec)
    ecc = np.linalg.norm(ecc_vec)
    if N < eps or ecc < eps:
        argp = 0.0
    else:
        if ecc_vec[2] >= 0:
            argp = np.arccos(np.dot(N_vec, ecc_vec) / (N * ecc))
        else:
            argp = 2 * np.pi - np.arccos(np.dot(N_vec, ecc_vec) / (N * ecc))
    if v_r >= 0:
        TA = np.arccos(np.dot(ecc_vec, r_vec) / (ecc * distance))
    else:
        TA = 2 * np.pi - np.arccos(np.dot(ecc_vec, r_vec) / (ecc * distance))
    inc = np.arccos(h_vec[2] / h)
    return h_vec, ecc_vec, inc, RAAN, argp, TA

def COEs2ECI(h, ecc, inc, RAAN, argp, TA, mu=398600):
    
    r = h**2 / (mu * (1 + ecc * np.cos(TA)))
    r_PQW = r * np.array([np.cos(TA), np.sin(TA), 0])
    v_PQW = (mu / h) * np.array([-np.sin(TA), ecc + np.cos(TA), 0])
    
    
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(argp), np.sin(argp)
    
    Q = np.array([
        [cO*co-sO*ci*so, -sO*ci*co-cO*so,  sO*si],
        [cO*ci*so+sO*co,  cO*ci*co-sO*so, -cO*si],
        [         si*so,           si*co,     ci]
    ])
    
    r_eci = Q @ r_PQW
    v_eci = Q @ v_PQW
    return r_eci, v_eci

def bisection(F, z_low, z_high, tol=1e-8, max_iter=100):
    """
    Basic bisection method for scalar root finding.
    F       : function to find root of
    z_low   : lower bound of bracket
    z_high  : upper bound of bracket
    tol     : tolerance for convergence
    """
    f_low = F(z_low)
    f_high = F(z_high)

    # Ensure the bracket actually contains a root
    if f_low * f_high > 0:
        raise ValueError("Bisection error: f(z_low) and f(z_high) must have opposite signs.")

    for i in range(max_iter):
        z_mid = 0.5 * (z_low + z_high)
        f_mid = F(z_mid)

        if abs(f_mid) < tol or abs(z_high - z_low) < tol:
            return z_mid, i  # root, iterations

        # Replace one bound
        if f_low * f_mid < 0:
            z_high, f_high = z_mid, f_mid
        else:
            z_low, f_low = z_mid, f_mid

    raise RuntimeError("Bisection did not converge within max_iter")

def stumpff_S(z):
    """
    Uses the piecewise trig function for the S stumpff equation
    """
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z)))/(np.sqrt(z))**3
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/(np.sqrt(-z))**3
    else:
        return 1/6

def stumpff_C(z):
    """
    Uses the piecewise trig function for the C stumpff equation
    """
    if z > 0:
        return (1-np.cos(np.sqrt(z)))/z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1)/(-z)
    else:
        return 1/2

def UniversalAnom_findr(X0, a, r0, vr, dt, mu=398600):
    """
    Takes an initial guess of the Universal Anomaly and some required variables for algorithm 3.3 in Curtis

    Args:
        X0 (int): Initial Universal Anomaly Guess
        a (_type_): Semi-Major Axis
        r0 (_type_): Distance magnitude
        vr (_type_): Radial Velocity
        dt (_type_): Time Step
        mu (int, optional): Gravitational Parameter. Defaults to 398600.

    Returns:
        _type_: Returns the correct Universal Anomaly
    """
    
    tol = 1
    ratio = 1
    iterations = 0
    while abs(tol) >= 1e-8:
        z = a*X0**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        
        fX = ((r0*vr)/np.sqrt(mu))*(X0**2)*C + (1-a*r0)*(X0**3)*S + r0*X0 - np.sqrt(mu)*dt
        fdX = ((r0*vr)/np.sqrt(mu))*X0*(1-a*X0**2*S) + (1 - a*r0)*(X0**2)*C + r0
        ratio = fX / fdX
        X = X0 - ratio
        tol = X - X0
        X0 = X
        iterations += 1
    return X0, iterations

def hohmann_transfer(r1, r2, mu=398600):
    """
    Simple Hohmann transfer calculator assuming Earth as the central body.

    Parameters:
        r1 (float): initial orbit radius [m]
        r2 (float): final orbit radius [m]

    Returns:
        tuple: (delta_v1, delta_v2, total_delta_v, time_of_flight)
               all in SI units (m/s, s)
    """

    a = 0.5 * (r1 + r2)

    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    v_transfer1 = np.sqrt(mu * (2 / r1 - 1 / a))
    v_transfer2 = np.sqrt(mu * (2 / r2 - 1 / a))
    
    delta_v1 = v_transfer1 - v1
    delta_v2 = v2 - v_transfer2
    total_delta_v = abs(delta_v1) + abs(delta_v2)

    ToF = np.pi * np.sqrt(a**3 / mu)

    return delta_v1, delta_v2, total_delta_v, ToF

def lagrange(r0_vec, v0_vec, a, dt, X, mu=398600):
    """
    Description of function:
        Uses the Lagrange coefficients to find the position and velocity
        vectors at a given time step.

    Args:
        r0_vec (array): Initial position vector [km]
        v0_vec (array): Initial velocity vector [km/s]
        a (float): Semi-major axis
        dt (float): Time step [s]
        X (float): Universal Anomaly
        mu (int, optional): Gravitational parameter. Defaults to 398600.

    Returns:
        r_vec (array): Position vector at t0 + dt [km]
        v_vec (array): Velocity vector at t0 + dt [km/s]
    """
    z = a*X**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    
    r0_mag = np.linalg.norm(r0_vec)
    f = 1 - ((X**2)/r0_mag)*C
    g = dt - (1/np.sqrt(mu))*(X**3)*S
    r_vec = f*r0_vec + g*v0_vec
    r_mag = np.linalg.norm(r_vec)
    f_dot = (np.sqrt(mu)/(r_mag*r0_mag))*(a*(X**3)*S - X)
    g_dot = 1 - (X**2/r_mag)*C
    v_vec = f_dot*r0_vec + g_dot*v0_vec
    return r_vec, v_vec
    
def ECI_dt(r0, v0, dt, mu=398600):
    """
    Description of function:
        Takes an initial position and velocity vector, time step, and
        gravitational parameter to find the position and velocity vectors
        at a given time step using the Universal Anomaly method.

    Args:
        r0 (array): Initial position vector [km]
        v0 (array): Initial velocity vector [km/s]
        dt (float): Time step [s]
        mu (int, optional): Gravitational parameter. Defaults to 398600.

    Returns:
        r_vec (array): Position vector at t0 + dt [km]
        v_vec (array): Velocity vector at t0 + dt [km/s]
        iterations (int): Number of iterations used to find the Universal Anomaly
    """
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    v_r0 = np.dot(v0, r0)/r0_mag
    a = (2/r0_mag) - (v0_mag**2/mu)
    X0 = np.sqrt(mu)*abs(a)*dt
    X, iterations = UniversalAnom_findr(X0, a, r0_mag, v_r0, dt)
    r_vec, v_vec = lagrange(r0, v0, a, dt, X)
    return r_vec, v_vec, iterations

def UnAnom_propagator(r0, v0, tf, t_step, mu=398600):
    
    rprop = []
    total_time = 0
    while total_time < tf:
        r1, v1, _ = ECI_dt(r0, v0, t_step)
        rprop.append(r1)
        r0, v0 = r1, v1
        total_time += t_step
        
    return rprop

def lamberts(r1_vec, r2_vec, dt, prograde=True, max_iter=25, tol=1e-8, mu=398600):
    r1_vec = np.asarray(r1_vec)
    r2_vec = np.asarray(r2_vec)
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    z = 0
    C = .5
    S = 1/6
    z_l = -4*np.pi**2
    z_r = 4*np.pi**2
    dt_loop = 1
   
    # NEED TO CALC delTHETA and A
    dtheta = np.arccos(np.dot(r1_vec,r2_vec)/(r1*r2))
    if prograde is True:
        if np.cross(r1_vec, r2_vec)[2] < 0:
            dtheta = 2*np.pi - dtheta
    if prograde is False:
        if np.cross(r1_vec, r2_vec)[2] >= 0:
            dtheta = 2*np.pi - dtheta
            
    A = np.sin(dtheta)*np.sqrt((r1*r2)/(1-np.cos(dtheta)))
    
    def yz(z):
        S = stumpff_S(z)
        C = stumpff_C(z)
        y = r1 + r2 + A*((z*S - 1)/(np.sqrt(C)))
        return y, C, S
    i = 0
    while abs(dt_loop - dt) > tol and i < 100:
        y, C, S = yz(z)
        
        if y <= 0:
            raise ValueError(f"Lambert failed: y <= 0 for this dt (y={y}, z={z})")
        
        chi = np.sqrt(y/C)
        dt_loop = (chi**3 * S)/np.sqrt(mu) + (A*np.sqrt(y))/np.sqrt(mu)
        i += 1
        if dt_loop < dt:
            z_l = z
        else:
            z_r = z
        z = (z_r + z_l)/2
    
    # lagrange
    y, C, S = yz(z)
    f = 1-(y/r1)
    g = A*np.sqrt(y/mu)
    f_dot = np.sqrt(mu)/(r1*r2) * ((z*S-1)/np.sqrt(C))*np.sqrt(y)
    g_dot = 1-(y/r2)
    
    v1_vec = (1/g)*(r2_vec - f*r1_vec)
    v2_vec = (1/g)*(g_dot*r2_vec - r1_vec)
    
    eps2 = 1e-12
    if z > eps2: orb = "Elliptical"
    elif z < -eps2: orb = "Hyperbolic"
    else: orb = "Parabolic"

    return v1_vec, v2_vec, orb

def lamberts_porkchop(
    y01, y02, t1_max, t2_max, t_step=1800,
    prograde=True, mu=398600, title="Required delta v versus Time of Flight"
):
    def perigee_r(r_vec, v_vec, mu=398600):
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        ecc_vec = (np.cross(v_vec, h_vec)/mu) - (r_vec/np.linalg.norm(r_vec))
        ecc = np.linalg.norm(ecc_vec)
        return h**2/(mu*(1+ecc))
    
    # Time
    t1 = np.arange(0, t1_max, t_step, dtype=float)
    t2 = np.arange(0, t2_max, t_step, dtype=float)
    
    # Propagate the orbit throughout teval
    sol1 = ODEprimer(y01[0:3], y01[3:6], [0, float(t1_max)], t1)
    sol2 = ODEprimer(y02[0:3], y02[3:6], [0, float(t2_max)], t2)
    
    r1_prop, v1_prop = sol1.y[0:3], sol1.y[3:6]
    r2_prop, v2_prop = sol2.y[0:3], sol2.y[3:6]

    T1, T2 = np.meshgrid(t1, t2, indexing='xy')
    TOF = T2 - T1
    delv = np.full_like(T1, np.nan, dtype=float)
    
    
    # For loop, calculating each delta v within the TOF meshgrid
    for j in range(len(t2)):
        for i in range(len(t1)):
            dt = TOF[j,i]
            if dt <= 0:
                continue
            try:
                v1_f, v2_i, _ = lamberts(r1_prop[:,i], r2_prop[:,j], dt)
                delv1 = np.linalg.norm(v1_f-v1_prop[:,i])
                delv2 = np.linalg.norm(v2_prop[:,j]-v2_i)
                delv[j,i] = abs(delv1)+abs(delv2)
            except ValueError:
                pass
            
    # Plot Initializing 
    dv_max = 12.0
    n_levels = 12
    
    T1_h = T1/3600
    T2_h = T2/3600
    TOF_h = TOF / 3600
    
    Z = ma.masked_invalid(delv)
    Z = ma.masked_where(Z > dv_max, Z)
    levels = np.linspace(0, dv_max, n_levels)
    
    cmap = plt.get_cmap('magma')
    
    # Plotting
    plt.figure(figsize=(9,7))
    ax = plt.gca()
    cs = plt.contourf(T1_h, T2_h, Z, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cs, label='delv (km/s)')
    
    valid_mask = ~Z.mask
    if np.any(valid_mask):
        plt.contour(T1_h, T2_h, ma.masked_where(~valid_mask, TOF_h),
                    levels=[2,4,6,8,10,12], colors='black',
                    linewidth=0.8, linestyles='dashed')
    
    lim = float(min(T1_h.max(), T2_h.max()))
    plt.plot([0, lim], [0, lim], 'r--', lw=0.8, label='t2=t1')
    
    plt.xlabel('Departure Time (hours)')
    plt.ylabel('Arrival Time (hours)')
    plt.title('Porkchop Plot')
    plt.tight_layout()
    
    rp_min = 6478  # km (100 km altitude)
    
    valid_delv = np.isfinite(delv)
    if not np.any(valid_delv):
        print("No valid delta-v values found.")
        plt.show()
        return
    
    cutoff_delv = np.percentile(delv[valid_delv], 10)
    cand_idx = np.where(valid_delv & (delv <= cutoff_delv))
    rows, cols = cand_idx
    
    kept = []
    
    for j, i in zip(rows, cols):
        dt = TOF[j,i]
        if dt <= 0:
            continue
        try:
            # Recompute Lambert at this grid point to get transfer v at departure
            v1_f, v2_i, _ = lamberts(r1_prop[:, i], r2_prop[:, j], dt, prograde=prograde, mu=mu)
            rp = perigee_r(r1_prop[:, i], v1_f, mu=mu)  # perigee of the transfer orbit
            if rp > rp_min:
                dv_dep = float(np.linalg.norm(v1_f - v1_prop[:, i]))
                dv_arr = float(np.linalg.norm(v2_prop[:, j] - v2_i))
                kept.append({
                    "t1_sec": float(T1[j, i]),
                    "t2_sec": float(T2[j, i]),
                    "tof_sec": float(dt),
                    "t1_hr": float(T1_h[j, i]),
                    "t2_hr": float(T2_h[j, i]),
                    "tof_hr": float(TOF_h[j, i]),
                    "delta_v_kms": float(delv[j, i]),
                    "dv_depart" : dv_dep,
                    "dv_arrive" : dv_arr,
                    "rp_km": float(rp),
                    
                    # NEW: states for propagation
                    "r_depart_km": r1_prop[:, i].tolist(),
                    "r_arrive_km": r2_prop[:, j].tolist(),
                    "v_depart_transfer_kms": v1_f.tolist(),   # burn-to velocity
                    "v_arrive_transfer_kms":  v2_i.tolist()   # transfer velocity at arrival
                })
        except ValueError:
            continue
    
    kept.sort(key=lambda x: x["delta_v_kms"])
    
    if kept:
        kept_t1 = [d["t1_hr"] for d in kept]
        kept_t2 = [d["t2_hr"] for d in kept]
        plt.scatter(kept_t1, kept_t2, s=22, edgecolor='white', linewidth=0.5, label='Bottom 10% Δv (rp > 6478 km)')
        plt.legend(loc='best')

        for d in kept[:5]:
            ax.annotate(f'{d["delta_v_kms"]:.2f}',
                        xy=(d["t1_hr"], d["t2_hr"]),
                        xytext=(3, 3), textcoords='offset points', fontsize=18)

    plt.show()

    return kept[:3]

def lamberts_porkchop_preset(
    y01,
    y02,
    t1_max,
    t2_max,
    t_step=600,
    prograde=True,
    mu=398600,
    title="Lambert Transfer Trade Study",
    output_png="porkchop.png",
    output_json="porkchop_summary.json",
    rp_min=6478,
    tof_min_hr=1.5,
    tof_max_hr=10.0,
    n_levels=12
):
    """
    Generate a cleaner static Lambert transfer trade plot for portfolio use.

    Inputs
    ------
    y01, y02 : ndarray
        Initial state vectors [rx, ry, rz, vx, vy, vz]
    t1_max, t2_max : float
        Max propagation times in seconds
    t_step : float
        Grid spacing in seconds
    prograde : bool
        Lambert direction
    mu : float
        Gravitational parameter
    title : str
        Plot title
    output_png : str
        Filepath to save PNG image
    output_json : str
        Filepath to save JSON summary
    rp_min : float
        Minimum acceptable perigee radius [km]
    tof_min_hr, tof_max_hr : float
        Minimum and maximum displayed time of flight [hours]
    n_levels : int
        Number of contour levels
    """

    def perigee_r(r_vec, v_vec, mu=398600):
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        ecc_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / np.linalg.norm(r_vec))
        ecc = np.linalg.norm(ecc_vec)
        return h**2 / (mu * (1 + ecc))

    tof_min_sec = float(tof_min_hr) * 3600.0
    tof_max_sec = float(tof_max_hr) * 3600.0

    # Build time grids
    t1 = np.arange(0, t1_max, t_step, dtype=float)
    t2 = np.arange(0, t2_max, t_step, dtype=float)

    # Propagate both states
    sol1 = ODEprimer(y01[0:3], y01[3:6], [0, float(t1_max)], t1, mu=mu)
    sol2 = ODEprimer(y02[0:3], y02[3:6], [0, float(t2_max)], t2, mu=mu)

    r1_prop, v1_prop = sol1.y[0:3], sol1.y[3:6]
    r2_prop, v2_prop = sol2.y[0:3], sol2.y[3:6]

    T1, T2 = np.meshgrid(t1, t2, indexing="xy")
    TOF = T2 - T1
    delv = np.full_like(T1, np.nan, dtype=float)

    # Compute delta-v grid
    for j in range(len(t2)):
        for i in range(len(t1)):
            dt = TOF[j, i]

            if dt <= 0:
                continue
            if dt < tof_min_sec or dt > tof_max_sec:
                continue

            try:
                v1_f, v2_i, _ = lamberts(
                    r1_prop[:, i],
                    r2_prop[:, j],
                    dt,
                    prograde=prograde,
                    mu=mu
                )

                delv1 = np.linalg.norm(v1_f - v1_prop[:, i])
                delv2 = np.linalg.norm(v2_prop[:, j] - v2_i)
                total_dv = abs(delv1) + abs(delv2)

                if np.isfinite(total_dv):
                    delv[j, i] = total_dv

            except ValueError:
                continue

    # Convert to hours for plotting
    T1_h = T1 / 3600.0
    TOF_h = TOF / 3600.0

    # Keep only valid, displayable data
    valid_mask = np.isfinite(delv)
    valid_mask &= (TOF_h >= tof_min_hr) & (TOF_h <= tof_max_hr)

    if not np.any(valid_mask):
        summary = {
            "title": title,
            "output_png": output_png,
            "mu": mu,
            "prograde": prograde,
            "t1_max_sec": float(t1_max),
            "t2_max_sec": float(t2_max),
            "t_step_sec": float(t_step),
            "tof_min_hr": float(tof_min_hr),
            "tof_max_hr": float(tof_max_hr),
            "best_solution": None
        }

        fig = plt.figure(figsize=(9, 6))
        plt.title(title)
        plt.xlabel("Departure Time (hours)")
        plt.ylabel("Time of Flight (hours)")
        plt.text(
            0.5, 0.5,
            "No valid solutions found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes
        )
        plt.tight_layout()
        plt.savefig(output_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    # Build candidate list, but keep plot uncluttered
    kept = []
    candidate_cutoff = np.percentile(delv[valid_mask], 20)
    candidate_idx = np.where(valid_mask & (delv <= candidate_cutoff))
    rows, cols = candidate_idx

    for j, i in zip(rows, cols):
        dt = TOF[j, i]

        try:
            v1_f, v2_i, orb_type = lamberts(
                r1_prop[:, i],
                r2_prop[:, j],
                dt,
                prograde=prograde,
                mu=mu
            )

            rp = perigee_r(r1_prop[:, i], v1_f, mu=mu)
            if rp <= rp_min:
                continue

            dv_dep = float(np.linalg.norm(v1_f - v1_prop[:, i]))
            dv_arr = float(np.linalg.norm(v2_prop[:, j] - v2_i))

            kept.append({
                "departure_time_sec": float(T1[j, i]),
                "tof_sec": float(dt),
                "departure_time_hr": float(T1_h[j, i]),
                "tof_hr": float(TOF_h[j, i]),
                "delta_v_kms": float(delv[j, i]),
                "dv_depart_kms": dv_dep,
                "dv_arrive_kms": dv_arr,
                "rp_km": float(rp),
                "orbit_type": orb_type,
                "r_depart_km": r1_prop[:, i].tolist(),
                "r_arrive_km": r2_prop[:, j].tolist(),
                "v_depart_transfer_kms": v1_f.tolist(),
                "v_arrive_transfer_kms": v2_i.tolist()
            })
        except ValueError:
            continue

    if not kept:
        # Fall back to absolute minimum valid point even if rp filter removes everything
        best_j, best_i = np.unravel_index(np.nanargmin(np.where(valid_mask, delv, np.nan)), delv.shape)
        best_solution = {
            "departure_time_sec": float(T1[best_j, best_i]),
            "tof_sec": float(TOF[best_j, best_i]),
            "departure_time_hr": float(T1_h[best_j, best_i]),
            "tof_hr": float(TOF_h[best_j, best_i]),
            "delta_v_kms": float(delv[best_j, best_i]),
            "dv_depart_kms": None,
            "dv_arrive_kms": None,
            "rp_km": None,
            "orbit_type": "Unavailable",
            "r_depart_km": None,
            "r_arrive_km": None,
            "v_depart_transfer_kms": None,
            "v_arrive_transfer_kms": None
        }
    else:
        kept.sort(key=lambda x: x["delta_v_kms"])
        best_solution = kept[0]

    # Readable dynamic color scaling
    dv_lo = np.percentile(delv[valid_mask], 5)
    dv_hi = np.percentile(delv[valid_mask], 85)

    if not np.isfinite(dv_hi) or dv_hi <= dv_lo:
        dv_lo = float(np.nanmin(delv[valid_mask]))
        dv_hi = float(np.nanmax(delv[valid_mask]))

    Z = ma.masked_invalid(delv)
    Z = ma.masked_where(~valid_mask, Z)
    Z = ma.masked_where(Z > dv_hi, Z)

    levels = np.linspace(dv_lo, dv_hi, n_levels)

    # Plot in departure time vs TOF space
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()

    cs = plt.contourf(
        T1_h,
        TOF_h,
        Z,
        levels=levels,
        cmap="magma",
        extend="max"
    )
    plt.colorbar(cs, label="Total Δv (km/s)")

    # Optional TOF guide lines are unnecessary now because TOF is the y-axis

    # Mark best solution only
    plt.scatter(
        [best_solution["departure_time_hr"]],
        [best_solution["tof_hr"]],
        s=48,
        c="deepskyblue",
        edgecolor="white",
        linewidth=0.9,
        label="Best solution",
        zorder=5
    )

    ax.annotate(
        f'{best_solution["delta_v_kms"]:.2f} km/s',
        xy=(best_solution["departure_time_hr"], best_solution["tof_hr"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color="white"
    )

    plt.xlabel("Departure Time (hours)")
    plt.ylabel("Time of Flight (hours)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "title": title,
        "output_png": output_png,
        "mu": mu,
        "prograde": prograde,
        "t1_max_sec": float(t1_max),
        "t2_max_sec": float(t2_max),
        "t_step_sec": float(t_step),
        "tof_min_hr": float(tof_min_hr),
        "tof_max_hr": float(tof_max_hr),
        "best_solution": best_solution
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def phase_change_delta_v(rp, ra, delta_theta_deg, mu=398600.0, k=1, p=0):
    a = (ra+rp)/2
    e = (ra-rp)/(ra+rp)
    n = np.sqrt(mu/a**3)
    T0 = 2 * np.pi / n
    
    theta = np.deg2rad(delta_theta_deg) % (2*np.pi)
    
    s = np.sqrt(1-e)*np.sin(theta/2)
    c = np.sqrt(1+e)*np.cos(theta/2)
    Et = 2*np.arctan2(s, c) % (2*np.pi)
    
    Mt = (Et - e * np.sin(Et)) % (2*np.pi)
    
    t_tp = (2*np.pi - Mt)/n
    
    T_ph = (t_tp + p*T0) / k
    if T_ph <= 0:
        raise ValueError("Chosen k,p produce non-positive phasing period. Increase p or k.")
    
    a_ph = (mu*(T_ph/(2*np.pi))**2)**(1/3)
    ra_ph = 2*a_ph - rp
    
    v_p0 = np.sqrt(mu*(2/rp - 1/a))
    v_pph = np.sqrt(mu*(2/rp - 1/a_ph))
    
    delv1 = v_pph - v_p0
    delv2 = -delv1
    dv_tot = abs(delv1) + abs(delv2)
    
    return delv1, delv2, T_ph

def parse_tle(TLE, mu=398600, tol=1e-8, max_iter=100):
    """
    Parse a Two-Line Element (TLE) set into a dictionary of orbital elements.
    
    Parameters:
        TLE (list[str]): The TLE lines (2 or 3 lines). The first line may be a satellite name.
        mu (int): Gravitational Parameter
        tol (int): 
        
    Returns:
        dict: Dictionary containing satellite name (if present) and all TLE parameters.
    """
    # Ensure we have 2 or 3 lines
    if len(TLE) not in (2, 3):
        raise ValueError("TLE must have 2 or 3 lines (name + 2 orbital lines).")

    if len(TLE) == 3:
        name = TLE[0].strip()
        line1, line2 = TLE[1], TLE[2]
    else:
        name = None
        line1, line2 = TLE

    # Line 1 parsing
    satnum = line1[2:7].strip()
    classification = line1[7].strip()
    int_designator = line1[9:17].strip()
    epoch_year = int(line1[18:20])
    epoch_day = float(line1[20:32])
    first_derivative = float(line1[33:43])
    second_derivative = float(line1[44:52].replace(' ', '0').replace('-', 'E-').replace('+', 'E+'))
    bstar_str = line1[53:61].strip()
    if bstar_str:
        bstar_mantissa = float(bstar_str[0:5]) * 1e-5
        bstar_exponent = int(bstar_str[5:])
        bstar = bstar_mantissa * (10 ** bstar_exponent)
    else:
        bstar = 0.0
    ephemeris_type = int(line1[62])
    element_set_number = int(line1[64:68])
    
    # Line 2 parsing
    inclination = float(line2[8:16])
    raan = float(line2[17:25])
    eccentricity = float(f"0.{line2[26:33].strip()}")
    arg_perigee = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion = float(line2[52:63])
    rev_number = int(line2[63:68])
    
    # Angular Momentum Calculation
    n_rad_s = mean_motion * 2 * np.pi / 86400.0
    a = (mu / n_rad_s**2) ** (1 / 3)
    h = np.sqrt(mu * a * (1 - eccentricity**2))
    
    M = np.deg2rad(mean_anomaly) % (2*np.pi)
    
    inc_rad = np.deg2rad(inclination)
    raan_rad = np.deg2rad(raan)
    argp_rad = np.deg2rad(arg_perigee)

    # Initial guess for eccentric anomaly E
    E = M if eccentricity < 0.8 else np.pi
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - eccentricity*np.sin(E) - M
        f_prime = 1 - eccentricity*np.cos(E)
        delta_E = -f / f_prime
        E += delta_E
        if np.all(np.abs(delta_E) < tol):
            break

    # Compute true anomaly TA
    TA = 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(E / 2),
        np.sqrt(1 - eccentricity) * np.cos(E / 2)
    ) % (2*np.pi)
    
    COEs = np.array([inclination, raan, eccentricity, arg_perigee, h, TA, rev_number])
    
    r, v = COEs2ECI(h, eccentricity, inc_rad, raan_rad, argp_rad, TA)
    ECI = np.hstack((r, v))
    time = np.hstack((epoch_year, epoch_day))
    return COEs, ECI, time

def time_between_true_anom(ta1, ta2, ecc, h, mu=398600.0):
    """
    Description of function:
        Calculate the time between two true anomalies.

    Args:
        ta1 (float): Initial true anomaly [rad]
        ta2 (float): Final true anomaly [rad]
        ecc (float): Eccentricity
        h (float): Angular momentum [km^2/s]
        mu (int, optional): Gravitational parameter. Defaults to 398600.

    Returns:
        dt (float): Time between the two true anomalies [s]
    """
    a = h**2 / (mu * (1 - ecc**2))
    
    def E_from_TA(ta, ecc):
        E = 2 * np.arctan(np.tan(ta / 2) / np.sqrt((1 + ecc) / (1 - ecc)))
        if E < 0:
            E += 2 * np.pi
        return E

    E1 = E_from_TA(ta1, ecc)
    E2 = E_from_TA(ta2, ecc)

    M1 = E1 - ecc * np.sin(E1)
    M2 = E2 - ecc * np.sin(E2)

    n = np.sqrt(mu / a**3)

    dt = (M2 - M1) / n
    if dt < 0:
        dt += 2 * np.pi / n

    return dt

def synodic_period(T1, T2):
    """
    Compute the synodic period of two orbits.

    Parameters:
        T1 (float): orbital period of body 1
        T2 (float): orbital period of body 2
                    (units must match T1)

    Returns:
        float: synodic period (same units as T1)
    """
    return 1.0 / abs(1.0/T1 - 1.0/T2)

def sphere_of_influence(a, mu, MU=1.32712e11):
    """
    Calculate the sphere of influence for an orbiting body within a larger system
    
    Parameters:
        a   (float): semi major axis of the orbiting body
        mu  (float): Gravitational parameter of orbiting body
        MU  (float): Gravitational parameter of main systems body
        
    Returns:
        float: Sphere of Influence
    """
    soi = a*(mu/MU)**(.4)
    return soi

def planetary_elements(planet_id, T):
    """
    Planetary Ephemerides from Meeus (1991:202-204) and J2000.0

    Parameters
    ----------
    planet_id : int
        1 = Mercury
        2 = Venus
        3 = Earth
        4 = Mars
        5 = Jupiter
        6 = Saturn
        7 = Uranus
        8 = Neptune
    T : float
        Julian centuries from J2000.0

    Returns
    -------
    planet_coes : dict
        {
          'a'    : semimajor axis (km),
          'ecc'  : eccentricity,
          'inc'  : inclination (deg),
          'raan' : right ascension of ascending node (deg),
          'w_hat': longitude of perihelion (deg),
          'L'    : mean longitude (deg)
        }
    """

    if planet_id == 1:
        a = 0.387098310  # AU
        ecc  = 0.20563175 + 0.000020406*T - 0.0000000284*T**2 - 0.00000000017*T**3
        inc  = 7.004986   - 0.0059516*T    + 0.00000081*T**2   + 0.000000041*T**3
        raan = 48.330893  - 0.1254229*T    - 0.00008833*T**2   - 0.000000196*T**3
        w_hat = 77.456119 + 0.1588643*T    - 0.00001343*T**2   + 0.000000039*T**3
        L    = 252.250906 + 149472.6746358*T - 0.00000535*T**2 + 0.000000002*T**3

    elif planet_id == 2:
        a = 0.723329820
        ecc  = 0.00677188 - 0.000047766*T  + 0.000000097*T**2 + 0.00000000044*T**3
        inc  = 3.394662   - 0.0008568*T    - 0.00003244*T**2  + 0.000000010*T**3
        raan = 76.679920  - 0.2780080*T    - 0.00014256*T**2  - 0.000000198*T**3
        w_hat = 131.563707 + 0.0048646*T   - 0.00138232*T**2  - 0.000005332*T**3
        L    = 181.979801 + 58517.8156760*T + 0.00000165*T**2 - 0.000000002*T**3

    elif planet_id == 3:
        a = 1.000001018
        ecc  = 0.01670862 - 0.000042037*T  - 0.0000001236*T**2 + 0.00000000004*T**3
        inc  = 0.0000000  + 0.0130546*T    - 0.00000931*T**2   - 0.000000034*T**3
        raan = 0.0
        w_hat = 102.937348 + 0.3225557*T   + 0.00015026*T**2   + 0.000000478*T**3
        L    = 100.466449 + 35999.372851*T - 0.00000568*T**2

    elif planet_id == 4:
        a = 1.523679342
        ecc  = 0.09340062 + 0.000090483*T  - 0.00000000806*T**2 - 0.00000000035*T**3
        inc  = 1.849726   - 0.0081479*T    - 0.00002255*T**2   - 0.000000027*T**3
        raan = 49.558093  - 0.2949846*T    - 0.00063993*T**2   - 0.000002143*T**3
        w_hat = 336.060234 + 0.4438898*T   - 0.00017321*T**2   + 0.000000300*T**3
        L    = 355.433275 + 19140.2993313*T + 0.00000261*T**2  - 0.000000003*T**3

    elif planet_id == 5:
        a = 5.202603191 + 0.0000001913*T
        ecc  = 0.04849485 + 0.000163244*T  - 0.0000004719*T**2 + 0.00000000197*T**3
        inc  = 1.303270   - 0.0019872*T    + 0.00003318*T**2   + 0.000000092*T**3
        raan = 100.464441 + 0.1766828*T    + 0.00090387*T**2   - 0.000007032*T**3
        w_hat = 14.331309 + 0.2155525*T    + 0.00072252*T**2   - 0.000004590*T**3
        L    = 34.351484 + 3034.9056746*T  - 0.00008501*T**2   + 0.000000004*T**3

    elif planet_id == 6:
        a = 9.5549009596 - 0.0000021389*T
        ecc  = 0.05550862 - 0.000346818*T  - 0.0000006456*T**2 + 0.00000000338*T**3
        inc  = 2.488878   + 0.0025515*T    - 0.00004903*T**2   + 0.000000018*T**3
        raan = 113.665524 - 0.2566649*T    - 0.00018345*T**2   + 0.000000357*T**3
        w_hat = 93.056787 + 0.5665496*T    + 0.00052809*T**2   - 0.000004882*T**3
        L    = 50.077471 + 1222.1137943*T  + 0.00021004*T**2   - 0.000000019*T**3

    elif planet_id == 7:
        a = 19.218446062 - 0.0000000372*T + 0.00000000098*T**2
        ecc  = 0.04629590 - 0.000027337*T  + 0.0000000790*T**2 + 0.00000000025*T**3
        inc  = 0.773196   - 0.0016869*T    + 0.00000349*T**2   + 0.00000000016*T**3
        raan = 74.005947  + 0.0741461*T    + 0.00040540*T**2   + 0.000000104*T**3
        w_hat = 173.005159 + 0.0893206*T   - 0.00009470*T**2   + 0.000000413*T**3
        L    = 314.055005 + 428.4669983*T  - 0.00000486*T**2   - 0.000000006*T**3

    elif planet_id == 8:
        a = 30.110386869 - 0.0000001663*T + 0.00000000069*T**2
        ecc  = 0.00898809 + 0.000006408*T  - 0.0000000008*T**2
        inc  = 1.769952   + 0.0002557*T    + 0.00000023*T**2
        raan = 131.784057 - 0.0061651*T    - 0.00000219*T**2   - 0.000000078*T**3
        w_hat = 48.123691 + 0.0291587*T    + 0.00007051*T**2
        L    = 304.348665 + 218.4862002*T  + 0.00000059*T**2   - 0.000000002*T**3

    else:
        raise ValueError("planet_id must be an integer from 1 to 8")

    # convert semimajor axis from AU to km
    AU_KM = 149597870.0
    a_km = a * AU_KM

    return {
        "a": a_km,
        "ecc": ecc,
        "inc": inc,
        "raan": raan,
        "w_hat": w_hat,
        "L": L,
    }

def mean_to_true_anomaly(M, e, tol=1e-10, max_iter=50):
    """
    Convert mean anomaly to true anomaly for an elliptical orbit.

    Parameters
    ----------
    M : float or array_like
        Mean anomaly [rad]. Can be any real value; it will be wrapped to [0, 2π).
    e : float
        Eccentricity (0 <= e < 1).
    tol : float, optional
        Convergence tolerance for solving Kepler's equation.
    max_iter : int, optional
        Maximum number of Newton iterations.

    Returns
    -------
    nu : float or ndarray
        True anomaly [rad], same shape as M.
    """

    M = np.mod(M, 2*np.pi)          # wrap to [0, 2π)
    M = np.asarray(M, dtype=float)

    if not (0 <= e < 1):
        raise ValueError("Function only valid for elliptical orbits: 0 <= e < 1")

    # --- Solve Kepler's equation: M = E - e*sin(E) for eccentric anomaly E ---
    # Good initial guess for E
    E = M.copy()
    if e > 0.8:                      # better initial guess for high e
        E = np.pi * np.ones_like(M)

    for _ in range(max_iter):
        f  = E - e*np.sin(E) - M     # Kepler's equation
        fp = 1 - e*np.cos(E)         # derivative
        dE = -f / fp
        E += dE
        if np.all(np.abs(dE) < tol):
            break

    # --- Convert eccentric anomaly E to true anomaly ν ---
    cos_nu = (np.cos(E) - e) / (1 - e*np.cos(E))
    sin_nu = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e*np.cos(E))
    nu = np.arctan2(sin_nu, cos_nu)

    # Wrap ν to [0, 2π)
    nu = np.mod(nu, 2*np.pi)
    return nu

def date_to_planet_rv(planet_id, year, month, day, hour, minute, second=0):
    """
    -----------------------------
    The purpose of this function is to take inputs of date and time and return
    the r and v vectors of a planet within our solar system.
    -----------------------------
    planet_id : int
        1 = Mercury
        2 = Venus
        3 = Earth
        4 = Mars
        5 = Jupiter
        6 = Saturn
        7 = Uranus
        8 = Neptune
    -----------------------------
    returns
        rvec - np.array
        vvec - np.array
    -----------------------------
    """
    # Time
    t = julian_date(year, month, day, hour, minute, second, False)
    jcent = (t-2451545)/36525
    
    # Grabbing planetary elements
    plan_el = planetary_elements(planet_id, jcent)
    
    w_hat = np.deg2rad(plan_el['w_hat'])
    raan = np.deg2rad(plan_el['raan'])
    L = np.deg2rad(plan_el['L'])
    ecc = plan_el['ecc']
    a = plan_el['a']
    inc = np.deg2rad(plan_el['inc'])
    
    # Convert to COEs
    argp = w_hat - raan
    meana = L - w_hat
    
    TA = mean_to_true_anomaly(meana, ecc)
    p = a*(1-ecc**2)
    
    r = p/(1+ecc*np.cos(TA))
    v = np.sqrt(1.32712e11*(2/r - 1/a))
    h = r*v
    
    # COEs to ECI
    rvec, vvec = COEs2ECI(h, ecc, inc, raan, argp, TA, mu=1.32712e11)
    
    return rvec, vvec