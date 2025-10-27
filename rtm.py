#%%
#
# Daniel Rodrigues Pipa
# Universidade Tecnológica Federal do Paraná
# 2025-08-04
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
import taichi as ti
from scipy.integrate import cumulative_trapezoid as cumtrapz
from findiff import coefficients as fdcoeffs

# Load Marmousi velocity model
# cpnp = np.load("cp_marmousi.npy")
c0 = 1500
# fname = "cp_rtm.npy"
fname = "cp_rtm_filled.npy"
c2_np = np.load(fname).T[:, ::1] ** 2
c2c_np = c0 ** 2 * np.ones_like(c2_np)
Nx, Ny = c2_np.shape
Nxyz = c2_np.shape
Nd = len(Nxyz)
dx = 4.
dt = 1e-3

# plt.figure(1)
# plt.imshow(c2_np.T, origin="lower")
# plt.colorbar()
# plt.show(block=True)
#%

cpmax = np.max(c2_np)
dtOdx = dt / dx
dt2Odx2 = dtOdx ** 2
# CN = cpmax * dtOdx

# if CN > 1.:
#     raise ValueError(f"Courant number {CN} > 1.")
# else:
#     print(f"Courant number: {CN}")


c2_np[0, :] = 0
c2_np[:, 0] = 0
c2_np[-1, :] = 0
c2_np[:, -1] = 0
c2c_np[0, :] = 0
c2c_np[:, 0] = 0
c2c_np[-1, :] = 0
c2c_np[:, -1] = 0

Lt = 2.1
Nt = round(Lt / dt)
tt = np.arange(Nt) * dt
bw = .95
fc = 40
t0 = 1.5 / (bw * fc)
t1 = 4 * t0
nt1 = round(t1 / dt)
s_np = ss.gausspulse(tt - t0, fc=fc, bw=bw).astype(np.float32)[:, np.newaxis]
# sd_np = np.diff(s_np[:, 0], prepend=0).astype(np.float32)[:, np.newaxis]
#sdnp =
# plt.figure(2)
# plt.plot(tt, s_np)
# plt.grid()
# plt.show(block=True)

xyz_s_np = np.array([[round(Nx / 2), Ny - 2]], dtype=np.int32)
xm = np.arange(round(Nx / 6), round(5 * Nx / 6))
xyz_m_np = np.block([[xm], [(Ny - 2) * np.ones(xm.shape)]]).T.astype(np.int32)

Nm = len(xyz_m_np)
Ns = len(xyz_s_np)

#%%

ti.init(arch=ti.gpu)

# Pressure and velocity fields
pv = ti.field(float, shape=(*Nxyz, 4))
pf = ti.field(float, shape=Nxyz)
pb = ti.field(float, shape=Nxyz)
vf = ti.Vector.field(n=Nd, dtype=float, shape=Nxyz)
vb = ti.Vector.field(n=Nd, dtype=float, shape=Nxyz)

rh = ti.field(float, shape=(Nx, Ny))
rhpoy = ti.field(float, shape=(Nx, Ny))

c2 = ti.field(float, shape=(Nx, Ny))
c2c = ti.field(float, shape=(Nx, Ny))
s = ti.field(float, shape=(Nt, len(xyz_s_np)))
# sd = ti.field(float, shape=(Nt, len(xyz_s_np)))
m = ti.field(float, shape=(Nt, len(xyz_m_np)))

xyz_s = ti.field(float, shape=xyz_s_np.shape)
xyz_m = ti.field(float, shape=xyz_m_np.shape)

xyz_s.from_numpy(xyz_s_np)
xyz_m.from_numpy(xyz_m_np)

pv.fill(1.)
pf.fill(0.)
pb.fill(0.)

c2.from_numpy(c2_np)
c2c.from_numpy(c2c_np)
s.from_numpy(s_np)
# sd.from_numpy(sd_np)
m.fill(0.)

# @ti.func
# def Dx(u, x, y, bf, nd):
#     return (u[x+1+bf, y, nd] - u[x+bf, y, nd])/2
#
# @ti.func
# def Dy(u, x, y, bf, nd):
#     return (u[x, y+1+bf, nd] - u[x, y+bf, nd])/2
#
# @ti.func
# def Dxb(x, y, bf, nd):
#     return (pb[x+1+bf, y, nd] - pb[x+bf, y, nd])/2
#
# @ti.func
# def Dyb(x, y, bf, nd):
#     return (pb[x, y+1+bf, nd] - pb[x, y+bf, nd])/2

deriv_acc = 8
offsets = tuple(np.arange(-deriv_acc, deriv_acc) + .5)
cd1 = tuple(fdcoeffs(deriv=1, offsets=offsets)["coefficients"][deriv_acc:].astype(np.float32))
offsets = tuple(np.arange(1 - deriv_acc, deriv_acc))
cd2 = tuple(fdcoeffs(deriv=2, offsets=offsets)["coefficients"][round(deriv_acc/2):].astype(np.float32))

# @ti.func
# def lap(u, xyz):
#     l = ti.static(Nd * cd2[0]) * u[xyz]
#     for nd in ti.static(range(Nd)):
#         for nc in ti.static(range(1, len(cd2))):
#             xyz_tmp = xyz[:]
#             xyz_tmp[nd] += nc
#             a = u[xyz_tmp]
#             xyz_tmp[nd] += - 2 * nc
#             l += ti.static(cd2[nc]) * (a + u[xyz_tmp])
#     return l


@ti.func
def Dv(u: ti.template(), xyz, nd: int, bf: int, imax: int):
    d = 0.
    # imax = u.shape[nd[0]]
    for nc in ti.static(range(deriv_acc)):
        xyz[nd] += nc + bf
        a = u[xyz][nd] if xyz[nd] < imax else 0
        xyz[nd] += - 2 * nc - 1
        b = u[xyz][nd] if xyz[nd] >= 0 else 0
        xyz[nd] += nc + 1 - bf
        d += ti.static(cd1[nc]) * (a - b)
    return d

@ti.func
def Dp(u: ti.template(), xyz, nd: int, bf: int, imax: int):
    d = 0.
    # imax = u.shape[nd[0]]
    for nc in ti.static(range(deriv_acc)):
        xyz[nd] += nc + bf
        a = u[xyz] if xyz[nd] < imax else 0
        xyz[nd] += - 2 * nc - 1
        b = u[xyz] if xyz[nd] >= 0 else 0
        xyz[nd] += nc + 1 - bf
        d += ti.static(cd1[nc]) * (a - b)
    return d


@ti.func
def add_source(p, xyz, s_, xyz_, nt_):
    for n_ in ti.ndrange(xyz_.shape[0]):
        if xyz[0] == xyz_[n_, 0] and xyz[1] == xyz_[n_, 1]:
            p[xyz] += s_[nt_, n_]

@ti.func
def read_sensors(p, xyz, m_, xyz_, nt_):
    for n_ in ti.ndrange(xyz_.shape[0]):
        if xyz[0] == xyz_[n_, 0] and xyz[1] == xyz_[n_, 1]:
            m_[nt_, n_] = p[xyz]

@ti.kernel
def update_p_(p: ti.template(), v: ti.template(), c2_: ti.template(),
              s_: ti.template(), xyz_s_: ti.template(), Ns_: int,
              m_: ti.template(), xyz_m_: ti.template(), Nm_: int, nt_: int):
    for xyz in ti.grouped(p):
        d = 0.
        for nd in ti.static(range(Nd)):
            d += Dv(v, xyz, nd, 1, Nxyz[nd])
        p[xyz] -=  dtOdx * c2_[xyz] * d
        if Ns_:
            add_source(p, xyz, s_, xyz_s_, nt_)
        if Nm_ and nt_ > nt1:
            read_sensors(p, xyz, m_, xyz_m_, nt_)

@ti.kernel
def update_v_(p: ti.template(), v: ti.template()):
    for xyz in ti.grouped(p):
        for nd in ti.static(range(Nd)):
            d = Dp(p, xyz, nd, 0, p.shape[nd])
            v[xyz][nd] -= dtOdx * d

@ti.kernel
def update_rh(pd: ti.template(), vd: ti.template(), pu: ti.template(), vu: ti.template()):
    for xyz in ti.grouped(pf):
        sdz = -pd[xyz] * vd[xyz][1]
        suz = -pu[xyz] * vu[xyz][1]
        pdpu = pd[xyz] * pu[xyz]
        rh[xyz] += pdpu
        ic = (sdz > 0) & (suz < 0)
        # ic = (sdz < 0) & (suz > 0)
        rhpoy[xyz] += pdpu * ic

@ti.kernel
def invsign(u: ti.template()):
    for xyz in ti.grouped(u):
        u[xyz] *= -1

red = ti.static(0)
green = ti.static(1)
blue = ti.static(2)

@ti.kernel
def prepare_visualization(p1: ti.template(), p2: ti.template(), gain: float):
    for xyz in ti.grouped(p1):
        # value = 1e7 * -v[1][x, y] * p[x, y]
        #sx = -(v[0][x, y] + v[0][x-1, y]) * p[x, y] / 2
        #sy = -(v[1][x, y] + v[1][x, y-1]) * p[x, y] / 2
        # sx = -v[x, y, 0] * p[x, y, nd]
        # sy = -v[1][x, y] * p[x, y]
        value = gain * (p1[xyz] + p2[xyz]) # if sy > 0 else 0
        pv[xyz, green] = 1 - ti.abs(value)
        if value > 0:
            pv[xyz, red] = 1
            pv[xyz, blue] = 1 - value
        else:
            pv[xyz, red] = 1 + value
            pv[xyz, blue] = 1
        
        c = 1e-1 * c2[xyz] / cpmax
        for ic in ti.static(range(3)):
            pv[xyz, ic] -= c

        # if xy - 1 < y < Nym + 1:
        #     pv[x, y, green] = 0

view = 20
if view:
    window = ti.ui.Window(name="Pressure", res=(Nx, Ny), pos=(0, 0))
    canvas = window.get_canvas()

# view = 0
for nt in tqdm(range(Nt)):
    update_p_(pf, vf, c2, s, xyz_s, Ns, m, xyz_m, Nm, nt)
    update_v_(pf, vf)
    if view:
        if not nt % view:
            prepare_visualization(pf, pf, 30)
            canvas.set_image(pv)
            window.show()

for nt in tqdm(range(Nt)):
    update_p_(pb, vb, c2c, m, xyz_m, Nm, s, xyz_s, 0, Nt - nt)
    update_v_(pb, vb)
    if view:
        if not nt % view:
            prepare_visualization(pb, pb, 30)
            canvas.set_image(pv)
            window.show()

invsign(vb)
pf.fill(0.)
vf.fill(0.)

for nt in tqdm(range(Nt - nt1)):
    update_p_(pf, vf, c2c, s, xyz_s, Ns, m, xyz_m, 0, nt)
    update_v_(pf, vf)
    update_p_(pb, vb, c2c, m, xyz_m, 0, s, xyz_s, 0, 0)
    update_v_(pb, vb)
    update_rh(pf, vf, pb, vb)
    if view:
        if not nt % view:
            prepare_visualization(pf, pb, 30)
            canvas.set_image(pv)
            window.show()

if view:
    window.destroy()
#%%
plt.figure()
vmm = .001
plt.imshow(m.to_numpy(), aspect="auto", cmap="gray", vmin=-vmm, vmax=+vmm, origin="lower")
plt.colorbar()
plt.title("Measurements")
plt.grid()

plt.figure()
vmm = .001
plt.imshow(rh.to_numpy().T + 1e-10 * c2_np.T, aspect="auto", cmap="gray", vmin=-vmm, vmax=+vmm, origin="lower")
plt.colorbar()
plt.title("RTM")
plt.grid()

plt.figure()
vmm = .001
plt.imshow(rhpoy.to_numpy().T + 1e-10 * c2_np.T, aspect="auto", cmap="gray", vmin=-vmm, vmax=+vmm, origin="lower")
plt.colorbar()
plt.title("RTM Poynting")
plt.grid()

plt.show(block=True)
# plt.show(block=False)