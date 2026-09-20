from __future__ import annotations
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# Aegis study_config qubit defaults
DEFAULTS = dict(
    omega_q=5.0, Omega=20.0, Gamma_m=1.0, kappa=5.0, eta=0.6,
    eps0=0.70, eps_T=1e-4, T=4.0, theta_b=0.20, lam=0.5,
    c=20.0, wgamma=1.0, wdelta=1e4, umax=1.0, gmax=3.90,
    dt=1e-3, seed=0,
)
# rho0 = [[0.78, 0.08],[0.08, 0.22]]  ->  r0 = (2 Re rho01, -2 Im rho01, rho00 - rho11)
R0 = (0.16, 0.0, 0.56)


def solve_qp(alpha, bH, bD, V, lam, wu, wg, wd, umax, gmax):
    """Five-case closed form. Returns (u, gamma, case)."""
    Theta = alpha + lam * V
    if Theta <= 0.0:
        return 0.0, 0.0, "V"
    nu = Theta / (bH * bH / wu + bD * bD / wg + 1.0 / wd)
    u, g = -nu * bH / wu, -nu * bD / wg
    u_ok, g_ok = abs(u) <= umax, g <= gmax
    if u_ok and g_ok:
        return u, g, "I"
    usat = math.copysign(umax, u) if u != 0.0 else 0.0
    if (not u_ok) and g_ok:
        nu2 = (alpha + bH * usat + lam * V) / (bD * bD / wg + 1.0 / wd)
        g2 = -nu2 * bD / wg
        if g2 <= gmax:
            return usat, g2, "II"
        return usat, gmax, "IV"
    if u_ok and (not g_ok):
        nu3 = (alpha + bD * gmax + lam * V) / (bH * bH / wu + 1.0 / wd)
        u3 = -nu3 * bH / wu
        if abs(u3) <= umax:
            return u3, gmax, "III"
        return math.copysign(umax, u3), gmax, "IV"
    return usat, gmax, "IV"


def simulate(p, controlled: bool, dW=None):
    """Return dict: t, x (n,3), xi, eps, u, gamma, case, dW, confined."""
    om, Om, Gm, kap, eta = p["omega_q"], p["Omega"], p["Gamma_m"], p["kappa"], p["eta"]
    dt = p["dt"]; T = p["T"]
    n = int(round(T / dt))
    if dW is None:
        dW = np.random.default_rng(int(p["seed"])).normal(0.0, math.sqrt(dt), n)
    dWl = dW.tolist()

    eps0, epsT = p["eps0"], p["eps_T"]
    r_f = -math.log(0.05) / T
    theta_b, lam = p["theta_b"], p["lam"]
    c, wg, wd, umax, gmax = p["c"], p["wgamma"], p["wdelta"], p["umax"], p["gmax"]
    eps_f = 1e-2
    sq = math.sqrt(eta * Gm)          # sigma_xi = -sq (1 - z^2)
    sq2 = 2.0 * sq                    # Bloch diffusion prefactor
    etaGm4 = 4.0 * eta * Gm           # Milstein prefactor

    x, y, z = R0
    X = np.empty((n, 3)); XI = np.empty(n); EPS = np.empty(n)
    U = np.zeros(n); GAM = np.zeros(n); CASE = [None] * n
    confined = True
    for k in range(n):
        t = k * dt
        xi = 0.5 * (1.0 - z)
        e_exp = (eps0 - epsT) * math.exp(-r_f * t)
        eps_t = epsT + e_exp
        if xi >= eps_t:
            confined = False
        u = g = 0.0
        case = "-"
        if controlled:
            s = max(eps_t - xi, theta_b * eps_t)
            V = -math.log(s / eps_t)
            sig = -sq * (1.0 - z * z)
            bH = -Om * y
            bD = -kap * xi
            eps_dot = -r_f * e_exp
            alpha = -(xi / eps_t) * eps_dot + 0.5 * (1.0 / s) * sig * sig
            wu = c / (abs(bH) + eps_f)
            u, g, case = solve_qp(alpha, bH, bD, V, lam, wu, wg, wd, umax, gmax)
        X[k, 0] = x; X[k, 1] = y; X[k, 2] = z
        XI[k] = xi; EPS[k] = eps_t; U[k] = u; GAM[k] = g; CASE[k] = case

        # drift
        hx, hz = 2.0 * u * Om, om
        kg = kap * g
        Fx = -hz * y - 2.0 * Gm * x - 0.5 * kg * x
        Fy = hz * x - hx * z - 2.0 * Gm * y - 0.5 * kg * y
        Fz = hx * y + kg * (1.0 - z)
        # diffusion and Milstein correction
        Gx, Gy, Gz = -sq2 * z * x, -sq2 * z * y, sq2 * (1.0 - z * z)
        q = 2.0 * z * z - 1.0
        Mx, My, Mz = etaGm4 * x * q, etaGm4 * y * q, -2.0 * etaGm4 * z * (1.0 - z * z)
        dw = dWl[k]; m = 0.5 * (dw * dw - dt)
        x += Fx * dt + Gx * dw + Mx * m
        y += Fy * dt + Gy * dw + My * m
        z += Fz * dt + Gz * dw + Mz * m
        rr = math.sqrt(x * x + y * y + z * z)
        if rr > 1.0:
            x /= rr; y /= rr; z /= rr
    return dict(t=np.arange(n) * dt, x=X, xi=XI, eps=EPS, u=U, gamma=GAM,
                case=CASE, dW=dW, confined=confined)


#gui
class Demo:
    INTERVAL = 33          # ms per animation frame (~30 fps target)
    PLAY_SECONDS = 8.0     # wall-clock length of one playback
    TAIL = 250             # bright highlighted tail, in sim steps
    DEBOUNCE = 150         # ms after the last slider motion before re-sim

    def __init__(self):
        self.p = dict(DEFAULTS)
        self.mode = "Both"
        self.runs = {}
        self.frame = 0
        self.playing = False
        self.n = 1

        self.fig = plt.figure("SCCQS - qubit stabilisation demo", figsize=(15.5, 8.6))
        self.fig.patch.set_facecolor("#0e0f13")
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.6, 1.0, 0.62],
                                   left=0.0, right=0.99, top=0.95, bottom=0.07,
                                   wspace=0.28, hspace=0.32)
        self.ax3 = self.fig.add_subplot(gs[:, 0], projection="3d", facecolor="#0e0f13")
        self.ax_xi = self.fig.add_subplot(gs[0, 1])
        self.ax_u = self.fig.add_subplot(gs[1, 1], sharex=self.ax_xi)

        self._build_sphere()
        self._build_widgets()
        self.timer = self.fig.canvas.new_timer(interval=self.INTERVAL)
        self.timer.add_callback(self._tick)
        self.debounce = self.fig.canvas.new_timer(interval=self.DEBOUNCE)
        self.debounce.single_shot = True
        self.debounce.add_callback(self.rerun)
        self.rerun()

    def _build_sphere(self):
        ax = self.ax3
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        ax.computed_zorder = False           # skip per-draw depth sorting (cheap + stable)
        th = np.linspace(0, 2 * np.pi, 90)
        # light wireframe: 3 great circles + a few latitudes
        for lat in (-0.6, -0.3, 0.3, 0.6):
            rr = math.sqrt(1 - lat * lat)
            ax.plot(rr * np.cos(th), rr * np.sin(th), np.full_like(th, lat),
                    color="#2f3646", lw=0.5)
        for lon in np.linspace(0, np.pi, 6, endpoint=False):
            ax.plot(np.cos(th) * np.cos(lon), np.cos(th) * np.sin(lon), np.sin(th),
                    color="#2f3646", lw=0.5)
        ax.plot(np.cos(th), np.sin(th), 0 * th, color="#59627a", lw=0.9)
        for a, lab, col in [((0, 0, 1), "|0>", "#c8ccd6"), ((0, 0, -1), "|1>", "#c8ccd6"),
                            ((1, 0, 0), "x", "#8b93a7"), ((0, 1, 0), "y", "#8b93a7")]:
            a = np.array(a, float)
            ax.plot([-a[0], a[0]], [-a[1], a[1]], [-a[2], a[2]], color=col, lw=0.7, alpha=0.5)
            ax.text(*(a * 1.15), lab, color=col, fontsize=11, ha="center")
        ax.scatter([0], [0], [1], marker="*", s=220, color="#ffd23f",
                   edgecolor="k", linewidth=0.5)

        # dynamic artists
        self.cap_eps, = ax.plot([], [], [], color="#ff5c5c", lw=1.8)
        self.cap_shell, = ax.plot([], [], [], color="#ff9f43", lw=1.0, ls="--", alpha=0.8)
        self.art = {}
        for key, col in (("ctrl", "#5ec8ff"), ("free", "#ff8c42")):
            self.art[key] = dict(
                ghost=ax.plot([], [], [], color=col, lw=0.6, alpha=0.22)[0],
                trail=ax.plot([], [], [], color=col, lw=1.3, alpha=0.55)[0],
                tail=ax.plot([], [], [], color="white" if key == "ctrl" else "#ffd9b3",
                             lw=2.6)[0],
                dot=ax.plot([], [], [], "o", color=col, ms=9, mec="white", mew=0.8)[0],
                arrow=ax.plot([], [], [], color=col, lw=1.2, alpha=0.7)[0],
            )
        self.txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, color="#e8eaf0",
                             fontsize=10, family="monospace", va="top")
        Z = 0.72
        ax.set_xlim(-Z, Z); ax.set_ylim(-Z, Z); ax.set_zlim(-Z, Z)
        ax.view_init(elev=22, azim=-55)

    def _build_widgets(self):
        fig = self.fig
        x0, w = 0.80, 0.16
        rax = fig.add_axes([x0, 0.79, w, 0.12], facecolor="#15171e")
        self.radio = RadioButtons(rax, ("Uncontrolled", "Controlled", "Both (same noise)"),
                                  active=2, activecolor="#5ec8ff")
        for lab in self.radio.labels:
            lab.set_color("#e8eaf0"); lab.set_fontsize(9)
        self.radio.on_clicked(self._on_mode)

        specs = [  # key, label, lo, hi, log?
            ("lam",     "λ  (barrier decay)",   0.0,  3.0,  False),
            ("theta_b", "θ_b (buffer frac.)",   0.02, 0.9,  False),
            ("eps0",    "ε₀ (funnel start)",    0.30, 0.95, False),
            ("T",       "T  (horizon)",         1.0,  8.0,  False),
            ("gmax",    "γ_max (dissip. bound)", 0.0, 8.0,  False),
            ("umax",    "u_max (coh. bound)",   0.0,  3.0,  False),
            ("c",       "c  (u-weight scale)",  0.1,  60.0, False),
            ("wdelta",  "log10 w_δ (slack)",    1.0,  6.0,  True),
            ("kappa",   "κ  (decay rate)",      0.0,  15.0, False),
            ("Omega",   "Ω  (drive amp.)",      0.0,  40.0, False),
            ("Gamma_m", "Γ_m (meas. rate)",     0.05, 4.0,  False),
            ("eta",     "η  (efficiency)",      0.05, 1.0,  False),
            ("omega_q", "ω_q (drift freq.)",    0.0,  10.0, False),
            ("seed",    "noise seed",           0,    99,   False),
        ]
        self.sliders = {}
        y = 0.76
        for key, label, lo, hi, islog in specs:
            ax = fig.add_axes([x0 + 0.02, y, w - 0.03, 0.022], facecolor="#2a2e38")
            v0 = math.log10(self.p[key]) if islog else self.p[key]
            s = Slider(ax, label, lo, hi, valinit=v0, color="#5ec8ff",
                       valstep=1 if key == "seed" else None)
            s.label.set_color("#cfd3dc"); s.label.set_fontsize(8)
            s.valtext.set_color("#e8eaf0"); s.valtext.set_fontsize(8)
            s.label.set_x(-0.02)
            s.on_changed(lambda v, k=key, lg=islog: self._on_slider(k, v, lg))
            self.sliders[key] = s
            y -= 0.038

        def button(rect, text, cb):
            b = Button(fig.add_axes(rect), text, color="#2a2e38", hovercolor="#3a4256")
            b.label.set_color("#e8eaf0"); b.on_clicked(cb)
            return b
        self.btn_play = button([x0 + 0.02, 0.09, 0.06, 0.04], "Play", self._on_play)
        self.btn_rerun = button([x0 + 0.09, 0.09, 0.06, 0.04], "Rerun", lambda e: self.rerun())
        self.btn_seed = button([x0 + 0.02, 0.03, 0.13, 0.04], "New noise seed",
                               self._on_new_seed)

        fig.text(0.80, 0.955, "SCCQS: measured qubit → |0⟩", color="#e8eaf0", fontsize=11,
                 weight="bold")
        fig.text(0.80, 0.925, "blue = controlled   orange = uncontrolled", color="#9aa1b3",
                 fontsize=8)
        self._set_controls_enabled()

    # ---- callbacks ---------------------------------------------------------
    def _on_slider(self, key, v, islog):
        self.p[key] = 10.0 ** v if islog else float(v)
        self.debounce.stop(); self.debounce.start()      # re-sim after drag settles

    def _on_mode(self, label):
        self.mode = label.split(" ")[0]
        self._set_controls_enabled()
        self.rerun()

    def _set_controls_enabled(self):
        ctrl_keys = {"lam", "theta_b", "eps0", "gmax", "umax", "c", "wdelta"}
        a = 1.0 if self.mode != "Uncontrolled" else 0.3
        for k, s in self.sliders.items():
            if k in ctrl_keys:
                s.poly.set_alpha(a); s.label.set_alpha(a); s.valtext.set_alpha(a)

    def _on_new_seed(self, _):
        self.sliders["seed"].set_val((int(self.p["seed"]) + 1) % 100)

    def _on_play(self, _):
        if self.playing:
            self._stop()
        else:
            if self.frame >= self.n - 1:
                self.frame = 0
            self.timer.start(); self.playing = True; self.btn_play.label.set_text("Pause")
        self.fig.canvas.draw_idle()

    def _stop(self):
        self.timer.stop(); self.playing = False; self.btn_play.label.set_text("Play")

    # ---- run + draw --------------------------------------------------------
    def rerun(self):
        self._stop()
        p = self.p
        self.runs = {}
        dW = np.random.default_rng(int(p["seed"])).normal(
            0.0, math.sqrt(p["dt"]), int(round(p["T"] / p["dt"])))
        if self.mode in ("Controlled", "Both"):
            self.runs["ctrl"] = simulate(p, True, dW)
        if self.mode in ("Uncontrolled", "Both"):
            self.runs["free"] = simulate(p, False, dW)
        ref = next(iter(self.runs.values()))
        self.n = len(ref["t"]); self.t = ref["t"]; self.eps = ref["eps"]
        self.stride = max(1, int(self.n / (self.PLAY_SECONDS * 1000.0 / self.INTERVAL)))
        self._draw_static_panels()
        self.frame = 0
        self._draw_frame()
        self.fig.canvas.draw_idle()

    def _draw_static_panels(self):
        axx, axu = self.ax_xi, self.ax_u
        axx.cla(); axu.cla()
        for ax in (axx, axu):
            ax.set_facecolor("#15171e"); ax.grid(True, color="#2a2e38", lw=0.6)
            ax.tick_params(colors="#cfd3dc")
            for sp in ax.spines.values():
                sp.set_color("#3a3f4c")
        leg = dict(fontsize=8, loc="upper right", facecolor="#15171e",
                   edgecolor="#3a3f4c", labelcolor="#e8eaf0")
        axx.fill_between(self.t, 0, self.eps, color="#ff5c5c", alpha=0.08)
        axx.plot(self.t, self.eps, color="#ff5c5c", lw=1.6, label="funnel ε(t)")
        axx.plot(self.t, (1 - self.p["theta_b"]) * self.eps, color="#ff9f43", lw=1.0,
                 ls="--", label="shell edge (1-θ_b)ε")
        if "ctrl" in self.runs:
            r = self.runs["ctrl"]
            axx.plot(self.t, r["xi"], color="#5ec8ff", lw=1.4,
                     label="ξ controlled" + ("" if r["confined"] else "  [BREACH]"))
        if "free" in self.runs:
            axx.plot(self.t, self.runs["free"]["xi"], color="#ff8c42", lw=1.2,
                     label="ξ uncontrolled")
        axx.set_ylim(-0.02, 1.02); axx.set_xlim(0, self.t[-1])
        axx.set_ylabel("infidelity ξ = 1 − ⟨0|ρ|0⟩", color="#cfd3dc")
        axx.set_title("error vs. tolerance funnel", color="#e8eaf0", fontsize=10)
        axx.legend(**leg)
        self.cursor_xi = axx.axvline(0, color="white", lw=0.8, alpha=0.6)

        if "ctrl" in self.runs:
            r = self.runs["ctrl"]
            axu.plot(self.t, r["u"], color="#5ec8ff", lw=1.2, label="u (coherent)")
            axu.plot(self.t, r["gamma"], color="#b388ff", lw=1.2, label="γ (dissipative)")
            for yv, col in ((self.p["gmax"], "#b388ff"), (self.p["umax"], "#5ec8ff"),
                            (-self.p["umax"], "#5ec8ff")):
                axu.axhline(yv, color=col, lw=0.6, ls=":", alpha=0.7)
            axu.legend(**leg)
            axu.set_title("control effort (QP output)", color="#e8eaf0", fontsize=10)
        else:
            axu.text(0.5, 0.5, "u = γ = 0  (no controller)", transform=axu.transAxes,
                     ha="center", va="center", color="#9aa1b3", fontsize=11)
            axu.set_title("control effort", color="#e8eaf0", fontsize=10)
        axu.set_xlabel("t  (units of 1/Γ_m)", color="#cfd3dc")
        self.cursor_u = axu.axvline(0, color="white", lw=0.8, alpha=0.6)

        for key, a in self.art.items():
            if key in self.runs:
                X = self.runs[key]["x"]; a["ghost"].set_data_3d(X[:, 0], X[:, 1], X[:, 2])
            else:
                for art in a.values():
                    art.set_data_3d([], [], [])

    @staticmethod
    def _cap_circle(z, th=np.linspace(0, 2 * np.pi, 80)):
        z = min(1.0, max(-1.0, float(z)))
        rr = math.sqrt(max(0.0, 1.0 - z * z))
        return rr * np.cos(th), rr * np.sin(th), np.full_like(th, z)

    def _draw_frame(self):
        k = self.frame
        t = self.t[k]; eps_t = self.eps[k]
        # safe cap on the sphere: xi < eps  <=>  z > 1 - 2 eps
        self.cap_eps.set_data_3d(*self._cap_circle(1 - 2 * eps_t))
        self.cap_shell.set_data_3d(*self._cap_circle(1 - 2 * (1 - self.p["theta_b"]) * eps_t))

        lines = []
        for key, a in self.art.items():
            if key not in self.runs:
                continue
            r = self.runs[key]; X = r["x"]
            k0 = max(0, k - self.TAIL)
            a["trail"].set_data_3d(X[:k0 + 1, 0], X[:k0 + 1, 1], X[:k0 + 1, 2])
            a["tail"].set_data_3d(X[k0:k + 1, 0], X[k0:k + 1, 1], X[k0:k + 1, 2])
            a["dot"].set_data_3d([X[k, 0]], [X[k, 1]], [X[k, 2]])
            a["arrow"].set_data_3d([0, X[k, 0]], [0, X[k, 1]], [0, X[k, 2]])
            tag = "CTRL" if key == "ctrl" else "FREE"
            line = f"{tag}  ξ={r['xi'][k]:.4f}  |r|={np.linalg.norm(X[k]):.3f}"
            if key == "ctrl":
                line += f"  u={r['u'][k]:+.3f} γ={r['gamma'][k]:.3f} case {r['case'][k]}"
            lines.append(line)
        self.txt.set_text(f"t = {t:5.2f}   ε(t) = {eps_t:.4f}\n" + "\n".join(lines))
        self.cursor_xi.set_xdata([t, t]); self.cursor_u.set_xdata([t, t])

    def _tick(self):
        self.frame = min(self.frame + self.stride, self.n - 1)
        self._draw_frame()
        self.fig.canvas.draw_idle()
        if self.frame >= self.n - 1:
            self._stop()


def main():
    if "--headless" in sys.argv:
        import time
        matplotlib.use("Agg")
        t0 = time.perf_counter(); r = simulate(DEFAULTS, True); t1 = time.perf_counter()
        print(f"sim: {1e3 * (t1 - t0):.0f} ms  confined={r['confined']}  xi_T={r['xi'][-1]:.2e}")
        d = Demo()
        d.frame = d.n // 3; d._draw_frame()
        t0 = time.perf_counter(); d.fig.canvas.draw(); t1 = time.perf_counter()
        print(f"full redraw: {1e3 * (t1 - t0):.0f} ms")
        d.fig.savefig("bloch_demo_smoke.png", dpi=110, facecolor=d.fig.get_facecolor())
        return
    Demo()
    plt.show()


if __name__ == "__main__":
    main()
