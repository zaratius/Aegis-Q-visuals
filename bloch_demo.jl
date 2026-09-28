# How to run
#   julia --project bloch_demo.jl              # interactive window
#   julia --project bloch_demo.jl --headless   # timing + smoke-test PNG
# Requires:  ] add GLMakie   

using GLMakie
using Random, Printf, LinearAlgebra

# parameters 
const DEFAULTS = Dict{Symbol,Float64}(
    :omega_q => 5.0, :Omega => 20.0, :Gamma_m => 1.0, :kappa => 5.0, :eta => 0.6,
    :eps0 => 0.70, :eps_T => 1e-4, :T => 4.0, :theta_b => 0.20, :lam => 0.5,
    :c => 20.0, :wgamma => 1.0, :wdelta => 1e4, :umax => 1.0, :gmax => 3.90,
    :dt => 1e-3, :seed => 0.0,
)
# rho0 = [0.78 0.08; 0.08 0.22]  ->  r0 = (2 Re rho01, -2 Im rho01, rho00 - rho11)
const R0 = (0.16, 0.0, 0.56)

# QP case codes (stored as Int8 for a compact, type-stable trace)
const CASE_NAMES = ("-", "V", "I", "II", "III", "IV")
const C_NONE, C_V, C_I, C_II, C_III, C_IV = Int8.(0:5)
casename(c::Integer) = CASE_NAMES[c + 1]

nsteps(p) = round(Int, p[:T] / p[:dt])
noise(p)  = randn(Xoshiro(round(Int, p[:seed])), nsteps(p)) .* sqrt(p[:dt])

# QP
function solve_qp(alpha, bH, bD, V, lam, wu, wg, wd, umax, gmax)
    Θ = alpha + lam * V
    Θ <= 0.0 && return 0.0, 0.0, C_V
    ν = Θ / (bH^2 / wu + bD^2 / wg + 1.0 / wd)
    u, g = -ν * bH / wu, -ν * bD / wg
    u_ok, g_ok = abs(u) <= umax, g <= gmax
    u_ok && g_ok && return u, g, C_I
    usat = u != 0.0 ? copysign(umax, u) : 0.0
    if !u_ok && g_ok
        ν2 = (alpha + bH * usat + lam * V) / (bD^2 / wg + 1.0 / wd)
        g2 = -ν2 * bD / wg
        return g2 <= gmax ? (usat, g2, C_II) : (usat, gmax, C_IV)
    end
    if u_ok && !g_ok
        ν3 = (alpha + bD * gmax + lam * V) / (bH^2 / wu + 1.0 / wd)
        u3 = -ν3 * bH / wu
        return abs(u3) <= umax ? (u3, gmax, C_III) : (copysign(umax, u3), gmax, C_IV)
    end
    return usat, gmax, C_IV
end

#sim
struct SimResult
    t::Vector{Float64}
    x::Matrix{Float64}        # n × 3 Bloch vector
    pts::Vector{Point3f}      # same, for plotting
    xi::Vector{Float64}
    eps::Vector{Float64}
    u::Vector{Float64}
    gamma::Vector{Float64}
    case::Vector{Int8}
    confined::Bool
end

function simulate(p::AbstractDict, controlled::Bool, dW::AbstractVector = noise(p))
    om, Om, Gm, kap, eta = p[:omega_q], p[:Omega], p[:Gamma_m], p[:kappa], p[:eta]
    dt, T = p[:dt], p[:T]
    n = nsteps(p)
    length(dW) == n || throw(ArgumentError("length(dW) = $(length(dW)) ≠ n = $n"))

    eps0, epsT = p[:eps0], p[:eps_T]
    r_f = -log(0.05) / T
    theta_b, lam = p[:theta_b], p[:lam]
    c, wg, wd, umax, gmax = p[:c], p[:wgamma], p[:wdelta], p[:umax], p[:gmax]
    eps_f  = 1e-2
    sq     = sqrt(eta * Gm)      # sigma_xi = -sq (1 - z^2)
    sq2    = 2.0 * sq            # Bloch diffusion prefactor
    etaGm4 = 4.0 * eta * Gm      # Milstein prefactor

    x, y, z = R0
    X = Matrix{Float64}(undef, n, 3)
    XI = Vector{Float64}(undef, n); EPS = Vector{Float64}(undef, n)
    U = zeros(n); GAM = zeros(n); CASE = fill(C_NONE, n)
    confined = true

    @inbounds for k in 1:n
        t = (k - 1) * dt
        xi = 0.5 * (1.0 - z)
        e_exp = (eps0 - epsT) * exp(-r_f * t)
        eps_t = epsT + e_exp
        xi >= eps_t && (confined = false)

        u = g = 0.0
        case = C_NONE
        if controlled
            s = max(eps_t - xi, theta_b * eps_t)
            V = -log(s / eps_t)
            sig = -sq * (1.0 - z^2)
            bH = -Om * y
            bD = -kap * xi
            eps_dot = -r_f * e_exp
            alpha = -(xi / eps_t) * eps_dot + 0.5 * (1.0 / s) * sig^2
            wu = c / (abs(bH) + eps_f)
            u, g, case = solve_qp(alpha, bH, bD, V, lam, wu, wg, wd, umax, gmax)
        end
        X[k, 1] = x; X[k, 2] = y; X[k, 3] = z
        XI[k] = xi; EPS[k] = eps_t; U[k] = u; GAM[k] = g; CASE[k] = case

        # drift
        hx, hz = 2.0 * u * Om, om
        kg = kap * g
        Fx = -hz * y - 2.0 * Gm * x - 0.5 * kg * x
        Fy = hz * x - hx * z - 2.0 * Gm * y - 0.5 * kg * y
        Fz = hx * y + kg * (1.0 - z)
        # diffusion and Milstein correction
        Gx, Gy, Gz = -sq2 * z * x, -sq2 * z * y, sq2 * (1.0 - z^2)
        q = 2.0 * z^2 - 1.0
        Mx, My, Mz = etaGm4 * x * q, etaGm4 * y * q, -2.0 * etaGm4 * z * (1.0 - z^2)
        dw = dW[k]; m = 0.5 * (dw^2 - dt)
        x += Fx * dt + Gx * dw + Mx * m
        y += Fy * dt + Gy * dw + My * m
        z += Fz * dt + Gz * dw + Mz * m
        rr = sqrt(x^2 + y^2 + z^2)
        if rr > 1.0
            x /= rr; y /= rr; z /= rr
        end
    end
    pts = Point3f.(view(X, :, 1), view(X, :, 2), view(X, :, 3))
    return SimResult(collect((0:n-1) .* dt), X, pts, XI, EPS, U, GAM, CASE, confined)
end

const INTERVAL     = 0.033   # s per animation frame (~30 fps)
const PLAY_SECONDS = 8.0     # wall-clock length of one playback
const TAIL         = 250     # bright highlighted tail, in sim steps
const DEBOUNCE     = 0.15    # s after the last slider motion before re-sim

const BG, PANEL, GRID, SPINE = "#0e0f13", "#15171e", "#2a2e38", "#3a3f4c"
const FG, FG2, FG3 = "#e8eaf0", "#cfd3dc", "#9aa1b3"
const C_CTRL, C_FREE, C_EPS, C_SHELL, C_GAM = "#5ec8ff", "#ff8c42", "#ff5c5c", "#ff9f43", "#b388ff"
const RUN_KEYS = (:ctrl, :free)
const CTRL_KEYS = (:lam, :theta_b, :eps0, :gmax, :umax, :c, :wdelta)

const SLIDER_SPECS = [
    (:lam,     "λ  (barrier decay)",    0.0,  3.0,  false),
    (:theta_b, "θ_b (buffer frac.)",    0.02, 0.9,  false),
    (:eps0,    "ε₀ (funnel start)",     0.30, 0.95, false),
    (:T,       "T  (horizon)",          1.0,  8.0,  false),
    (:gmax,    "γ_max (dissip. bound)", 0.0,  8.0,  false),
    (:umax,    "u_max (coh. bound)",    0.0,  3.0,  false),
    (:c,       "c  (u-weight scale)",   0.1,  60.0, false),
    (:wdelta,  "log₁₀ w_δ (slack)",     1.0,  6.0,  true),
    (:kappa,   "κ  (decay rate)",       0.0,  15.0, false),
    (:Omega,   "Ω  (drive amp.)",       0.0,  40.0, false),
    (:Gamma_m, "Γ_m (meas. rate)",      0.05, 4.0,  false),
    (:eta,     "η  (efficiency)",       0.05, 1.0,  false),
    (:omega_q, "ω_q (drift freq.)",     0.0,  10.0, false),
    (:seed,    "noise seed",            0.0,  99.0, false),
]

function cap_circle(z; m = 80)
    z = clamp(Float64(z), -1.0, 1.0)
    rr = sqrt(max(0.0, 1.0 - z^2))
    [Point3f(rr * cos(θ), rr * sin(θ), z) for θ in range(0, 2π; length = m)]
end

struct RunArtists
    ghost::Observable{Vector{Point3f}}
    trail::Observable{Vector{Point3f}}
    tail::Observable{Vector{Point3f}}
    dot::Observable{Vector{Point3f}}
    arrow::Observable{Vector{Point3f}}
    xi::Observable{Vector{Point2f}}        # 2-D ξ(t) trace
end
RunArtists() = RunArtists((Observable(Point3f[]) for _ in 1:5)..., Observable(Point2f[]))

mutable struct Demo
    p::Dict{Symbol,Float64}
    mode::Symbol                            # :Uncontrolled | :Controlled | :Both
    runs::Dict{Symbol,SimResult}
    frame::Int
    playing::Bool
    n::Int
    stride::Int
    t::Vector{Float64}
    eps::Vector{Float64}
    debounce::Union{Timer,Nothing}

    fig::Figure
    ax3::Axis3
    ax_xi::Axis
    ax_u::Axis
    art::Dict{Symbol,RunArtists}
    xi_plots::Dict{Symbol,Any}
    cap_eps::Observable{Vector{Point3f}}
    cap_shell::Observable{Vector{Point3f}}
    info::Observable{String}
    cursor::Observable{Vector{Float64}}
    band::Observable{Vector{Point2f}}
    funnel::Observable{Vector{Point2f}}
    shell::Observable{Vector{Point2f}}
    u_line::Observable{Vector{Point2f}}
    g_line::Observable{Vector{Point2f}}
    bounds::Observable{Vector{Float64}}
    u_plots::Vector{Any}
    noctrl_txt::Any
    sliders::Dict{Symbol,Slider}
    slider_labels::Dict{Symbol,Label}
    btn_play::Button
end

function Demo()
    set_theme!(merge(Theme(backgroundcolor = BG, textcolor = FG), theme_dark()))
    fig = Figure(size = (1550, 860), backgroundcolor = BG)

    # Bloch sphere
    ax3 = Axis3(fig[1:2, 1]; aspect = (1, 1, 1), viewmode = :fit,
                limits = (-1.1, 1.1, -1.1, 1.1, -1.1, 1.1),
                elevation = deg2rad(22), azimuth = 1.275π,
                backgroundcolor = BG, protrusions = 0,
                xypanelvisible = false, xzpanelvisible = false, yzpanelvisible = false,
                xspinesvisible = false, yspinesvisible = false, zspinesvisible = false)
    hidedecorations!(ax3)

    θ = range(0, 2π; length = 90)
    for lat in (-0.6, -0.3, 0.3, 0.6)
        rr = sqrt(1 - lat^2)
        lines!(ax3, [Point3f(rr * cos(a), rr * sin(a), lat) for a in θ]; color = "#2f3646", linewidth = 0.7)
    end
    for lon in range(0, π; length = 7)[1:end-1]
        lines!(ax3, [Point3f(cos(a) * cos(lon), cos(a) * sin(lon), sin(a)) for a in θ];
               color = "#2f3646", linewidth = 0.7)
    end
    lines!(ax3, [Point3f(cos(a), sin(a), 0) for a in θ]; color = "#59627a", linewidth = 1.2)
    for (a, lab, col) in (((0, 0, 1), "|0⟩", "#c8ccd6"), ((0, 0, -1), "|1⟩", "#c8ccd6"),
                          ((1, 0, 0), "x", "#8b93a7"), ((0, 1, 0), "y", "#8b93a7"))
        v = Point3f(a...)
        lines!(ax3, [-v, v]; color = (col, 0.5), linewidth = 1.0)
        text!(ax3, 1.15f0 * v; text = lab, color = col, fontsize = 16, align = (:center, :center))
    end
    scatter!(ax3, [Point3f(0, 0, 1)]; marker = :star5, markersize = 24,
             color = "#ffd23f", strokecolor = :black, strokewidth = 0.5)

    cap_eps   = Observable(Point3f[])
    cap_shell = Observable(Point3f[])
    lines!(ax3, cap_eps;   color = C_EPS, linewidth = 2.2)
    lines!(ax3, cap_shell; color = (C_SHELL, 0.8), linewidth = 1.3, linestyle = :dash)

    art = Dict{Symbol,RunArtists}()
    for (key, col) in ((:ctrl, C_CTRL), (:free, C_FREE))
        a = RunArtists()
        lines!(ax3, a.ghost; color = (col, 0.22), linewidth = 0.8)
        lines!(ax3, a.trail; color = (col, 0.55), linewidth = 1.6)
        lines!(ax3, a.tail;  color = key === :ctrl ? :white : "#ffd9b3", linewidth = 3.0)
        lines!(ax3, a.arrow; color = (col, 0.7), linewidth = 1.5)
        scatter!(ax3, a.dot; color = col, markersize = 14, strokecolor = :white, strokewidth = 1)
        art[key] = a
    end
    info = Observable("")
    Label(fig[1:2, 1], info; tellwidth = false, tellheight = false, halign = :left,
          valign = :top, justification = :left, color = FG, fontsize = 13,
          padding = (12, 0, 0, 12))

    # panels
    axkw = (backgroundcolor = PANEL, xgridcolor = GRID, ygridcolor = GRID,
            xtickcolor = FG2, ytickcolor = FG2, xticklabelcolor = FG2, yticklabelcolor = FG2,
            bottomspinecolor = SPINE, topspinecolor = SPINE,
            leftspinecolor = SPINE, rightspinecolor = SPINE, titlecolor = FG, titlesize = 14)
    ax_xi = Axis(fig[1, 2]; title = "error vs. tolerance funnel",
                 ylabel = L"\xi = 1 - \langle 0|\rho|0\rangle", ylabelcolor = FG2, axkw...)
    ax_u  = Axis(fig[2, 2]; title = "control effort (QP output)",
                 xlabel = "t  (units of 1/Γ_m)", xlabelcolor = FG2, axkw...)
    linkxaxes!(ax_xi, ax_u)

    band   = Observable(Point2f[])
    funnel = Observable(Point2f[])
    shell  = Observable(Point2f[])
    cursor = Observable([0.0])
    poly!(ax_xi, band; color = (C_EPS, 0.08), strokewidth = 0)
    lines!(ax_xi, funnel; color = C_EPS, linewidth = 1.8, label = "funnel ε(t)")
    lines!(ax_xi, shell;  color = C_SHELL, linewidth = 1.2, linestyle = :dash,
           label = "shell edge (1-θ_b)ε")
    xi_plots = Dict{Symbol,Any}(
        :ctrl => lines!(ax_xi, art[:ctrl].xi; color = C_CTRL, linewidth = 1.6, label = "ξ controlled"),
        :free => lines!(ax_xi, art[:free].xi; color = C_FREE, linewidth = 1.4, label = "ξ uncontrolled"),
    )
    vlines!(ax_xi, cursor; color = (:white, 0.6), linewidth = 1)
    axislegend(ax_xi; position = :rt, labelsize = 11, backgroundcolor = PANEL,
               framecolor = SPINE, labelcolor = FG)

    u_line = Observable(Point2f[])
    g_line = Observable(Point2f[])
    bounds = Observable([DEFAULTS[:gmax], DEFAULTS[:umax], -DEFAULTS[:umax]])
    u_plots = Any[
        lines!(ax_u, u_line; color = C_CTRL, linewidth = 1.4, label = "u (coherent)"),
        lines!(ax_u, g_line; color = C_GAM,  linewidth = 1.4, label = "γ (dissipative)"),
        hlines!(ax_u, bounds; color = [(C_GAM, 0.7), (C_CTRL, 0.7), (C_CTRL, 0.7)],
                linewidth = 1, linestyle = :dot),
    ]
    vlines!(ax_u, cursor; color = (:white, 0.6), linewidth = 1)
    axislegend(ax_u; position = :rt, labelsize = 11, backgroundcolor = PANEL,
               framecolor = SPINE, labelcolor = FG)
    noctrl_txt = text!(ax_u, 0.5, 0.5; text = "u = γ = 0  (no controller)", space = :relative,
                       align = (:center, :center), color = FG3, fontsize = 15, visible = false)

    # controls
    gl = GridLayout(fig[1:2, 3]; tellheight = false, valign = :top)
    Label(gl[1, 1], "SCCQS: measured qubit → |0⟩"; color = FG, fontsize = 16,
          font = :bold, halign = :left)
    Label(gl[2, 1], "blue = controlled   orange = uncontrolled"; color = FG3,
          fontsize = 12, halign = :left)
    menu = Menu(gl[3, 1]; options = ["Uncontrolled", "Controlled", "Both (same noise)"],
                default = "Both (same noise)", textcolor = FG,
                cell_color_inactive_even = PANEL, cell_color_inactive_odd = PANEL,
                cell_color_hover = SPINE, cell_color_active = "#2a4a60",
                selection_cell_color_inactive = GRID, dropdown_arrow_color = FG)

    sg = SliderGrid(gl[4, 1],
        [(label = lab,
          range = key === :seed ? (0:99) : range(lo, hi; length = 301),
          startvalue = islog ? log10(DEFAULTS[key]) : DEFAULTS[key],
          format = key === :seed ? (v -> string(round(Int, v))) : (v -> @sprintf("%.3g", v)))
         for (key, lab, lo, hi, islog) in SLIDER_SPECS]...;
        width = 330, tellheight = true)
    sliders = Dict{Symbol,Slider}()
    slider_labels = Dict{Symbol,Label}()
    for (i, (key, _, _, _, _)) in enumerate(SLIDER_SPECS)
        sliders[key] = sg.sliders[i]
        slider_labels[key] = sg.labels[i]
        sg.labels[i].color = FG2; sg.labels[i].fontsize = 12
        sg.valuelabels[i].color = FG; sg.valuelabels[i].fontsize = 12
    end

    bkw = (buttoncolor = GRID, buttoncolor_hover = "#3a4256", buttoncolor_active = "#4a5470",
           labelcolor = FG, labelcolor_hover = FG, labelcolor_active = FG)
    bl = GridLayout(gl[5, 1])
    btn_play  = Button(bl[1, 1]; label = "Play",  width = 150, bkw...)
    btn_rerun = Button(bl[1, 2]; label = "Rerun", width = 150, bkw...)
    btn_seed  = Button(bl[2, 1:2]; label = "New noise seed", width = 306, bkw...)
    rowgap!(gl, 10)

    colsize!(fig.layout, 1, Relative(0.42))
    colsize!(fig.layout, 3, Fixed(360))

    d = Demo(Dict(DEFAULTS), :Both, Dict{Symbol,SimResult}(), 1, false, 1, 1,
             Float64[], Float64[], nothing,
             fig, ax3, ax_xi, ax_u, art, xi_plots, cap_eps, cap_shell, info, cursor,
             band, funnel, shell, u_line, g_line, bounds, u_plots, noctrl_txt,
             sliders, slider_labels, btn_play)

    for (key, _, _, _, islog) in SLIDER_SPECS
        on(sliders[key].value) do v
            d.p[key] = islog ? 10.0^v : Float64(v)
            schedule_rerun!(d)                        # re-sim after drag settles
        end
    end
    on(menu.selection) do lab
        d.mode = Symbol(first(split(lab)))
        set_controls_enabled!(d)
        rerun!(d)
    end
    on(_ -> toggle_play!(d), btn_play.clicks)
    on(_ -> rerun!(d), btn_rerun.clicks)
    on(btn_seed.clicks) do _
        set_close_to!(d.sliders[:seed], mod(round(Int, d.p[:seed]) + 1, 100))
    end

    set_controls_enabled!(d)
    rerun!(d)
    return d
end

function set_controls_enabled!(d::Demo)
    col = d.mode === :Uncontrolled ? "#555b69" : FG2
    for k in CTRL_KEYS
        d.slider_labels[k].color = col
    end
end

function schedule_rerun!(d::Demo)
    d.debounce === nothing || close(d.debounce)
    d.debounce = Timer(DEBOUNCE) do _
        try
            rerun!(d)
        catch err
            @error "re-simulation failed" exception = (err, catch_backtrace())
        end
    end
end

function stop!(d::Demo)
    d.playing = false
    d.btn_play.label[] = "Play"
end

function toggle_play!(d::Demo)
    d.playing && return stop!(d)
    d.frame >= d.n && (d.frame = 1)
    d.playing = true
    d.btn_play.label[] = "Pause"
    errormonitor(@async begin
        while d.playing && d.frame < d.n
            d.frame = min(d.frame + d.stride, d.n)
            draw_frame!(d)
            sleep(INTERVAL)
        end
        d.playing && stop!(d)
    end)
end

function rerun!(d::Demo)
    stop!(d)
    p = d.p
    dW = noise(p)                                  # shared noise → "Both (same noise)"
    empty!(d.runs)
    d.mode in (:Controlled, :Both)   && (d.runs[:ctrl] = simulate(p, true, dW))
    d.mode in (:Uncontrolled, :Both) && (d.runs[:free] = simulate(p, false, dW))
    ref = first(values(d.runs))
    d.n, d.t, d.eps = length(ref.t), ref.t, ref.eps
    d.stride = max(1, floor(Int, d.n / (PLAY_SECONDS / INTERVAL)))
    draw_static_panels!(d)
    d.frame = 1
    draw_frame!(d)
end

function draw_static_panels!(d::Demo)
    t, eps, θb = d.t, d.eps, d.p[:theta_b]
    d.band[]   = vcat(Point2f.(t, eps), Point2f.(reverse(t), 0))
    d.funnel[] = Point2f.(t, eps)
    d.shell[]  = Point2f.(t, (1 - θb) .* eps)

    for key in RUN_KEYS
        a = d.art[key]
        if haskey(d.runs, key)
            r = d.runs[key]
            a.xi[] = Point2f.(t, r.xi)
            a.ghost[] = r.pts
            d.xi_plots[key].visible = true
        else
            a.xi[] = Point2f[]
            for o in (a.ghost, a.trail, a.tail, a.dot, a.arrow)
                o[] = Point3f[]
            end
            d.xi_plots[key].visible = false
        end
    end
    breach = haskey(d.runs, :ctrl) && !d.runs[:ctrl].confined
    d.ax_xi.title = breach ? "error vs. tolerance funnel   [CONTROLLED BREACH]" :
                             "error vs. tolerance funnel"

    hasctrl = haskey(d.runs, :ctrl)
    if hasctrl
        r = d.runs[:ctrl]
        d.u_line[] = Point2f.(t, r.u)
        d.g_line[] = Point2f.(t, r.gamma)
        d.bounds[] = [d.p[:gmax], d.p[:umax], -d.p[:umax]]
    else
        d.u_line[] = Point2f[]; d.g_line[] = Point2f[]
    end
    foreach(pl -> pl.visible = hasctrl, d.u_plots)
    d.noctrl_txt.visible = !hasctrl
    d.ax_u.title = hasctrl ? "control effort (QP output)" : "control effort"

    xlims!(d.ax_xi, 0, t[end]); ylims!(d.ax_xi, -0.02, 1.02)
    if hasctrl
        lo = min(minimum(d.runs[:ctrl].u), -d.p[:umax], 0.0)
        hi = max(maximum(d.runs[:ctrl].gamma), maximum(d.runs[:ctrl].u), d.p[:gmax], d.p[:umax])
        pad = 0.05 * max(hi - lo, 1e-3)
        ylims!(d.ax_u, lo - pad, hi + pad)
    else
        ylims!(d.ax_u, -1, 1)
    end
end

function draw_frame!(d::Demo)
    k = d.frame
    t, e = d.t[k], d.eps[k]
    # safe cap on the sphere: ξ < ε  ⇔  z > 1 − 2ε
    d.cap_eps[]   = cap_circle(1 - 2e)
    d.cap_shell[] = cap_circle(1 - 2 * (1 - d.p[:theta_b]) * e)

    io = IOBuffer()
    @printf(io, "t = %5.2f   ε(t) = %.4f", t, e)
    for key in RUN_KEYS
        haskey(d.runs, key) || continue
        r, a = d.runs[key], d.art[key]
        P = r.pts
        k0 = max(1, k - TAIL)
        a.trail[] = P[1:k0]
        a.tail[]  = P[k0:k]
        a.dot[]   = [P[k]]
        a.arrow[] = [Point3f(0, 0, 0), P[k]]
        @printf(io, "\n%s  ξ=%.4f  |r|=%.3f", key === :ctrl ? "CTRL" : "FREE",
                r.xi[k], norm(@view r.x[k, :]))
        key === :ctrl && @printf(io, "  u=%+.3f γ=%.3f case %s",
                                 r.u[k], r.gamma[k], casename(r.case[k]))
    end
    d.info[] = String(take!(io))
    d.cursor[] = [t]
    return nothing
end

function main(args = ARGS)
    if "--headless" in args
        GLMakie.activate!(visible = false)
        simulate(DEFAULTS, true)                               
        t0 = time_ns(); r = simulate(DEFAULTS, true); t1 = time_ns()
        @printf("sim: %.1f ms  confined=%s  xi_T=%.2e\n", (t1 - t0) / 1e6, r.confined, r.xi[end])
        d = Demo()
        d.frame = d.n ÷ 3; draw_frame!(d)
        save("bloch_demo_smoke.png", d.fig)
        println("wrote bloch_demo_smoke.png")
        return
    end
    d = Demo()
    screen = display(d.fig)
    wait(screen)                      
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end