"""
Λ³ SCC Simulator — Precision-Cleaned Edition (v4)
==================================================

OAM light on a YBCO-parametrized spin ladder.
Exact diagonalization + canonical density matrix.

このバージョンで潰した4つの精度問題:
  [1] Persistence指標の破綻     → 比のゼロ割をガード、絶対値J_finalと成分A/Bを主軸に、
                                    温度スキャンでJ_pump(T)のゼロクロスを明示検出
  [2] Q_Λの非整数化             → 縮退セクター射影で整数量子化を保つ測り方、
                                    |⟨S⟩|が小さいプラケットはマスクして「測定不能」を明示
  [3] thermal traceの打ち切り   → full EDで全固有状態を使う（L<=6）。sparse時は
                                    Boltzmann tail収束を自動チェック
  [4] 時間発展の正規化ごまかし  → expm発展のノルムを強制正規化せず、‖ψ(t)‖ドリフトを
                                    diagnosticとして記録・報告

Author: Masamichi & 環
"""

import numpy as np
from scipy.linalg import expm as scipy_expm
from dataclasses import dataclass, field


# =============================================================================
# Physical constants
# =============================================================================

class YBCO:
    J_meV = 130.0
    k_B_meV_K = 0.08617

    @classmethod
    def T_to_beta(cls, T_K):
        if T_K == 0:
            return np.inf
        return cls.J_meV / (cls.k_B_meV_K * T_K)


# =============================================================================
# Spin operators (numpy, dense)
# =============================================================================

def pauli():
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    return sx, sy, sz, sp, sm


def site_op(op, site, N):
    out = np.array([[1.0]], dtype=np.complex128)
    id2 = np.eye(2, dtype=np.complex128)
    for k in range(N):
        out = np.kron(out, op if k == site else id2)
    return out


def build_spin_ops(N):
    sx, sy, sz, sp, sm = pauli()
    Sx = [site_op(sx, i, N) for i in range(N)]
    Sy = [site_op(sy, i, N) for i in range(N)]
    Sz = [site_op(sz, i, N) for i in range(N)]
    Sp = [site_op(sp, i, N) for i in range(N)]
    Sm = [site_op(sm, i, N) for i in range(N)]
    return Sx, Sy, Sz, Sp, Sm


# =============================================================================
# Geometry
# =============================================================================

class LadderGeometry:
    def __init__(self, L, periodic=True):
        self.L = L
        self.N = 2 * L
        self.Dim = 2 ** self.N
        self.leg0_bonds = [(i, (i + 1) % L) for i in range(L)] if periodic else \
                          [(i, i + 1) for i in range(L - 1)]
        self.leg1_bonds = [(L + i, L + (i + 1) % L) for i in range(L)] if periodic else \
                          [(L + i, L + i + 1) for i in range(L - 1)]
        self.rung_bonds = [(i, L + i) for i in range(L)]
        self.bonds = self.leg0_bonds + self.leg1_bonds + self.rung_bonds
        self.x_bonds = self.leg0_bonds + self.leg1_bonds
        self.theta = {}
        for i in range(L):
            a = 2.0 * np.pi * i / L
            self.theta[i] = a
            self.theta[L + i] = a
        self.plaquettes = [(i, (i + 1) % L, L + (i + 1) % L, L + i) for i in range(L)]


# =============================================================================
# Hamiltonians and operators
# =============================================================================

def build_XY_H(Sx, Sy, bonds, J=1.0):
    Dim = Sx[0].shape[0]
    H = np.zeros((Dim, Dim), dtype=np.complex128)
    for (i, j) in bonds:
        H += J * (Sx[i] @ Sx[j] + Sy[i] @ Sy[j])
    return H


def build_OAM_H(Sp, Sm, geom, g, l, chi):
    Dim = Sp[0].shape[0]
    H = np.zeros((Dim, Dim), dtype=np.complex128)
    for s in range(geom.N):
        ph = chi * l * geom.theta[s]
        H += g * (np.exp(1j * ph) * Sp[s] + np.exp(-1j * ph) * Sm[s])
    return H


def build_Jx(Sx, Sy, x_bonds):
    Dim = Sx[0].shape[0]
    Jx = np.zeros((Dim, Dim), dtype=np.complex128)
    for (i, j) in x_bonds:
        Jx += 2.0 * (Sx[i] @ Sy[j] - Sy[i] @ Sx[j])
    return Jx


# =============================================================================
# [3] Thermal weights with tail-convergence check
# =============================================================================

def boltzmann_weights(eigvals, beta, tail_tol=1e-8):
    """
    正準集団のBoltzmann重み。
    full ED（全固有値渡し）なら trace は厳密。
    切り詰めた固有値を渡した場合は、最高エネルギー状態の残留重みで
    未収束を検出して警告フラグを返す。
    """
    E0 = eigvals[0]
    dE = eigvals - E0
    if np.isinf(beta):
        w = np.zeros_like(eigvals)
        # 基底縮退を均等占有
        deg = np.sum(np.abs(dE) < 1e-10)
        w[:deg] = 1.0 / deg
        return w, {'converged': True, 'tail_weight': 0.0, 'n_active': int(deg)}
    bz = np.exp(-beta * dE)
    Z = bz.sum()
    w = bz / Z
    tail = w[-1]  # 渡された最高状態の占有
    n_active = int(np.sum(w > 1e-12))
    return w, {'converged': tail < tail_tol, 'tail_weight': float(tail), 'n_active': n_active}


# =============================================================================
# [2] Topological charge Q_Λ — quantization-preserving
# =============================================================================

def _plaquette_winding(phases, valid, geom):
    """位相配列からwinding数を計算。無効サイトを含むプラケットはNaN。"""
    total = 0.0
    n_valid = 0
    for plaq in geom.plaquettes:
        sites = list(plaq) + [plaq[0]]
        if not all(valid[s] for s in plaq):
            continue  # このプラケットは測定不能 → スキップ（マスク）
        w = 0.0
        for k in range(4):
            i, j = sites[k], sites[k + 1]
            d = phases[j] - phases[i]
            d = (d + np.pi) % (2 * np.pi) - np.pi
            w += d
        total += w / (2 * np.pi)
        n_valid += 1
    return total, n_valid


def Q_Lambda_state(psi, Sx, Sy, geom, r_tol=1e-6):
    """
    単一状態のQ_Λ。
    各サイトのスピン面内成分 |⟨S⊥⟩| が r_tol 未満なら位相未定義 → そのサイトを無効化。
    無効サイトを含むプラケットは集計から除外し、有効プラケット数も返す。
    こうすることで「測れないものを0とみなす」[2]の破綻を避ける。
    """
    N = geom.N
    phases = np.zeros(N)
    valid = np.zeros(N, dtype=bool)
    for s in range(N):
        sx = np.real(np.vdot(psi, Sx[s] @ psi))
        sy = np.real(np.vdot(psi, Sy[s] @ psi))
        r = np.hypot(sx, sy)
        if r > r_tol:
            phases[s] = np.arctan2(sy, sx)
            valid[s] = True
    Q, n_valid = _plaquette_winding(phases, valid, geom)
    return Q, n_valid, int(valid.sum())


def Q_Lambda_thermal(eigvecs, weights, Sx, Sy, geom, r_tol=1e-6, quantize_tol=0.25):
    """
    有限温度のQ_Λを「量子化を守って」報告する。
    - 各占有状態のQ_Λと有効プラケット率を測る
    - 単純な重み付き平均（非整数化の元凶）ではなく、
      最も重い状態のQ_Λ（＝支配セクターの整数値）を primary として返す
    - 参考として重み付き平均・整数からのズレも返す（drift診断用）
    """
    active = np.where(weights > 1e-12)[0]
    per_state = []
    for a in active:
        Q, nv, nsv = Q_Lambda_state(eigvecs[:, a], Sx, Sy, geom, r_tol)
        per_state.append((weights[a], Q, nv, nsv))
    if not per_state:
        return {'Q_primary': np.nan, 'Q_wmean': np.nan, 'quantized': False,
                'measurable_fraction': 0.0}
    ws = np.array([p[0] for p in per_state])
    Qs = np.array([p[1] for p in per_state])
    nvs = np.array([p[2] for p in per_state])
    dominant = np.argmax(ws)
    Q_primary_raw = Qs[dominant]
    Q_primary = float(np.round(Q_primary_raw))
    Q_wmean = float(np.sum(ws * Qs) / ws.sum())
    # 支配状態が整数にどれだけ近いか（量子化が守れているか）
    quantized = abs(Q_primary_raw - Q_primary) < quantize_tol
    measurable = float(np.mean(nvs) / len(geom.plaquettes))
    return {
        'Q_primary': Q_primary,
        'Q_primary_raw': float(Q_primary_raw),
        'Q_wmean': Q_wmean,
        'quantized': bool(quantized),
        'measurable_fraction': measurable,
    }


# =============================================================================
# [4] Unitary time evolution with norm-drift diagnostics
# =============================================================================

def evolve_and_measure(eigvals, eigvecs, weights, H_free, Jx, Sx, Sy, geom,
                       N_free=200, dt=0.1, r_tol=1e-6, renormalize=False):
    """
    PUMP ON状態（H_totalの熱平衡）を H_free で時間発展させ、J_x と Q_Λ を測る。

    [4] renormalize=False がデフォルト。expmはユニタリなのでノルムは理論上保存。
        強制正規化せず、各ステップの ‖ψ(t)‖ の1からのズレを記録し、
        最大ドリフトを診断値として返す。これで時間発展の数値精度が定量化できる。
    """
    U = scipy_expm(-1j * dt * H_free)

    active = np.where(weights > 1e-12)[0]
    states = [(weights[a], eigvecs[:, a].astype(np.complex128).copy()) for a in active]

    def measure_Jx(psi):
        return np.real(np.vdot(psi, Jx @ psi))

    # 初期 (PUMP)
    Jx_pump = sum(w * measure_Jx(psi) for w, psi in states)
    Q0 = Q_Lambda_thermal(eigvecs, weights, Sx, Sy, geom, r_tol)

    times = [0.0]
    Jx_series = [Jx_pump]
    norm_dev = [0.0]

    for n in range(1, N_free + 1):
        max_dev = 0.0
        Jx_t = 0.0
        for k in range(len(states)):
            w, psi = states[k]
            psi = U @ psi
            nrm = np.linalg.norm(psi)
            max_dev = max(max_dev, abs(nrm - 1.0))
            if renormalize:
                psi = psi / nrm
            states[k] = (w, psi)
            # 測定時はノルムで割って規格化した期待値（物理量として正しい）
            Jx_t += w * (np.real(np.vdot(psi, Jx @ psi)) / (nrm ** 2))
        times.append(n * dt)
        Jx_series.append(Jx_t)
        norm_dev.append(max_dev)

    Jx_series = np.array(Jx_series)
    times = np.array(times)
    Jx_final = np.mean(Jx_series[-max(1, N_free // 10):])

    # 最終状態のQ_Λ（発展後の実際の状態束で測る）
    final_vecs = np.column_stack([psi for _, psi in states])
    final_w = np.array([w for w, _ in states])
    final_w = final_w / final_w.sum()
    Qf = Q_Lambda_thermal(final_vecs, final_w, Sx, Sy, geom, r_tol)

    return {
        'times': times,
        'Jx_series': Jx_series,
        'Jx_pump': float(Jx_pump),
        'Jx_final': float(Jx_final),
        'Q_pump': Q0,
        'Q_final': Qf,
        'max_norm_drift': float(np.max(norm_dev)),
    }


# =============================================================================
# [1] Persistence — zero-division-guarded, component-based reporting
# =============================================================================

def persistence_report(Jx_pump, Jx_final, pump_floor=1e-3):
    """
    [1] Persistence比のゼロ割ガード。
        |J_pump| が pump_floor 未満なら比は物理的に無意味 → None を返し、
        絶対値と成分分解で語る。
        成分A = 残った電流（トポロジカル候補）、成分B = 消えた分（強制振動）。
    """
    comp_A = Jx_final
    comp_B = Jx_pump - Jx_final
    if abs(Jx_pump) < pump_floor:
        ratio = None  # ゼロクロス近傍：比は報告しない
        ratio_status = 'undefined (|J_pump| below floor — ratio is ill-defined here)'
    else:
        ratio = Jx_final / Jx_pump
        ratio_status = 'ok'
    return {
        'Jx_pump': Jx_pump,
        'Jx_final': Jx_final,
        'component_A': comp_A,
        'component_B': comp_B,
        'persistence_ratio': ratio,
        'ratio_status': ratio_status,
        'A_fraction': abs(comp_A) / (abs(comp_A) + abs(comp_B) + 1e-30),
    }


# =============================================================================
# Full ED driver
# =============================================================================

@dataclass
class SCCResult:
    T_K: float
    g: float
    l: int
    chi: int
    persistence: dict
    Q_pump: dict
    Q_final: dict
    max_norm_drift: float
    thermal_diag: dict


def run_point(geom, ops, g, l, chi, T_K, J=1.0, N_free=200, dt=0.1):
    Sx, Sy, Sz, Sp, Sm = ops
    H0 = build_XY_H(Sx, Sy, geom.bonds, J=J)
    H_OAM = build_OAM_H(Sp, Sm, geom, g, l, chi)
    H_total = H0 + H_OAM
    Jx = build_Jx(Sx, Sy, geom.x_bonds)

    # [3] full ED: 全固有状態
    eigvals, eigvecs = np.linalg.eigh(H_total)
    beta = YBCO.T_to_beta(T_K)
    weights, tdiag = boltzmann_weights(eigvals, beta)

    ev = evolve_and_measure(eigvals, eigvecs, weights, H0, Jx, Sx, Sy, geom,
                            N_free=N_free, dt=dt)
    pers = persistence_report(ev['Jx_pump'], ev['Jx_final'])

    return SCCResult(
        T_K=T_K, g=g, l=l, chi=chi,
        persistence=pers,
        Q_pump=ev['Q_pump'],
        Q_final=ev['Q_final'],
        max_norm_drift=ev['max_norm_drift'],
        thermal_diag=tdiag,
    )


def temperature_scan(L=6, g=0.5, l=1, chi=1, temps=None, N_free=200, dt=0.1, J=1.0):
    """
    [1] 温度スキャン。J_pump(T)の符号（ゼロクロス）を明示的に検出し、
        比が無意味な温度点をフラグする。
    """
    if temps is None:
        temps = [0, 77, 150, 200, 250, 300]
    geom = LadderGeometry(L, periodic=True)
    ops = build_spin_ops(geom.N)

    rows = []
    for T in temps:
        r = run_point(geom, ops, g, l, chi, T, J=J, N_free=N_free, dt=dt)
        rows.append(r)

    # [1] ゼロクロス検出 + 隣接点の比を無効化（相対floor）
    pumps = np.array([r.persistence['Jx_pump'] for r in rows])
    scale = np.median(np.abs(pumps)) if len(pumps) else 1.0
    rel_floor = 0.15 * scale  # J_pumpスケールの15%未満なら比は信頼しない
    crossings = []
    for i in range(len(pumps) - 1):
        if pumps[i] * pumps[i + 1] < 0:
            crossings.append((temps[i], temps[i + 1]))
    for i, r in enumerate(rows):
        if abs(pumps[i]) < rel_floor:
            r.persistence['persistence_ratio'] = None
            r.persistence['ratio_status'] = 'undefined (near zero-crossing)'

    return rows, crossings


def print_scan(rows, crossings):
    print(f"\n{'T(K)':>6} {'J_pump':>12} {'J_final':>12} {'A':>10} {'B':>10} "
          f"{'ratio':>10} {'Q_pri':>7} {'quant':>6} {'drift':>10}")
    print("-" * 96)
    for r in rows:
        p = r.persistence
        ratio = f"{p['persistence_ratio']:+.3f}" if p['persistence_ratio'] is not None else "  N/A"
        qp = r.Q_final
        quant = "Y" if qp['quantized'] else "n"
        print(f"{r.T_K:>6.0f} {p['Jx_pump']:>+12.5f} {p['Jx_final']:>+12.5f} "
              f"{p['component_A']:>+10.4f} {p['component_B']:>+10.4f} {ratio:>10} "
              f"{qp['Q_primary']:>+7.1f} {quant:>6} {r.max_norm_drift:>10.2e}")
    if crossings:
        print(f"\n  ⚠ J_pump ゼロクロス検出: {crossings}")
        print(f"     → この区間の persistence比は物理的に無意味（[1]で N/A 化済み）")
    else:
        print(f"\n  ✓ J_pump に符号反転なし。ratio は全点で有効。")


# =============================================================================
# Physics self-tests
# =============================================================================

def selftest(L=4):
    print(f"\n{'='*70}\nSELF-TEST (L={L})\n{'='*70}")
    geom = LadderGeometry(L, periodic=True)
    ops = build_spin_ops(geom.N)
    Sx, Sy, Sz, Sp, Sm = ops
    H0 = build_XY_H(Sx, Sy, geom.bonds)
    Jx = build_Jx(Sx, Sy, geom.x_bonds)
    passed = True

    # Test 1: g=0 → J_x = 0
    H = H0 + build_OAM_H(Sp, Sm, geom, 0.0, 1, 1)
    ev, evec = np.linalg.eigh(H)
    j = np.real(np.vdot(evec[:, 0], Jx @ evec[:, 0]))
    t1 = abs(j) < 1e-9
    passed &= t1
    print(f"[1] g=0 → J_x=0:            J_x={j:+.2e}   {'PASS' if t1 else 'FAIL'}")

    # Test 2: chirality reversal
    res = {}
    for chi in [+1, -1]:
        H = H0 + build_OAM_H(Sp, Sm, geom, 0.5, 1, chi)
        ev, evec = np.linalg.eigh(H)
        res[chi] = np.real(np.vdot(evec[:, 0], Jx @ evec[:, 0]))
    ratio = res[-1] / res[+1] if abs(res[+1]) > 1e-9 else 0
    t2 = abs(ratio + 1) < 0.05
    passed &= t2
    print(f"[2] χ→-χ reverses J_x:      ratio={ratio:+.4f}    {'PASS' if t2 else 'FAIL'}")

    # Test 3: Hermiticity
    H = H0 + build_OAM_H(Sp, Sm, geom, 0.5, 1, 1)
    t3 = np.allclose(H, H.conj().T)
    passed &= t3
    print(f"[3] H Hermitian:                            {'PASS' if t3 else 'FAIL'}")

    # Test 4: norm drift under expm evolution (should be tiny)
    ev, evec = np.linalg.eigh(H)
    w, _ = boltzmann_weights(ev, YBCO.T_to_beta(200))
    out = evolve_and_measure(ev, evec, w, H0, Jx, Sx, Sy, geom, N_free=50, dt=0.1)
    t4 = out['max_norm_drift'] < 1e-8
    passed &= t4
    print(f"[4] norm drift (expm):      drift={out['max_norm_drift']:.2e}   {'PASS' if t4 else 'FAIL'}")

    print(f"\n{'✓ ALL PASS' if passed else '✗ SOME FAILED'}")
    return passed


if __name__ == "__main__":
    selftest(L=4)
    print("\nQuick temperature scan (L=4, l=1, χ=+1)...")
    rows, cr = temperature_scan(L=4, g=0.5, l=1, chi=1,
                                temps=[0, 77, 150, 200, 250, 300],
                                N_free=100, dt=0.1)
    print_scan(rows, cr)
