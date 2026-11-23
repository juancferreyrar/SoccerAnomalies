from fractions import Fraction as F
from math import isqrt, log, exp
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- helpers
def fmt(x):
    """Fraction -> 'a/b (= decimal)'; integers print plainly."""
    x = F(x)
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator} (={float(x):.6g})"


def show_vec(name, v):
    print(f"{name} = [" + ", ".join(fmt(a) for a in v) + "]")


def show_mat(name, M):
    print(f"{name} =")
    for row in M:
        print("   [ " + ",  ".join(fmt(a) for a in row) + " ]")


def T(M):
    return [list(r) for r in zip(*M)]


def mm(A, B):
    Bt = T(B)
    return [[sum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def mv(A, v):
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def eye(k):
    return [[F(int(i == j)) for j in range(k)] for i in range(k)]


def diag(vals):
    k = len(vals)
    return [[vals[i] if i == j else F(0) for j in range(k)] for i in range(k)]


def fsqrt(x):
    """Exact square root of a non-negative Fraction (raises if irrational)."""
    x = F(x)
    n, d = isqrt(x.numerator), isqrt(x.denominator)
    if n * n != x.numerator or d * d != x.denominator:
        raise ValueError(f"sqrt({x}) is not rational")
    return F(n, d)


def header(txt):
    print("\n" + "=" * 72 + f"\n{txt}\n" + "=" * 72)


# ------------------------------------------------------------ raw data
names = ["P1", "P2", "P3", "P4", "P5"]
minutes = [F(1800), F(2700), F(1800), F(2700), F(2700)]
raw = [  # season totals: (progressive passes, tackles)
    [F(72), F(16)],
    [F(144), F(72)],
    [F(68), F(94)],
    [F(180), F(120)],
    [F(216), F(168)],
]
cand_name, cand_min, cand_raw = "C", F(2700), [F(114), F(207)]
m_min = F(900)
alpha = 0.05
n, d = len(raw), 2

header("INPUT")
for nm, m, c in zip(names, minutes, raw):
    print(f"{nm}: minutes = {m}, raw totals (c_i1, c_i2) = ({c[0]}, {c[1]})")
print(f"candidate {cand_name}: minutes = {cand_min}, raw = ({cand_raw[0]}, {cand_raw[1]})")
print(f"minutes cutoff m_min = {m_min}, alpha = {alpha}, n = {n}, d = {d}")

# ---------------------------------------------------- STEP 1: filter + per-90
header("STEP 1: filter (m_i >= m_min) and per-90 scaling, t = identity")
assert all(m >= m_min for m in minutes)
print("All five players have m_i >= 900, so no row is dropped.")
X = [[90 * c / m for c in row] for row, m in zip(raw, minutes)]
show_mat("X (rows = players, x_ij = 90 c_ij / m_i)", X)
x_cand = [90 * c / cand_min for c in cand_raw]
show_vec("x_C (candidate)", x_cand)

# ---------------------------------------------------- STEP 2: mean
header("STEP 2: sample mean  mu_hat = (1/n) X^T 1")
mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
show_vec("mu_hat", mu)

# ---------------------------------------------------- STEP 3: center
header("STEP 3: centered data  X_c = X - 1 mu_hat^T")
Xc = [[X[i][j] - mu[j] for j in range(d)] for i in range(n)]
show_mat("X_c", Xc)
col_sums = [sum(Xc[i][j] for i in range(n)) for j in range(d)]
show_vec("column sums of X_c (must be 0)", col_sums)
assert all(s == 0 for s in col_sums)

# ---------------------------------------------------- STEP 4: covariance
header("STEP 4: MLE covariance  Sigma_hat = (1/n) X_c^T X_c")
XtX = mm(T(Xc), Xc)
show_mat("X_c^T X_c", XtX)
S = [[XtX[i][j] / n for j in range(d)] for i in range(d)]
show_mat("Sigma_hat", S)
assert S[0][1] == S[1][0]
tr = S[0][0] + S[1][1]
det = S[0][0] * S[1][1] - S[0][1] ** 2
print(f"trace = {fmt(tr)},  det = {fmt(det)}")

# ---------------------------------------------------- STEP 5: eigendecomposition
header("STEP 5: eigendecomposition  Sigma_hat = Q Lambda Q^T  (exact, 2x2)")
print("characteristic polynomial: lambda^2 - tr*lambda + det = 0")
disc = tr * tr - 4 * det
print(f"discriminant tr^2 - 4 det = {fmt(disc)}, sqrt = {fmt(fsqrt(disc))}")
lam1 = (tr + fsqrt(disc)) / 2
lam2 = (tr - fsqrt(disc)) / 2
lams = [lam1, lam2]
show_vec("eigenvalues (lambda_1 >= lambda_2)", lams)
Q_cols = []
for lam in lams:
    # (S - lam I) v = 0  ->  v = (S12, lam - S11) is a solution when S12 != 0
    v = [S[0][1], lam - S[0][0]]
    nrm = fsqrt(dot(v, v))
    q = [a / nrm for a in v]
    Q_cols.append(q)
    print(f"lambda = {fmt(lam)}: unnormalised v = ({fmt(v[0])}, {fmt(v[1])}), "
          f"||v|| = {fmt(nrm)}, unit eigenvector q = ({fmt(q[0])}, {fmt(q[1])})")
Q = T(Q_cols)  # columns are eigenvectors
Lam = diag(lams)
show_mat("Q", Q)
show_mat("Lambda", Lam)
show_mat("Q^T Q (should be I)", mm(T(Q), Q))
assert mm(T(Q), Q) == eye(d)
assert mm(mm(Q, Lam), T(Q)) == S
print("Check: Q Lambda Q^T reproduces Sigma_hat exactly.")

# ---------------------------------------------------- STEP 6: PD check
header("STEP 6: positive-definiteness check")
kappa = lam1 / lam2
print(f"lambda_min = {fmt(lam2)} > 0  ->  Sigma_hat is positive definite")
print(f"condition number kappa = lambda_1 / lambda_2 = {fmt(kappa)}")
assert lam2 > 0

# ---------------------------------------------------- STEP 7: inverse sqrt
header("STEP 7: inverse square root  Sigma_hat^(-1/2) = Q Lambda^(-1/2) Q^T")
Lam_mhalf = diag([1 / fsqrt(l) for l in lams])
show_mat("Lambda^(-1/2)", Lam_mhalf)
S_mhalf = mm(mm(Q, Lam_mhalf), T(Q))
show_mat("Sigma_hat^(-1/2)", S_mhalf)

# ---------------------------------------------------- STEP 8: whiten
header("STEP 8: whitening  Z = X_c Sigma_hat^(-1/2)")
Z = mm(Xc, S_mhalf)
show_mat("Z", Z)

# ---------------------------------------------------- STEP 9: distances
header("STEP 9: squared Mahalanobis distances  d_i^2 = ||z_i||^2")
d2 = [dot(z, z) for z in Z]
for nm, z, v in zip(names, Z, d2):
    print(f"{nm}: z = ({fmt(z[0])}, {fmt(z[1])}),  d^2 = {fmt(z[0])}^2 + {fmt(z[1])}^2 = {fmt(v)}")
show_vec("d^2 (training)", d2)

# candidate scored with the SAME mu_hat and Sigma_hat (out of sample)
vC = [x_cand[j] - mu[j] for j in range(d)]
zC = mv(S_mhalf, vC)
d2C = dot(zC, zC)
show_vec("candidate centered v_C = x_C - mu_hat", vC)
show_vec("candidate whitened z_C", zC)
print(f"candidate d^2 = {fmt(d2C)}")

# ---------------------------------------------------- STEP 10: threshold
header("STEP 10: threshold  tau = F^{-1}_{chi2_d}(1 - alpha)")
print("For d = 2, chi^2_2 is Exponential with mean 2: F(x) = 1 - exp(-x/2),")
print("so tau = -2 ln(alpha) exactly.")
tau = -2 * log(alpha)
print(f"tau = -2 ln({alpha}) = {tau:.6f}")
print(f"n - 1 = {n - 1}: with the MLE covariance, every TRAINING d_i^2 <= n - 1 = {n-1} < tau")

# ---------------------------------------------------- STEP 11: flags
header("STEP 11: flag  a_i = 1{ d_i^2 > tau }")
for nm, v in zip(names, d2):
    print(f"{nm}: d^2 = {fmt(v)}  vs tau = {tau:.4f} -> flag = {int(float(v) > tau)}")
print(f"{cand_name}: d^2 = {fmt(d2C)}  vs tau = {tau:.4f} -> flag = {int(float(d2C) > tau)}")

# ====================================================================
header("VERIFICATION (independent routes)")

# V1: substitute eigenpairs back: S q = lambda q
print("V1  S q_k = lambda_k q_k")
for k in range(d):
    lhs = mv(S, Q_cols[k])
    rhs = [lams[k] * a for a in Q_cols[k]]
    print(f"    k={k+1}: S q = ({fmt(lhs[0])}, {fmt(lhs[1])}) = lambda q = ({fmt(rhs[0])}, {fmt(rhs[1])})  {lhs == rhs}")
    assert lhs == rhs

# V2: Sigma^{-1} by the 2x2 adjugate formula (no eigendecomposition)
S_inv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
show_mat("V2  Sigma_hat^(-1) from adjugate formula (1/det)[[s22,-s12],[-s12,s11]]", S_inv)
print("    Sigma^(-1/2) * Sigma^(-1/2) == Sigma^(-1):", mm(S_mhalf, S_mhalf) == S_inv)
print("    Sigma_hat * Sigma^(-1) == I:", mm(S, S_inv) == eye(d))
assert mm(S_mhalf, S_mhalf) == S_inv and mm(S, S_inv) == eye(d)

# V3: whitened covariance is the identity
cov_Z = [[x / n for x in r] for r in mm(T(Z), Z)]
show_mat("V3  (1/n) Z^T Z (should be I)", cov_Z)
assert cov_Z == eye(d)

# V4: d^2 computed directly from the definition (x-mu)^T Sigma^{-1} (x-mu)
d2_direct = [dot(v, mv(S_inv, v)) for v in Xc]
show_vec("V4  d^2 via (x-mu)^T Sigma^(-1) (x-mu)", d2_direct)
print("    equals whitened route:", d2_direct == d2)
assert d2_direct == d2

# V5: d^2 in principal-axis coordinates sum_k (q_k^T v)^2 / lambda_k
d2_axes = [sum(dot(Q_cols[k], v) ** 2 / lams[k] for k in range(d)) for v in Xc]
show_vec("V5  d^2 via sum_k (q_k^T v)^2 / lambda_k", d2_axes)
print("    equals whitened route:", d2_axes == d2)
assert d2_axes == d2

# V6: trace identity  sum_i d_i^2 = n d
print(f"V6  sum_i d_i^2 = {fmt(sum(d2))}  and  n*d = {n*d}  ->  {sum(d2) == n*d}")
assert sum(d2) == n * d
print("    (proof: sum_i tr(Sigma^-1 v_i v_i^T) = n tr(Sigma^-1 Sigma) = n d)")

# V7: candidate by two other routes
d2C_direct = dot(vC, mv(S_inv, vC))
d2C_axes = sum(dot(Q_cols[k], vC) ** 2 / lams[k] for k in range(d))
print(f"V7  candidate d^2: whitened = {fmt(d2C)}, direct = {fmt(d2C_direct)}, axes = {fmt(d2C_axes)}")
assert d2C == d2C_direct == d2C_axes
proj = [dot(Q_cols[k], vC) for k in range(d)]
print(f"    projections q_k^T v_C = ({fmt(proj[0])}, {fmt(proj[1])})")
print(f"    Euclidean ||v_C||^2 = {fmt(dot(vC, vC))}  (vs. max training ||v||^2 = "
      f"{fmt(max(dot(v, v) for v in Xc))}, whose d^2 is only {fmt(d2[[dot(v,v) for v in Xc].index(max(dot(v,v) for v in Xc))])})")

# V8: p-value route.  For d = 2,  P(chi2_2 > x) = exp(-x/2).  Flag <=> p < alpha.
print("V8  p-value route (flag iff p < alpha):")
for nm, v in list(zip(names, d2)) + [(cand_name, d2C)]:
    p = exp(-float(v) / 2)
    print(f"    {nm}: p = exp(-d^2/2) = {p:.5f}  < {alpha}?  {p < alpha}")

# V9: floating-point cross-check with numpy / scipy
try:
    import numpy as np
    from scipy import stats
    Xf = np.array([[float(a) for a in r] for r in X])
    Xcf = Xf - Xf.mean(axis=0)
    Sf = Xcf.T @ Xcf / n
    w, V = np.linalg.eigh(Sf)
    print(f"V9  numpy eigenvalues of Sigma_hat: {w[::-1]}   (exact: 4, 1)")
    Sih = V @ np.diag(w ** -0.5) @ V.T
    d2f = ((Xcf @ Sih) ** 2).sum(axis=1)
    print(f"    numpy d^2: {np.round(d2f, 10)}")
    d2Cf = float(((np.array([float(a) for a in vC]) @ Sih) ** 2).sum())
    print(f"    numpy d^2 (candidate): {d2Cf:.10f}")
    print(f"    scipy chi2.ppf(1-alpha, 2) = {stats.chi2.ppf(1 - alpha, 2):.6f}  (exact formula: {tau:.6f})")
    print(f"    scipy p-value of candidate = {stats.chi2.sf(d2Cf, 2):.6f}")
except ImportError:
    print("V9  numpy/scipy not available; skipped float cross-check")

print("\nPart A: all exact checks passed.")


# ====================================================================
# PART B: the SAME five players through the from-scratch functions in methods.py
# (floating point), compared with the exact fractions from Part A.
# Part A's eigendecomposition route is kept as an INDEPENDENT exact check of the
# Gauss-Jordan route used by methods.py.
# ====================================================================
import numpy as np
from methods import (my_mean, my_center, my_covariance, my_inverse,
                     my_mahalanobis2, my_threshold, my_anomalies)
from checks import report

header("PART B: methods.py functions vs exact fractions")
fl = lambda M: np.array([[float(a) for a in r] for r in M]) if isinstance(M[0], list) else np.array([float(a) for a in M])

# data preparation (per-90 scaling is done in the data script, not in methods.py)
C_raw = np.array([[float(a) for a in r] for r in raw]); m_arr = np.array([float(m) for m in minutes])
assert (m_arr >= float(m_min)).all()
Xb = 90.0 * C_raw / m_arr[:, None]
print("Step 0  X = 90 c / m =\n", Xb);                       report("Step 0 X vs exact", Xb, fl(X))

mu_b = my_mean(Xb);          print("Step 1  my_mean       mu_hat =", mu_b);    report("Step 1 mu_hat vs exact", mu_b, fl(mu))
Xc_b = my_center(Xb, mu_b);  print("Step 2  my_center     X_c =\n", Xc_b);    report("Step 2 X_c vs exact", Xc_b, fl(Xc))
S_b = my_covariance(Xc_b);   print("Step 3  my_covariance Sigma_hat =\n", S_b); report("Step 3 Sigma_hat vs exact", S_b, fl(S))
Si_b = my_inverse(S_b);      print("Step 4  my_inverse    Sigma_hat^(-1) =\n", Si_b)
report("Step 4 Sigma^-1 vs exact (adjugate)", Si_b, fl(S_inv))
report("Step 4 Sigma * Sigma^-1 vs I", S_b @ Si_b, np.eye(2))
d2_b = my_mahalanobis2(Xb, mu_b, Si_b); print("Step 5  my_mahalanobis2  d^2 =", d2_b)
report("Step 5 d^2 vs exact (eigen route)", d2_b, fl(d2))
tau_b = my_threshold(2, alpha); print(f"Step 6  my_threshold  tau = {tau_b:.6f}   (exact formula -2 ln alpha = {-2*log(alpha):.6f})")
print("Step 7  my_anomalies  flags =", my_anomalies(d2_b, tau_b).astype(int))

# candidate player scored with the SAME mu_hat and Sigma_hat^{-1}
xC_b = 90.0 * np.array([[float(c) for c in cand_raw]]) / float(cand_min)
d2C_b = my_mahalanobis2(xC_b, mu_b, Si_b)[0]
print(f"candidate: x_C = {xC_b[0]}, d^2 = {d2C_b:.12f}  (exact {float(d2C)}),  flag = {bool(my_anomalies(d2C_b, tau_b))}")
print("\nPart B: done.")
