#!/usr/bin/env python3
"""Statistical analysis of the Ice Cream × Emotion study (protocol 1.2.0), in a single run.

Usage:
    python run_analysis.py <export_folder> <output_folder>

<export_folder> is the unzipped researcher export (01_participants, 02_matching, 03_ratings, ...).
Every table is written to <output_folder>; nothing is written next to the code, so no
participant data ends up in the repository.

Analyses (see the thesis, section 3.4):
  H1  a) chi-square goodness-of-fit per flavour (one choice per participant and flavour), Holm-corrected
      b) binomial GEE: choosing a positive emoticon (F2 or F5) as a function of the flavour family
  H2  GEE with the leave-one-out agreement of the chosen pairing as predictor of log response time,
      log maximum deviation, log area under the curve and the presence of direction changes
      (trials with extreme response times excluded, Salgado-Montejo et al., 2015a)
  H3/H4  conditional logit over the five emoticons of each trial with participant-clustered robust
      standard errors: base model (emoticon constants + screen position), + taste-profile distance
      (Phase 3), + affective differences (Phase 2), and both
  Sensitivity: H3/H4 in participants who rated before matching; main models without English sessions
"""
import sys, json, math, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats, optimize
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.discrete.conditional_models import ConditionalLogit

PROTOCOL = '1.2.0'
FACES = ['F1', 'F2', 'F3', 'F4', 'F5']
FACE_NAMES = {'F1': 'Enojo', 'F2': 'Alegría', 'F3': 'Sorpresa', 'F4': 'Tristeza', 'F5': 'Sonrisa'}
POSITIVE = {'F2', 'F5'}
FAMILIES = ['sweet', 'sour', 'bitter', 'umami', 'spicy']
FAMILY_ES = {'sweet': 'Dulce', 'sour': 'Ácido', 'bitter': 'Amargo', 'umami': 'Umami', 'spicy': 'Picante'}
TASTES = ['sweet', 'sour', 'bitter', 'umami', 'spicy']
AFFECT = ['wanting', 'liking', 'tension_relaxation']
RT_SD_RULE = 2.0          # trials above mean + 2 SD of log RT are excluded from H2
MATCHING_LAST = {'AGM', 'GAM'}  # phase orders in which both rating phases preceded matching

log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); log_lines.append(s)

def fmt_p(p):
    """APA 7: exact p with three decimals and no leading zero; p < .001 when smaller."""
    if p is None or (isinstance(p, float) and math.isnan(p)): return ''
    return '< .001' if p < .001 else '= ' + f'{p:.3f}'.lstrip('0')

def with_apa(t):
    """Adds a column with the APA-formatted p next to every p-value column (raw p stays unrounded)."""
    t = t.copy()
    for col in [c for c in t.columns if c == 'p' or c.startswith('p_') or c.endswith('_p')]:
        t[col + '_APA'] = t[col].map(lambda v: 'p ' + fmt_p(v) if pd.notna(v) else '')
    return t

def save(t, path, index=False):
    t = with_apa(t); num = t.select_dtypes('number').columns
    keep = [c for c in num if not (c == 'p' or c.startswith('p_') or c.endswith('_p'))]
    t[keep] = t[keep].round(4); t.to_csv(path, index=index)

def block_wald(params, cov, names):
    """Robust Wald chi-square for H0: all coefficients in `names` are zero."""
    b = params[names].values; V = cov.loc[names, names].values
    w = float(b @ np.linalg.solve(V, b)); df = len(names)
    return w, df, float(stats.chi2.sf(w, df))

# ---------------------------------------------------------------- data
def load(export):
    r = lambda p: pd.read_csv(export / p, encoding='utf-8-sig')
    return dict(participants=r('01_participants/participants.csv'), trials=r('02_matching/matching_trials.csv'),
                alternatives=r('02_matching/choices_long.csv'), affective=r('03_ratings/ratings_affective.csv'),
                taste=r('03_ratings/ratings_gustatory.csv'))

def select_sessions(d):
    p = d['participants']
    keep = p[(p.dry_run == 0) & (p.status == 'completed') & (p.protocol_version == PROTOCOL)].copy()
    log(f'Sesiones en la exportación: {len(p)}; incluidas (reales, completas, protocolo {PROTOCOL}): {len(keep)}')
    ids = set(keep.session_id)
    out = {k: v[v.session_id.isin(ids)].copy() for k, v in d.items()}
    t, a, af, ta = out['trials'], out['alternatives'], out['affective'], out['taste']
    checks = {'15 ensayos por sesión': t.groupby('session_id').size().eq(15).all(),
              '5 alternativas por ensayo y una elegida': (a.groupby(['session_id', 'trial_index']).chosen.agg(['size', 'sum']) == [5, 1]).all().all(),
              '60 respuestas afectivas por sesión': af.groupby('session_id').size().eq(60).all(),
              '100 respuestas de sabor por sesión': ta.groupby('session_id').size().eq(100).all()}
    for k, v in checks.items(): log(f'  control de integridad - {k}: {"OK" if v else "FALLA"}')
    if not all(checks.values()): raise SystemExit('Integrity check failed')
    return out

# ---------------------------------------------------------------- sample profile and descriptives
def sample_profile(d, out):
    p = d['participants']; rows = []
    rows.append(['Participantes', '', len(p)])
    rows.append(['Edad', 'M (DE)', f'{p.age.mean():.2f} ({p.age.std():.2f})'])
    rows.append(['Edad', 'Mín.–Máx.', f'{p.age.min()}–{p.age.max()}'])
    for col, label in (('sex', 'Sexo'), ('language', 'Idioma'), ('device', 'Dispositivo'), ('phase_order', 'Orden de fases'), ('face_layout', 'Disposición de emoticones')):
        for k, n in p[col].value_counts().sort_index().items(): rows.append([label, k, f'{n} ({100 * n / len(p):.1f}%)'])
    t = pd.DataFrame(rows, columns=['Variable', 'Categoría', 'Valor']); t.to_csv(out / 'T01_perfil_muestra.csv', index=False)
    log(f'Perfil: edad M = {p.age.mean():.2f}, DE = {p.age.std():.2f}, rango {p.age.min()}–{p.age.max()}; '
        f'{dict(p.language.value_counts())}; {dict(p.device.value_counts())}')

def choice_table(trials, out=None):
    ct = pd.crosstab([trials.flavor_family, trials.flavor_name_es], trials.selected_face).reindex(columns=FACES, fill_value=0)
    pct = ct.div(ct.sum(1), axis=0) * 100
    fam = pd.crosstab(trials.flavor_family, trials.selected_face).reindex(index=FAMILIES, columns=FACES, fill_value=0)
    fam_pct = fam.div(fam.sum(1), axis=0) * 100
    if out is not None:
        ct.to_csv(out / 'T02_elecciones_por_sabor_n.csv'); pct.round(1).to_csv(out / 'T02_elecciones_por_sabor_pct.csv')
        fam.to_csv(out / 'T03_elecciones_por_familia_n.csv'); fam_pct.round(1).to_csv(out / 'T03_elecciones_por_familia_pct.csv')
    return ct

# ---------------------------------------------------------------- H1
def h1(trials, ct, out=None, label='principal'):
    rows = []
    for (fam, name), obs in ct.iterrows():
        o = obs.values; n = o.sum(); chi, p = stats.chisquare(o)
        top = obs.idxmax()
        rows.append(dict(familia=FAMILY_ES[fam], sabor=name, n=n, chi2=chi, df=4, p=p, w_cohen=math.sqrt(chi / n),
                         emoticon_mas_elegido=f'{top} ({FACE_NAMES[top]})', pct_mas_elegido=100 * obs.max() / n))
    t = pd.DataFrame(rows); t['p_holm'] = multipletests(t.p, method='holm')[1]
    if out is not None: save(t, out / 'T04_H1_chi2_por_sabor.csv')
    sig = (t.p_holm < .05).sum(); log(f'H1a ({label}): {sig} de {len(t)} sabores difieren del azar tras corrección de Holm (p < .05)')

    df = trials.assign(positive=trials.selected_face.isin(POSITIVE).astype(int))
    df['family'] = pd.Categorical(df.flavor_family, categories=FAMILIES)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = smf.gee('positive ~ C(family, Treatment("sweet"))', groups='session_id', data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        for x in w: log('  aviso GEE H1:', x.message)
    names = [n for n in m.params.index if n.startswith('C(family')]
    W, dfw, pw = block_wald(m.params, m.cov_params(), names)
    rows = []
    for n in names:
        b, se = m.params[n], m.bse[n]; lo, hi = m.conf_int().loc[n]
        rows.append(dict(contraste=f'{FAMILY_ES[n.split("T.")[1].rstrip("]")]} vs. Dulce', beta=b, ee_robusto=se, z=b / se,
                         p=m.pvalues[n], OR=math.exp(b), IC95_inf=math.exp(lo), IC95_sup=math.exp(hi)))
    obs = df.groupby('flavor_family').positive.mean().reindex(FAMILIES) * 100
    t = pd.DataFrame(rows)
    if out is not None:
        save(t, out / 'T05_H1_GEE_emoticon_positivo.csv')
        obs.round(1).rename('pct_emoticon_positivo').to_csv(out / 'T05b_H1_pct_positivo_por_familia.csv')
    log(f'H1b ({label}): GEE binomial, efecto de la familia: Wald χ²({dfw}) = {W:.2f}, p {fmt_p(pw)}; correlación de trabajo = {m.cov_struct.dep_params:.3f}')
    for r in rows: log(f"   {r['contraste']}: OR = {r['OR']:.3f}, 95% IC [{r['IC95_inf']:.3f}, {r['IC95_sup']:.3f}], z = {r['z']:.2f}, p {fmt_p(r['p'])}")
    omnibus = pd.DataFrame([dict(muestra=label, prueba='Efecto de la familia de sabor', wald_chi2=W, df=dfw, p=pw,
                                 correlacion_trabajo=float(m.cov_struct.dep_params), ensayos=len(df), participantes=df.session_id.nunique())])
    return dict(W=W, df=dfw, p=pw, contrasts=t, omnibus=omnibus)

# ---------------------------------------------------------------- H2
def add_agreement(trials):
    """Leave-one-out agreement: share of the OTHER participants who chose the same emoticon for that flavour."""
    k = trials.groupby(['stimulus_id', 'selected_face']).session_id.transform('size')
    n = trials.groupby('stimulus_id').session_id.transform('size')
    return trials.assign(agreement=(k - 1) / (n - 1))

def rt_cutoff(trials):
    lr = np.log(trials.rt_ms); return lr.mean() + RT_SD_RULE * lr.std()

def h2(trials, out, label='principal', cut=None):
    """`cut` is the log-RT cutoff computed once on the full analysis sample (re-used in sensitivity runs)."""
    df = add_agreement(trials).copy()
    df['log_rt'] = np.log(df.rt_ms)
    cut = rt_cutoff(trials) if cut is None else cut
    excluded = df[df.log_rt > cut]
    log(f'H2 ({label}): regla de atípicos = media + {RT_SD_RULE} DE del log del tiempo -> tope {math.exp(cut) / 1000:.1f} s; '
        f'excluidos {len(excluded)} de {len(df)} ensayos ({100 * len(excluded) / len(df):.1f}%), de {excluded.session_id.nunique()} participantes')
    df = df[df.log_rt <= cut].copy()
    df['agreement10'] = df.agreement * 10            # coefficient = change per 10 percentage points of agreement
    df['trackpad'] = (df.device == 'trackpad').astype(int)
    df['trial_c'] = df.trial_index - df.trial_index.mean()
    df['log_md'] = np.log1p(df.max_perpendicular_deviation_px)
    df['log_auc'] = np.log1p(df.absolute_auc_px2)
    df['any_flip'] = ((df.x_reversals_4px > 0) | (df.y_reversals_4px > 0)).astype(int)
    df['log_grab'] = np.log(df.selection_rt_ms)
    outcomes = [('log_rt', 'Tiempo de respuesta (log ms)', sm.families.Gaussian()),
                ('log_md', 'Desviación máxima (log)', sm.families.Gaussian()),
                ('log_auc', 'Área bajo la curva (log)', sm.families.Gaussian()),
                ('any_flip', 'Algún cambio de dirección (sí/no)', sm.families.Binomial()),
                # exploratory: is the decision made before the drag starts? (time from onset to grabbing the ice cream)
                ('log_grab', 'Tiempo hasta tomar el helado (log ms, exploratorio)', sm.families.Gaussian())]
    rows = []
    for y, name, fam in outcomes:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = smf.gee(f'{y} ~ agreement10 + trackpad + trial_c', groups='session_id', data=df, family=fam,
                        cov_struct=sm.cov_struct.Exchangeable()).fit()
            for x in w: log(f'  aviso GEE H2 {y}:', x.message)
        for term in ('agreement10', 'trackpad', 'trial_c'):
            b, se = m.params[term], m.bse[term]; lo, hi = m.conf_int().loc[term]
            eff = math.exp(b); effl, effh = math.exp(lo), math.exp(hi)
            rows.append(dict(muestra=label, resultado=name, predictor=term, beta=b, ee_robusto=se, z=b / se, wald_chi2=(b / se) ** 2, df=1, p=m.pvalues[term],
                             efecto='OR' if y == 'any_flip' else '% de cambio',
                             efecto_valor=eff if y == 'any_flip' else 100 * (eff - 1),
                             efecto_IC95_inf=effl if y == 'any_flip' else 100 * (effl - 1),
                             efecto_IC95_sup=effh if y == 'any_flip' else 100 * (effh - 1), n_ensayos=len(df), n_participantes=df.session_id.nunique()))
        r = rows[-3]
        log(f"   {name}: acuerdo (+10 pp) β = {r['beta']:.4f}, EE = {r['ee_robusto']:.4f}, Wald χ²(1) = {r['wald_chi2']:.2f}, p {fmt_p(r['p'])}, "
            f"{r['efecto']} = {r['efecto_valor']:.2f} [{r['efecto_IC95_inf']:.2f}, {r['efecto_IC95_sup']:.2f}]")
    t = pd.DataFrame(rows)
    desc = df.groupby(pd.cut(df.agreement, [0, .2, .4, .6, 1], include_lowest=True), observed=False).agg(
        ensayos=('rt_ms', 'size'), tr_mediana_ms=('rt_ms', 'median'), md_mediana=('max_perpendicular_deviation_px', 'median'), pct_cambio_direccion=('any_flip', 'mean'))
    desc['pct_cambio_direccion'] *= 100
    return t, desc, excluded

# ---------------------------------------------------------------- H3 / H4: conditional logit
def build_choice_data(d):
    a = d['alternatives'][['session_id', 'trial_index', 'stimulus_id', 'candidate_face', 'chosen', 'slot']].copy()
    tr = d['trials'][['session_id', 'trial_index', 'flavor_family']]
    a = a.merge(tr, on=['session_id', 'trial_index'], how='left')
    g = d['taste'].pivot_table(index=['session_id', 'stimulus_id'], columns='construct', values='value')[TASTES]
    f = d['affective'].pivot_table(index=['session_id', 'stimulus_id'], columns='construct', values='value')[AFFECT]
    cup_g = g.add_prefix('cup_'); face_g = g.add_prefix('face_')
    cup_f = f.add_prefix('cup_'); face_f = f.add_prefix('face_')
    a = a.merge(cup_g, left_on=['session_id', 'stimulus_id'], right_index=True, how='left')
    a = a.merge(face_g, left_on=['session_id', 'candidate_face'], right_index=True, how='left')
    a = a.merge(cup_f, left_on=['session_id', 'stimulus_id'], right_index=True, how='left')
    a = a.merge(face_f, left_on=['session_id', 'candidate_face'], right_index=True, how='left')
    if a.isna().any().any(): raise SystemExit('Missing ratings for some alternatives')
    a['taste_distance'] = np.sqrt(sum((a[f'cup_{t}'] - a[f'face_{t}']) ** 2 for t in TASTES))
    for k in AFFECT: a[f'diff_{k}'] = (a[f'cup_{k}'] - a[f'face_{k}']).abs()
    raw = ['taste_distance'] + [f'diff_{k}' for k in AFFECT]
    scale = {c: (a[c].mean(), a[c].std()) for c in raw}
    for c in raw: a[f'z_{c}'] = (a[c] - scale[c][0]) / scale[c][1]
    for fc in FACES[1:]: a[f'asc_{fc}'] = (a.candidate_face == fc).astype(float)
    for s in (1, 2, 3, 4): a[f'pos_{s}'] = (a.slot == s).astype(float)
    a['trial_id'] = a.session_id + '_' + a.trial_index.astype(str)
    a = a.sort_values(['trial_id', 'candidate_face']).reset_index(drop=True)
    return a, scale

def clogit(df, cols):
    """Conditional (McFadden) logit with participant-clustered sandwich standard errors."""
    X = df[cols].to_numpy(float); y = df.chosen.to_numpy(float)
    trial_codes, trial_idx = np.unique(df.trial_id.to_numpy(), return_inverse=True)
    T = len(trial_codes)
    def parts(b):
        u = X @ b
        mx = np.zeros(T); np.maximum.at(mx, trial_idx, u)  # stability
        e = np.exp(u - mx[trial_idx]); den = np.zeros(T); np.add.at(den, trial_idx, e)
        pr = e / den[trial_idx]
        return u, pr, den, mx
    def nll(b):
        u, pr, den, mx = parts(b); return -float(np.sum(y * np.log(pr)))
    def grad(b):
        u, pr, den, mx = parts(b); return -((y - pr) @ X)
    groups = [np.where(trial_idx == t)[0] for t in range(T)]
    def hessian(pr):
        # observed information: sum over trials of X'(diag(p) - pp')X
        H = np.zeros((X.shape[1], X.shape[1]))
        for idx in groups:
            Xt = X[idx]; pt = pr[idx]; xbar = pt @ Xt
            H += (Xt * pt[:, None]).T @ Xt - np.outer(xbar, xbar)
        return H
    res = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method='BFGS', options=dict(gtol=1e-8, maxiter=5000))
    b = res.x
    for _ in range(50):  # Newton refinement to full precision (the log-likelihood is concave)
        u, pr, den, mx = parts(b); step = np.linalg.solve(hessian(pr), -grad(b)); b = b + step
        if np.max(np.abs(step)) < 1e-10: break
    u, pr, den, mx = parts(b); H = hessian(pr)
    converged = bool(np.max(np.abs(grad(b))) < 1e-6)
    Hinv = np.linalg.inv(H)
    scores = (y - pr)[:, None] * X
    clusters = df.session_id.to_numpy(); G = len(np.unique(clusters))
    S = pd.DataFrame(scores).groupby(clusters).sum().to_numpy()
    meat = S.T @ S * G / (G - 1)
    V = Hinv @ meat @ Hinv
    ll = -nll(b); ll0 = T * math.log(1 / 5)
    params = pd.Series(b, index=cols); cov = pd.DataFrame(V, index=cols, columns=cols)
    return dict(params=params, cov=cov, bse=np.sqrt(np.diag(V)), ll=ll, ll0=ll0, mcfadden=1 - ll / ll0,
                aic=2 * len(cols) - 2 * ll, n_trials=T, n_participants=G, converged=converged, naive_cov=Hinv)

def h3_h4(a, out, label='principal'):
    base = [f'asc_{f}' for f in FACES[1:]] + [f'pos_{s}' for s in (1, 2, 3, 4)]
    sens = ['z_taste_distance']; aff = [f'z_diff_{k}' for k in AFFECT]
    specs = {'M0 Base': base, 'M1 Base + sabor (Fase 3)': base + sens, 'M2 Base + afecto (Fase 2)': base + aff, 'M3 Base + sabor + afecto': base + sens + aff}
    fits = {k: clogit(a, v) for k, v in specs.items()}
    rows = []
    for k, f in fits.items():
        row = dict(muestra=label, modelo=k, ensayos=f['n_trials'], participantes=f['n_participants'], log_verosimilitud=f['ll'],
                   R2_McFadden=f['mcfadden'], AIC=f['aic'], convergio=f['converged'])
        for blk, names in (('sabor', sens), ('afecto', aff)):
            if all(n in f['params'].index for n in names):
                W, df_, p = block_wald(f['params'], f['cov'], names); row[f'Wald_{blk}_chi2'] = W; row[f'Wald_{blk}_df'] = df_; row[f'Wald_{blk}_p'] = p
        rows.append(row)
    comp = pd.DataFrame(rows)
    coef_rows = []
    for k, f in fits.items():
        for n, b in f['params'].items():
            se = float(np.sqrt(f['cov'].loc[n, n])); z = b / se; p = 2 * stats.norm.sf(abs(z))
            coef_rows.append(dict(muestra=label, modelo=k, termino=n, beta=b, ee_robusto=se, z=z, p=p, OR=math.exp(b),
                                  IC95_inf=math.exp(b - 1.959964 * se), IC95_sup=math.exp(b + 1.959964 * se)))
    coefs = pd.DataFrame(coef_rows)
    m3 = fits['M3 Base + sabor + afecto']
    log(f'H3/H4 ({label}): {m3["n_trials"]} ensayos, {m3["n_participants"]} participantes; R² McFadden M0 = {fits["M0 Base"]["mcfadden"]:.3f}, '
        f'M1 = {fits["M1 Base + sabor (Fase 3)"]["mcfadden"]:.3f}, M2 = {fits["M2 Base + afecto (Fase 2)"]["mcfadden"]:.3f}, M3 = {m3["mcfadden"]:.3f}')
    for r in rows[1:]:
        parts = [f"{blk}: Wald χ²({int(r[f'Wald_{blk}_df'])}) = {r[f'Wald_{blk}_chi2']:.2f}, p {fmt_p(r[f'Wald_{blk}_p'])}" for blk in ('sabor', 'afecto') if f'Wald_{blk}_chi2' in r]
        log(f"   {r['modelo']}: " + '; '.join(parts))
    for n in sens + aff:
        c = coefs[(coefs.modelo == 'M3 Base + sabor + afecto') & (coefs.termino == n)].iloc[0]
        log(f"   M3 {n}: OR por 1 DE = {c.OR:.3f} [{c.IC95_inf:.3f}, {c.IC95_sup:.3f}], z = {c.z:.2f}, p {fmt_p(c.p)}")
    return comp, coefs, fits

def check_clogit_against_statsmodels(a, cols):
    """Cross-check: point estimates must match statsmodels' ConditionalLogit."""
    mine = clogit(a, cols)
    ref = ConditionalLogit(a.chosen.to_numpy(), a[cols].to_numpy(float), groups=a.trial_id.to_numpy()).fit(disp=0, method='newton', maxiter=200, tol=1e-12)
    diff = float(np.max(np.abs(mine['params'].to_numpy() - ref.params)))
    diff_se = float(np.max(np.abs(np.sqrt(np.diag(mine['naive_cov'])) - ref.bse)))
    log(f'Control con ConditionalLogit de statsmodels: diferencia máxima en coeficientes = {diff:.1e}, en errores estándar (no robustos) = {diff_se:.1e}')
    if diff > 1e-6 or diff_se > 1e-6: raise SystemExit('Conditional logit cross-check failed')
    return diff

# ---------------------------------------------------------------- descriptives of the rating phases
def rating_descriptives(d, out):
    for key, name in (('affective', 'T10_fase2_medias_por_estimulo.csv'), ('taste', 'T11_fase3_medias_por_estimulo.csv')):
        r = d[key]
        t = r.groupby(['stimulus_type', 'stimulus_id', 'construct']).value.agg(['mean', 'std', 'count']).round(2).reset_index()
        t.to_csv(out / name, index=False)
    proc = pd.concat([d['affective'].assign(fase='Fase 2'), d['taste'].assign(fase='Fase 3')])
    # Response mode: a direct click versus sliding the mark while the button is still pressed (first press).
    slid = proc.adjustments_json.map(lambda js: any(x.get('type') == 'move' and x.get('press') == 1 for x in json.loads(js)))
    proc = proc.assign(deslizo=slid, ajuste_posterior=proc.adjustment_presses > 0,
                       cambio_valor=proc.value_changed.astype(str).str.lower().isin(['1', 'true']))
    rows = []
    for fase, g in list(proc.groupby('fase')) + [('Total', proc)]:
        rows.append(dict(fase=fase, respuestas=len(g), pct_clic_directo=100 * (~g.deslizo).mean(), pct_deslizo=100 * g.deslizo.mean(),
                         participantes_que_deslizaron=int(g[g.deslizo].session_id.nunique()), participantes=int(g.session_id.nunique()),
                         pct_ajuste_despues_de_soltar=100 * g.ajuste_posterior.mean(), pct_cambio_de_valor=100 * g.cambio_valor.mean()))
    save(pd.DataFrame(rows), out / 'T13_modo_respuesta_escalas.csv')
    t = rows[-1]
    log(f"Escalas: {t['pct_deslizo']:.1f}% de las respuestas se dieron deslizando la marca y {t['pct_clic_directo']:.1f}% con un clic directo; "
        f"{t['participantes_que_deslizaron']} de {t['participantes']} participantes deslizaron al menos una vez; "
        f"{t['pct_ajuste_despues_de_soltar']:.1f}% con algún ajuste después del primer clic; {t['pct_cambio_de_valor']:.1f}% cambiaron de valor")

# ---------------------------------------------------------------- main
def main(export, out):
    out.mkdir(parents=True, exist_ok=True)
    log(f'Análisis del estudio Ice Cream × Emotion — protocolo {PROTOCOL}')
    log(f'Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, statsmodels {statsmodels.__version__}')
    raw = load(export); d = select_sessions(raw)
    sample_profile(d, out)
    trials = d['trials'].merge(d['participants'][['session_id', 'device']], on='session_id', how='left', suffixes=('', '_p'))
    log(f'Fase 1: {len(trials)} ensayos; cambios de elección en la ventana de corrección: {int(trials.choice_changed.sum())}; '
        f'intentos no válidos previos: {int((trials.attempt_no - 1).sum())}')
    ct = choice_table(trials, out)
    h1r = h1(trials, ct, out)
    h2t, h2desc, excl = h2(trials, out)
    h2desc.round(2).to_csv(out / 'T07_H2_descriptivos_por_acuerdo.csv')
    a, scale = build_choice_data(d)
    pd.DataFrame(scale, index=['media', 'DE']).T.round(3).to_csv(out / 'T08b_escalado_predictores.csv')
    check_clogit_against_statsmodels(a, [f'asc_{f}' for f in FACES[1:]] + [f'pos_{s}' for s in (1, 2, 3, 4)] + ['z_taste_distance'] + [f'z_diff_{k}' for k in AFFECT])
    comp, coefs, fits = h3_h4(a, out)

    log('--- Sensibilidad 1: participantes que evaluaron antes del emparejamiento (orden AGM o GAM)')
    before = set(d['participants'][d['participants'].phase_order.isin(MATCHING_LAST)].session_id)
    comp_s1, coefs_s1, _ = h3_h4(a[a.session_id.isin(before)], out, 'evaluaron antes del emparejamiento')
    log('--- Sensibilidad 2: sin sesiones en inglés')
    es = set(d['participants'][d['participants'].language == 'es'].session_id)
    tr_es = trials[trials.session_id.isin(es)]
    h1_es = h1(tr_es, choice_table(tr_es), label='sin inglés')
    h2t_es, _, _ = h2(tr_es, out, 'sin inglés', cut=rt_cutoff(trials))
    comp_s2, coefs_s2, _ = h3_h4(a[a.session_id.isin(es)], out, 'sin inglés')
    save(h1_es['contrasts'].assign(muestra='sin inglés'), out / 'T12_sens_H1_sin_ingles.csv')
    save(pd.concat([h1r['omnibus'], h1_es['omnibus']]), out / 'T05c_H1_GEE_prueba_global.csv')

    save(pd.concat([h2t, h2t_es]), out / 'T06_H2_GEE.csv')
    save(pd.concat([comp, comp_s1, comp_s2]), out / 'T08_H3H4_comparacion_modelos.csv')
    save(pd.concat([coefs, coefs_s1, coefs_s2]), out / 'T09_H3H4_coeficientes.csv')
    rating_descriptives(d, out)
    (out / 'registro_analisis.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    log(f'Resultados guardados en {out}')

if __name__ == '__main__':
    if len(sys.argv) != 3: raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
