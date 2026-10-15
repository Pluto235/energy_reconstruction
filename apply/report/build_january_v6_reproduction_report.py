#!/usr/bin/env python3
"""Render saved January diagnostics in the original v6 report layout, without refitting."""
import csv
from dataclasses import fields
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from apply.tools.january_v6_nonnegative import ROOT, OLD, SELECTOR, F, G, D, context, npz, load_json, sha256, require
from apply.tools.january_v6_nonnegative_delivery import Links
from apply.report.prepare_v6_scheme_r_double_rayleigh_analytic_bon_report import esc, fmt, table
from apply.report import build_v6_64748_nhit100_highEplus1_report as reference_sed

REPORT_DIR = REPO / 'apply/report'
TEMPLATE = REPORT_DIR / 'crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_analytic_bon_report.html'
REFERENCE = REPORT_DIR / 'assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-analytic-bon'
REPORT = REPORT_DIR / 'crab_sed_v6_64748_202201_nonnegative6_reproduction_report.html'
ASSETS = REPORT_DIR / 'assets/v6-64748-202201-nonnegative6-reproduction'


def rel(path):
    return os.path.relpath(path, REPORT_DIR)


def link(path, label):
    return f'<a href="{esc(rel(path))}">{esc(label)}</a>'


def figure(path, caption):
    require(path.is_file(), f'Missing image: {path}')
    return f'<figure class="figure"><a href="{esc(rel(path))}"><img loading="lazy" src="{esc(rel(path))}" alt="{esc(caption)}"></a><figcaption>{esc(caption)}</figcaption></figure>'


def profiles(cache, model, cells):
    """Use the frozen map geometry; normalize counts and model by the same observed total."""
    plt = D.setup_matplotlib()
    nhit, pred, by_key = D.prepare_grid(cells)
    edges = cache['x_edges_deg']; centers = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(centers, centers); disk = x*x + y*y <= 36
    for axis, axis_name in ((1, 'ra'), (2, 'dec')):
        observed = np.where(disk, cache['counts_map'], 0).sum(axis=axis)
        background = np.where(disk, model, 0).sum(axis=axis)
        norm = np.maximum(observed.sum(axis=1), 1)[:, None]
        for excess in (False, True):
            fig, axes = plt.subplots(len(nhit), len(pred), figsize=(25, 12), sharex=True, squeeze=False)
            for i, n in enumerate(nhit):
                for j, p in enumerate(pred):
                    ax = axes[i, j]; cell = by_key[n, p]; k = cell.index
                    if np.isfinite(model[k]).any():
                        ax.plot(centers, (observed[k]-background[k] if excess else observed[k])/norm[k], lw=.6)
                        if not excess: ax.plot(centers, background[k]/norm[k], color='C3', lw=.7)
                        else: ax.axhline(0, color='k', lw=.4)
                    else: ax.text(.5, .5, 'outside fixed 44', ha='center', transform=ax.transAxes, fontsize=5)
                    ax.set_title(f'cell {cell.cell_id}: {p}', fontsize=6)
                    ax.tick_params(labelsize=5); ax.set_xlim(-6, 6)
                    if j == 0: ax.set_ylabel(n, fontsize=6)
                    if i == len(nhit)-1: ax.set_xlabel(f'{axis_name} tangent offset (deg)', fontsize=6)
            kind = 'excess' if excess else 'counts'
            fig.suptitle(f'202201 fixed44 normalized {axis_name} {kind}; observed ROI total normalization', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .97))
            fig.savefig(ASSETS/f'normalized_{axis_name}_{kind}.png', dpi=140); plt.close(fig)


def plot_sed_overlay(points, frozen, path):
    plt = D.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6.4), constrained_layout=True)
    energy = np.geomspace(.12, 260., 320)
    ax.plot(energy, G.sed_curve(energy, model_name=frozen['model'], params=frozen['parameters'], pivot_tev=3.),
        color='#2563eb', lw=2., label='202201 Stage F LogPar')
    pass5_e, pass5_y = reference_sed.pass5_points()
    require(len(pass5_e)==len(pass5_y)==7, 'Expected seven archived Pass5 points')
    pass5_fit = G.official_pass5_point_fit()
    ax.plot(pass5_e, pass5_y, 'o', ms=5, color='#111827', label='Pass5 WCDA points', zorder=5)
    ax.plot(energy, G.sed_curve(energy, model_name='logpar', params=pass5_fit, pivot_tev=3.),
        color='#111827', lw=1.7, label='Pass5 point-fit LogPar')
    pool = G.pool1_reference_points()
    ax.errorbar([p['emed_tev'] for p in pool], [p['E2_dnde'] for p in pool], yerr=[p['E2_dnde_err'] for p in pool],
        fmt='^', color='#7f3fbf', ms=4, capsize=2, lw=.8, alpha=.65, label='WCDA-1 Pool-1 Table 1')
    for dataset, marker, color, label in [('magic_joint_crab','v','#9467bd','MAGIC'),
            ('hess_2024_stereo','D','#8c564b','H.E.S.S.'), ('hawc_2019_nn','P','#17becf','HAWC NN')]:
        refs = [p for p in G.external_reference_points(include_upper_limits=False) if p['dataset']==dataset]
        ax.errorbar([p['energy_tev'] for p in refs], [p['e2_dnde'] for p in refs], yerr=[p['e2_dnde_err'] for p in refs],
            fmt=marker, color=color, label=label, ms=3.5, capsize=2, lw=.7, alpha=.55)
    for grouping, marker, color, label in [('nhit','o','#d62728','202201 Nhit grouped'),('predE','s','#2ca02c','202201 predE grouped')]:
        selected = [p for p in points if p.grouping==grouping and p.e2_dnde>0]
        if selected:
            ax.errorbar([p.effective_energy_tev for p in selected], [p.e2_dnde for p in selected],
                yerr=[p.e2_dnde_err for p in selected], fmt=marker, color=color, label=label, ms=5, capsize=3, lw=.9, zorder=6)
    ax.set(xscale='log', yscale='log', xlim=(.12,260), xlabel='Effective true energy [TeV]',
        ylabel=r'$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]', title='202201 diagnostic SED; uncertainty NOT certified')
    ax.grid(True, which='both', alpha=.22)
    labels = ax.get_legend_handles_labels()[1]
    require('Pass5 WCDA points' in labels and 'Pass5 point-fit LogPar' in labels, 'Missing Pass5 overlay')
    require(not any('full-array PL' in label for label in labels), 'Removed reference curve returned')
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180); plt.close(fig)


def make_sed_figures(fit_meta, sed):
    points = [G.SedPoint(**{f.name: p[f.name] for f in fields(G.SedPoint)}) for p in sed['propagated']]
    frozen = G.stage_f_preferred_fit(fit_meta['propagated'])
    for name, selected in [('sed_nhit_overlay', [p for p in points if p.grouping=='nhit']), ('sed_both_overlay', points)]:
        plot_sed_overlay(selected, frozen, ASSETS/f'{name}.png')


def make_figures(background, signal, fit_arrays, fit_meta, sed):
    _, cache, full = context()
    cells = D.load_cells(SELECTOR)
    model = np.full_like(cache['counts_map'], np.nan, dtype=float)
    for row in background:
        if row['required']:
            model[row['cell_id']-1] = (full @ np.asarray(row['coefficients'])) * row['scale']
    edges = cache['x_edges_deg']; centers = (edges[1:]+edges[:-1])/2
    x, y = np.meshgrid(centers, centers); disk = x*x+y*y <= 36
    D.plot_roi_counts_grid(cache['counts_map'], cells, edges, ASSETS/'roi_counts_grid.png',
        title='202201 observed counts: all91 candidate cells; circle = 6 deg fiducial', roi_fiducial_deg=6.)
    for name, values, title in (
        ('roi_excess_grid', np.where(disk, cache['counts_map']-model, np.nan), '202201 nonnegative6 excess: fixed44; others blank'),
        ('annulus_residual_grid', np.where(cache['training_mask'], cache['counts_map']-model, np.nan), '202201 fixed44 annulus residual: counts minus constrained surface')):
        D.plot_roi_signed_grid(values, cells, edges, ASSETS/f'{name}.png', title=title,
            colorbar_label='counts / 0.1 deg pixel', roi_fiducial_deg=6., r_opt_deg=cache['r_opt_deg'])
    profiles(cache, model, cells)
    D.plot_background_grid([{'cell_id': int(cid), 'B_on': background[int(cid)-1].get('B_on', np.nan)
        if background[int(cid)-1]['required'] else np.nan} for cid in cache['cell_id']], cells, ASSETS/'on_background_grid.png')
    for mode in ('legacy', 'propagated'):
        a = fit_arrays[mode]
        F.plot_heatmap(a['logpar_conservative_pull'], a, ASSETS/f'pull_grid_{mode}.png',
            title=f'202201 LogPar diagnostic pulls: {mode} errors, fixed44', colorbar_label='residual / marginal error')
    F.plot_heatmap(signal['excess']/signal['excess_err_conservative'], signal, ASSETS/'signal_error_grid.png',
        title='202201 diagnostic excess / propagated error (not calibrated significance)', colorbar_label='excess / error')
    a = fit_arrays['propagated']
    rows = [dict(excess=a['excess'][i], error_conservative=a['excess_err_conservative'][i],
        pl_model=a['pl_conservative_model_counts'][i], logpar_model=a['logpar_conservative_model_counts'][i]) for i in range(44)]
    F.plot_model_counts(signal, rows, fit_meta['propagated']['preferred_fit'], ASSETS/'model_counts_vs_excess.png')
    points = [G.SedPoint(**{f.name: p[f.name] for f in fields(G.SedPoint)}) for p in sed['propagated']]
    make_sed_figures(fit_meta, sed)
    G.plot_ratio_points(points, ASSETS/'sed_points_ratio.png', frozen_model_label='LogPar (diagnostic only)', hide_last_prede_points=0)


def main():
    require('SLURM_JOB_ID' in os.environ, 'Generate the grid figures on ETO through Slurm')
    manifest = load_json(ROOT/'analysis_manifest.json')
    receipt = load_json(ROOT/'delivery_receipt.json'); completion = load_json(ROOT/'delivery_completion.json')
    require(sha256(ROOT/'delivery_receipt.json') == completion['delivery_receipt_sha256'], 'Delivery receipt binding changed')
    for path, digest in receipt['artifact_hashes'].items():
        require(sha256(ROOT/path) == digest, f'Certified artifact changed: {path}')
    for path, digest in manifest['reuse_hashes'].items():
        # Verify the reused selector/cache/time inputs read by this presentation build.
        if Path(path) in (SELECTOR, OLD/'background_diagnostic/background_signal_diagnostic.npz', OLD/'gti/gti_audit.json', OLD/'theta_exposure.json'):
            require(sha256(path) == digest, f'Reused presentation input changed: {path}')
    background = load_json(ROOT/'background_all91.json'); selected = [r for r in background if r['required']]
    signal = npz(ROOT/'signal_propagated_errors.npz')
    fit_arrays = {m: npz(ROOT/m/'fit.npz') for m in ('legacy', 'propagated')}
    fit_meta = {m: load_json(ROOT/m/'fit.json') for m in fit_arrays}
    sed = {m: load_json(ROOT/m/'sed.json') for m in fit_arrays}
    acceptance = load_json(ROOT/'final_acceptance.json'); coverage = load_json(ROOT/'coverage.json')
    boot = load_json(ROOT/'bootstrap_cells.json'); toys = load_json(ROOT/'spectral_toys.json')
    gti = load_json(OLD/'gti/gti_audit.json'); exposure = load_json(OLD/'theta_exposure.json')
    c = load_json(OLD/'c/runs'/f'{OLD.name}_stage_c/obs_events_metadata.json')
    with SELECTOR.open() as handle: selector = list(csv.DictReader(handle))
    ids = manifest['selected_cell_ids']
    require(len(background)==91 and len(ids)==44 and [r['cell_id'] for r in selected]==signal['cell_id'].tolist()==ids, 'Cell identity changed')
    for mode, arrays in fit_arrays.items():
        require(arrays['cell_id'].tolist()==ids, 'Fit alignment changed')
        for model in ('pl', 'logpar'):
            require(np.isclose(np.sum(arrays[f'{model}_conservative_pull']**2), fit_meta[mode]['fits'][f'{model}_conservative']['chi2']), 'Pull chi2 mismatch')
        require(len(sed[mode])==18 and all(p['n_cells']==p['n_valid_cells'] for p in sed[mode]), 'SED dropped cells')
        for group, count in (('nhit', 7), ('predE', 11)):
            points = [p for p in sed[mode] if p['grouping']==group]
            require(len(points)==count and sorted(c for p in points for c in p['cell_ids'])==ids, 'SED grouping mismatch')
    require(acceptance['scientific_status']=='DIAGNOSTIC_ONLY_UNCERTAINTY_NOT_CERTIFIED', 'Review changed scientific status')
    ASSETS.mkdir(parents=True, exist_ok=True)
    reused = {}
    for pattern in ('psf_r_opt_deg_grid.png', 'psf_effective_events_grid.png', '*_stage_b_raw_theta_profiles.png',
                    '*_stage_b_radial_psf_profiles_fit_shaded.png', '*_true_energy_cell_grid.png'):
        matches = list(REFERENCE.glob(pattern)); require(len(matches)==1, f'Ambiguous reference image: {pattern}')
        path = matches[0]; shutil.copy2(path, ASSETS/path.name); reused[rel(path)] = sha256(path)
    for name in ('cell62_comparison.png', 'coverage.png', 'spectrum_pulls.png', 'sed_comparison.png'):
        shutil.copy2(ROOT/name, ASSETS/name)
    shutil.copy2(OLD/'theta_exposure.png', ASSETS/'theta_exposure.png')
    make_figures(background, signal, fit_arrays, fit_meta, sed)
    sections = []
    def section(anchor, title, body): sections.append((anchor, title, f'<section id="{anchor}"><h2>{esc(title)}</h2>{body}</section>'))
    def p(text): return '<p>'+text+'</p>'
    def figs(items): return '<div class="figgrid">'+''.join(figure(ASSETS/name, caption) for name, caption in items)+'</div>'
    lp = fit_meta['propagated']['fits']['logpar_conservative']
    cards = [('Candidate cells', '91', '7 Nhit × 13 predE'), ('Fit cells', '44 / 44', '固定选择；负 excess 保留'),
        ('Stage C files', '681', '21,559,258 selected rows'), ('GTI live time', f"{gti['total_live_days']:.5f} d", '事件间隙定义有效活时'),
        ('Crab exposure', f"{exposure['source_visible_live_days']:.5f} d", '旧 v6 源轨迹积分'),
        ('Stage F LogPar', f"{lp['chi2']:.2f} / {lp['ndof']}", '传播误差；诊断 χ²/dof'),
        ('Stage G points', '7 + 11', 'Nhit / predE 分组'), ('Science status', '未通过', '数值通过；不确定度未认证')]
    card_html = '<div class="grid">'+''.join(f'<div class="metric"><div class="label">{esc(a)}</div><div class="value">{esc(b)}</div><div class="sub">{esc(c)}</div></div>' for a,b,c in cards)+'</div>'
    gates = [('数据完整性', 'PASS', '681 分片校验及全量独立重数已通过'), ('44-cell 数值与背景门槛', 'PASS', '正积分；连续 6° 圆盘非负（数值容差内）'),
        ('不确定度校准', 'FAIL', f"{sum(not r['pass_calibration'] for r in coverage)}/8 个真值/计数场景未通过"),
        ('谱拟合收敛', 'PASS', '两误差 × 两模型 Minuit valid'), ('谱拟合优度', 'FAIL', f"LogPar 渐近 p={lp['p_value']:.4g} < 0.01"),
        ('科学结论', 'DIAGNOSTIC ONLY', '仅交付诊断谱与 SED；非原算法严格复现')]
    section('executive', 'Executive Check · 执行与验收', card_html+table(['Gate','Status','Evidence'], [[esc(x) for x in row] for row in gates]))
    section('jobs', 'Slurm Jobs · 实际作业', p('下表来自已完成交付收据；66091 是首次 smoke 的路径解析失败，修复后重跑成功。退出成功表示对应计算完成，科学门槛单独判断。')+
        table(['Job','Stage','State','Exit','Elapsed','CPU','Memory'], [[esc(j[k]) for k in ('id','name','state','exit_code','elapsed','cpus','memory')] for j in completion['all_jobs']])+
        p(f'本页图表生成作业：<code>{os.environ["SLURM_JOB_ID"]}</code>。只读取既有结果并制图，最终退出状态见随附 report_completion.json。'))
    inputs = [(ROOT/'analysis_manifest.json','输入与冻结契约'), (OLD/'gti/gti_audit.json','原始时间与 GTI'), (SELECTOR,'固定 selector'),
        (ROOT/'background_all91.json','91-cell 背景'), (ROOT/'signal_propagated_errors.npz','44-cell 信号'), (ROOT/'propagated/fit.json','传播版谱参数'),
        (ROOT/'propagated/sed.csv','18 点 SED'), (ROOT/'delivery_completion.json','已完成科学运行收据'), (ROOT/'report_202201_nonnegative6.html','原详细验收报告'), (TEMPLATE,'原 v6 版式参考')]
    section('inputs', 'Inputs And Outputs · 输入与输出', table(['Artifact','Path'], [[esc(label),link(path,rel(path))] for path,label in inputs])+
        p(f'本次 run：<code>{esc(ROOT.name)}</code>。数据与统计量均取一月独立约束运行；源运行收据保持原始哈希。'))
    chosen = [r for r in selector if int(r['cell_id']) in ids]
    selection_rows = [[r['cell_id'],esc(r['nhit_bin']),esc(r['predE_bin']),esc(r['subset_reason']),esc(r['selection_override_flag'])] for r in chosen]
    section('selector', 'Selector Rule · 固定 44-cell', p('沿用原 MC ridge 与显式 override 后的固定选择。首个 Nhit bin 为 [100,200)，≥6 的 predE 尾部仅诊断；75 保留、90 排除。没有按一月谱形或显著性重新选 cell。')+
        p('<code>'+esc(', '.join(map(str,ids)))+'</code>')+
        '<details><summary>查看全部 44-cell 选择来源</summary>'+table(['Cell','Nhit','predE','Reason','Override'],selection_rows)+'</details>')
    section('contract', 'Analytic B_on Contract · 连续非负背景', p('q(x,y)=c₀+c₁x+c₂y+c₃x²+c₄xy+c₅y²；x/y 单位 degree，q 为 counts / 0.1° pixel。保持原 weighted-LS 目标 Σ(z−q)²/max(z,1)，增加连续 x²+y²≤36 上 q≥0 的约束，允许接触零。')+
        p('环区归一化 scale=Σz/Σmax(q_pixel,0)；解析孔径积分 B_on=scale/h² × [πr²c₀+πr⁴(c₃+c₅)/4]，h=0.1°。N_on 仍逐事件按球面距离计数。')+
        p('连续极小值由内部与圆周解析驻点候选求得，网格图不作为非负证明。容差 10⁻¹⁰×环区均值；SLSQP ftol=10⁻¹²。没有正偏置 epsilon，也没有用事后裁零替代曲面约束。')+
        '<div class="okbox">Response contract：双 Rayleigh，F(r_opt)=0.7129790300890827；Aeff_R=0.7129790300890827×Aeff_nominal 只乘一次；Stage E/F/G containment=1。</div>')
    reference_path = REFERENCE/'scheme_R_double_rayleigh_analytic_bon_comparison.json'
    historical = next(r for r in load_json(reference_path)['comparison'] if r['label']=='Scheme R 2R analytic B_on')
    compare_rows = [['旧 v6 多月基线（历史参考）',fmt(historical['phi0']),fmt(historical['alpha']),fmt(historical['beta']),f"{historical['chi2']:.2f}/{historical['ndof']}",'旧数据；不作为验收目标'],
        ['一月原无约束版','—','—','—','未拟合','cell 62 背景门槛失败']]
    for mode in fit_meta:
        fit = fit_meta[mode]['fits']['logpar_conservative']; par=fit['parameters']
        compare_rows.append([f'一月非负约束 / {mode}',fmt(par['phi0']),fmt(par['alpha']),fmt(par['beta']),f"{fit['chi2']:.2f}/{fit['ndof']}",'诊断结果'])
    section('comparison', 'Reproduction Comparison · 基线与一月结果', p('本节对应原报告的分支对照。两种一月误差口径共享同一背景与 44-cell；历史多月基线观测时间不同，不是单因素统计对照，不用 χ² 大小推断改善程度。')+
        table(['Branch','φ₀ @ 3 TeV','α','β','χ²/dof','Meaning'],compare_rows)+p(link(reference_path,'历史参数来源')))
    changed = [r for r in selected if r['active']]
    section('cell62', 'Cell 62 Diagnostic · 约束改变的 cell', table(['Cell','N_on','N_ann','Old B_on','New B_on','min q on 6°','Tolerance','Bootstrap SD'],
        [[r['cell_id'],r['N_on'],r['annulus_counts'],fmt(r['old_B_on']),fmt(r['B_on']),fmt(r['minimum']),fmt(r['tolerance']),fmt(next(b['sd'] for b in boot if b['cell_id']==r['cell_id']))] for r in changed])+
        p('约束改变 cell 48、62、70；其余 41 个保持旧可行最优解。Cell 62 的正 B_on 修复了数值可行性，但其 bootstrap SD 大于估计本身，不能据此认证背景准确性。')+
        figure(ASSETS/'cell62_comparison.png','Cell 62 原始曲面与全域非负约束曲面；黑圈为孔径和环区边界，图像未事后裁零。'))
    pull_new = fit_arrays['propagated']['logpar_conservative_pull']; pull_old=fit_arrays['legacy']['logpar_conservative_pull']
    order = np.argsort(-np.abs(pull_new))[:12]
    section('pulls', 'Large-Pull Diagnostics · 最大残差', p('以下按传播误差版 |pull| 排序，两列仅比较误差口径；没有删掉大 pull 或负 excess cell。')+
        table(['Cell','Excess','σ legacy','σ propagated','Pull legacy','Pull propagated'],[[ids[i],fmt(signal['excess'][i]),fmt(signal['excess_err_legacy'][i]),fmt(signal['excess_err_conservative'][i]),fmt(pull_old[i]),fmt(pull_new[i])] for i in order])+
        figs([('pull_grid_legacy.png','同一 44-cell：旧 direct-expectation 误差的 LogPar pull。'),('pull_grid_propagated.png','同一 44-cell：传播背景估计方差后的 LogPar pull。')]))
    section('time', 'Stage C Time Audit · 时间与曝光', table(['Quantity','Value'],[['预测文件','681'],['输入预测事件','48,726,074'],['Stage C 事件',f"{c['processing']['selected_rows']:,}"],['完整原始时间行','69,279,369'],['保留异常时间','464'],['GTI 段数',gti['merged_interval_count']],['GTI live days',fmt(gti['total_live_days'])],['Crab source days',fmt(exposure['source_visible_live_days'])],['UTC start',esc(gti['utc_start'])],['UTC stop',esc(gti['utc_stop'])]])+
        p('202201 是北京时间文件标签；逐文件按 >60 s 间隙断开并合并重叠 GTI，源轨迹使用原 v6 的 60 s 中点积分。活时与源可见曝光不同；没有独立硬件 deadtime 记录。全部预测/time friend 已验收；本轮没有重新推理或时间对齐。')+
        figure(ASSETS/'theta_exposure.png','2022 年 1 月 GTI 对应 Crab 天顶角曝光。'))
    mc_images = [(Path(name).name,'历史 v6 MC/PSF 诊断图复用；响应、分箱和 aperture 来源已核验，非一月观测图。') for name in reused]
    section('mc', 'Stage A–B Diagnostics · 响应与 PSF', p('以下五图复用指定原报告的 MC/PSF 图，逐文件哈希见 report_validation.json；一月兼容性审计验证全部 91 个 r_opt 与历史数组一致。图中历史 run 标签保留，避免冒充本次新生成 MC。')+figs(mc_images))
    section('maps', 'Stage D–E Diagnostics · 一月观测与背景', p('全部 91-cell 观测计数图来自已独立重数一致的一月缓存。背景/残差只展示固定 44-cell；其他格子留空。非必选 candidate 失败独立记录在 91-cell JSON。圆盘外不作非负认证。')+
        figure(ASSETS/'roi_counts_grid.png','一月 91-cell ROI 观测计数；圆为 6° 有效域。')+
        figure(ASSETS/'roi_excess_grid.png','一月固定 44-cell，观测减去非负约束二次曲面；负残差保留。')+
        figs([('normalized_ra_counts.png','RA tangent offset：蓝色观测、红色模型，以同一观测 ROI 总数归一。'),('normalized_ra_excess.png','RA tangent offset：归一后的观测减模型。'),('normalized_dec_counts.png','Dec offset：蓝色观测、红色模型，以同一观测 ROI 总数归一。'),('normalized_dec_excess.png','Dec offset：归一后的观测减模型。')])+
        figure(ASSETS/'annulus_residual_grid.png','实际训练环区的观测减模型，未除误差；并非谱 pull。')+
        figs([('on_background_grid.png','固定 44-cell 的解析 B_on。'),('signal_error_grid.png','excess / 传播误差仅为诊断比值，不是经校准的探测显著性。')]))
    rows=[]
    for i,r in enumerate(selected):
        rows.append([r['cell_id'],esc(signal['nhit_bin'][i]),esc(signal['predE_bin'][i]),r['N_on'],fmt(r['B_on']),fmt(signal['excess'][i]),fmt(signal['excess_err_conservative'][i]),fmt(pull_new[i]),r['active']])
    section('signal', 'Stage E Signal · 44-cell 信号账本', p('负 excess cell 4、5、6、21、75 仍在拟合中。表中 σ²=N_on+Var_boot(B̂)，以 bootstrap 背景方差替换旧 B_on 项；没有将两项重复相加。')+table(['Cell','Nhit','predE','N_on','B_on','Excess','σ propagated','LP pull','Constraint active'],rows))
    fit_rows=[]
    for mode in fit_meta:
        for name, fit in fit_meta[mode]['fits'].items():
            params='; '.join(f'{esc(k)}={fmt(v)} ± {fmt(fit["errors"][k])}' for k,v in fit['parameters'].items())
            fit_rows.append([mode,esc(name.replace('_conservative','')),params,fit['valid'],f"{fit['chi2']:.5f}/{fit['ndof']}",fmt(fit['p_value'])])
    section('fit', 'Stage F Fit · 谱拟合', p('PL：φ(E)=φ₀(E/3 TeV)⁻ᵞ。LogPar：φ(E)=φ₀ exp[−α ln(E/3 TeV)−β ln²(E/3 TeV)]。φ₀ 单位 TeV⁻¹ cm⁻² s⁻¹；pivot 固定 3 TeV，自然对数。')+
        p('legacy 指本次非负背景搭配旧误差 sqrt(N_on+B_on)，propagated 指 sqrt(N_on+Var_boot(B̂))。以下参数误差为条件统计误差，不含 MC、PSF、曝光及公共系统项。')+
        table(['Error','Model','Parameters ± 1σ','Minuit valid','χ²/dof','Asymptotic p'],fit_rows)+
        figure(ASSETS/'model_counts_vs_excess.png','逐 cell 模型预期与观测 excess：一月传播误差版，全部 44-cell。')+
        figure(ASSETS/'spectrum_pulls.png','PL / LogPar 谱形和残差；两个误差口径均为诊断结果。'))
    section('sed', 'Stage G SED · 能谱点', p('固定 Stage F 优选 LogPar 谱形后，分组拟合单参数归一化；代表能量为模型加权真能量中位数。7 个 Nhit 点和 11 个 predE 点由同一 44-cell 产生，不能作为 18 个互相独立观测再拟合。')+
        p('外部参考点沿用归档 Pass5、WCDA-1、MAGIC、H.E.S.S.、HAWC。Pass5 的 7 个点不含可用误差列，因此不补造误差棒；黑色曲线是在 log(dN/dE)–log(E/3 TeV) 空间对归档点做非加权二次拟合得到的 LogPar，仅用于视觉对照。已移除 1LHAASO WCDA full-array PL 曲线。外部数据并非共同 GTI 或相同样本，没有根据外部曲线调整本次参数。')+
        figs([('sed_nhit_overlay.png','一月 Nhit 分组 SED、Pass5 点及其点拟合 LogPar 曲线。'),('sed_both_overlay.png','一月 Nhit + predE 分组 SED、Pass5 点及其点拟合 LogPar 曲线；18 点均保留。')])+
        figs([('sed_points_ratio.png','相对本次冻结 LogPar 与归档 Pass5 点拟合曲线的比值；不隐藏末尾点。'),('sed_comparison.png','两种误差定义的 SED 对照；轻微横向错开仅用于辨认。')])+
        table(['Group','Bin','Cells','E50 [TeV]','E16–E84 [TeV]','E²dN/dE','Stat. error','χ²/dof'],[[esc(p['grouping']),esc(p['group_label']),p['n_cells'],fmt(p['effective_energy_tev']),f"{p['true_energy_p16_tev']:.4g}–{p['true_energy_p84_tev']:.4g}",fmt(p['e2_dnde']),fmt(p['e2_dnde_err']),f"{p['chi2']:.3f}/{p['ndof']}"] for p in sed['propagated']]))
    section('coverage', 'Uncertainty Calibration · 覆盖率与偏差', p('冻结设计：44-cell 各 512 次背景 bootstrap；4 类独立真值 × 两档环区计数，各 256 外层 × 128 内层。仅验证 cell 62 几何。失败保留在覆盖率分母，bias/RMSE 条件于有效估计。')+
        p('2048 次外层点估计均成功；表中 failure 是内层 bootstrap 导致区间构造失败的外层试验比例。低统计 N_ann=10 的名义 95% 区间实际覆盖仅 53.5%–84.4%。')+
        table(['Truth','N_ann','Relative bias','Interval failure','Cov 68%','Cov 95%','Gate'],[[esc(r['shape']),r['expected_annulus_count'],f"{r['bias']/r['truth_B_on']:+.3%}",f"{r['failure_rate']:.3%}",f"{r['coverage68']:.3%}",f"{r['coverage95']:.3%}",'PASS' if r['pass_calibration'] else 'FAIL'] for r in coverage])+
        figure(ASSETS/'coverage.png','独立真值覆盖率；误差棒为 1σ Monte Carlo 误差。')+
        p('另有 PL / LogPar 各 256 次条件谱传播试验，背景方差固定于观测估计，不能认证全 44-cell 嵌套谱覆盖。')+
        table(['Truth','Error','Parameter','Bias','RMSE','Coverage 95%'],[[esc(r['truth_model']),esc(r['mode']),esc(r['parameter']),fmt(r['bias']),fmt(r['rmse']),f"{r['coverage95']:.3%}"] for r in toys['parameters']]))
    section('audit', 'Metadata Audit · 来源与复现', table(['Check','Evidence'],[['原交付产物',f"{len(receipt['artifact_hashes'])} 个文件 SHA256 一致"],['固定选择与顺序','背景/信号/两误差谱均 44-cell 对齐'],['残差一致性','两误差 × 两模型 Σpull² 与保存 χ² 一致'],['SED 完整性','每个分组覆盖全部 44-cell，无丢失；7 + 11 点'],['科学标记',esc(acceptance['scientific_status'])],['分析来源提交',esc(completion['analysis_source_commit'])]])+
        p(link(ASSETS/'report_validation.json','本页输入、图表与源码 SHA256 清单')+' · '+link(ROOT/'pseudoexperiment_design.json','冻结伪实验设计')+' · '+link(ROOT/'delivery_completion.json','原分析完成收据'))+
        p('本页只复用一月结果和制作诊断图，没有重新进行一月背景/谱优化、bootstrap、推理或时间对齐。Pass5 参考曲线由原 Stage G 辅助函数对归档点拟合，用于 SED 叠加图与比值图。原 v6 报告及原一月交付文件保持原状。'))
    pars=lp['parameters']; err=lp['errors']
    section('final', 'Final v6 Fit Parameters · 最终诊断参数', table(['Parameter','Value','Conditional 1σ'],[[esc(k),fmt(v),fmt(err[k])] for k,v in pars.items()])+
        '<div class="callout"><strong>科学验收未通过。</strong>连续非负与正积分只证明数值有效；背景区间存在偏差与欠覆盖，谱的渐近拟合优度也未通过。上列谱参数和 SED 仅作诊断，不能作为已校准的正式 Crab 能谱。</div>')
    css = re.search(r'<style>(.*?)</style>', TEMPLATE.read_text(), re.S).group(1).strip()
    css += '\n.table-wrap{max-width:100%;overflow-x:auto}td{overflow-wrap:anywhere}main{min-width:0}section{scroll-margin-top:18px}.figure{min-width:0}.figure a{display:block}.figgrid{grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr))}nav{display:flex;flex-wrap:wrap;gap:8px 18px;margin:18px 0;font-size:13px}a{color:var(--accent)}details{margin:16px 0}summary{cursor:pointer;font-weight:700}code{overflow-wrap:anywhere}.status{color:var(--fail);font-weight:700}@media(max-width:600px){main{padding:20px 12px}h1{font-size:25px}.grid{grid-template-columns:1fr 1fr}.metric .value{font-size:20px}th,td{white-space:nowrap}.figgrid{grid-template-columns:1fr}}'
    title='Crab SED v6 64748 · 2022 年 1 月复现报告'
    nav='<nav aria-label="章节目录">'+''.join(f'<a href="#{anchor}">{esc(title.split(" · ")[0])}</a>' for anchor,title,_ in sections)+'</nav>'
    document=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head><body><main><header><h1>{title}</h1><p class="lede">Scheme R · Double-Rayleigh PSF · Analytic B_on · 连续 6° 全域非负二次背景约束</p><div class="callout"><strong>数值验收通过，科学验收未通过。</strong>本页沿用原 v6 报告的阶段结构与图表形式，展示一月独立约束分析的实际结果。谱与 SED 仅供诊断。</div>{nav}</header>'+''.join(s[2] for s in sections)+'</main></body></html>'
    REPORT.write_text(document)
    # This file is linked in the page and filled after the remaining links are checked.
    validation_path=ASSETS/'report_validation.json'
    parser=Links(); parser.feed(document)
    for target in parser.links:
        if target.startswith('#'): require(target[1:] in parser.ids, f'Broken anchor {target}')
        elif '://' not in target:
            dest=(REPORT_DIR/target).resolve()
            require(dest==validation_path.resolve() or dest.is_file(), f'Broken local link {target}')
    input_paths=[ROOT/name for name in ('analysis_manifest.json','delivery_receipt.json','delivery_completion.json','background_all91.json','signal_propagated_errors.npz','final_acceptance.json','coverage.json','bootstrap_cells.json','spectral_toys.json')]
    input_paths += [ROOT/m/name for m in fit_arrays for name in ('fit.npz','fit.json','sed.json')]
    input_paths += [TEMPLATE,SELECTOR,reference_path,G.OFFICIAL_PASS5_CSV,OLD/'theta_exposure.png',OLD/'theta_exposure.json',OLD/'gti/gti_audit.json',OLD/'c/runs'/f'{OLD.name}_stage_c/obs_events_metadata.json',OLD/'background_diagnostic/background_signal_diagnostic.npz']
    source_paths=[Path(__file__),Path(D.__file__),Path(F.__file__),Path(G.__file__),Path(reference_sed.__file__),REPO/'apply/report/prepare_v6_scheme_r_double_rayleigh_analytic_bon_report.py']
    for path,digest in receipt['artifact_hashes'].items(): require(sha256(ROOT/path)==digest, f'Analysis artifact modified by report: {path}')
    validation=dict(result='PASS_PRESENTATION_CHECKS',scientific_status=acceptance['scientific_status'],
        render_job=os.environ['SLURM_JOB_ID'],render_job_status='RUNNING_AT_WRITE; see report_completion.json',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        report_sha256=sha256(REPORT),fixed_cells=44,sed_points_per_mode=18,sections=len(sections),
        input_hashes={rel(p):sha256(p) for p in input_paths},source_hashes={str(p.relative_to(REPO)):sha256(p) for p in source_paths},
        reused_mc_images=reused,images={p.name:sha256(p) for p in sorted(ASSETS.glob('*.png'))},
        checks=['all original receipt artifacts unchanged before/after','44-cell identities','SED grouping exhaustiveness','sum pull squared = chi2','local links and anchors'])
    validation_path.write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if not isinstance(v,dict)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
