#!/usr/bin/env python3
"""Report constrained January results with numerical/scientific gates kept separate."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from apply.tools.january_v6_nonnegative import ROOT,OLD,TAG,F,G,D,RESPONSE,context,cell_fit,npz,load_json,write,sha256,require
from apply.report.build_january_v6_report import table


def figures(background,bootstrap,coverage,spectra,toys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in background if r['required']]
    _,cache,full=context(); row=next(r for r in selected if r['cell_id']==62)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    edges=cache['x_edges_deg']; centers=(edges[1:]+edges[:-1])/2; x,y=np.meshgrid(centers,centers)
    mask=x*x+y*y<=36
    old=full@cache['surface_coefficients'][61]*cache['annulus_surface_scale'][61]
    new=full@np.array(row['coefficients'])*row['scale']
    for ax,z,label in zip(axes,(old,new),('Old unconstrained','Constrained on continuous 6 deg disk')):
        image=ax.imshow(np.where(mask,z,np.nan),origin='lower',extent=[edges[0],edges[-1],edges[0],edges[-1]],cmap='coolwarm')
        for r in [row['radius_deg'],float(cache['annulus_inner_deg'][61]),float(cache['annulus_outer_deg'][61])]:
            ax.add_patch(plt.Circle((0,0),r,fill=False,color='k',lw=.7))
        ax.set(xlim=(-6,6),ylim=(-6,6),xlabel='x (deg)',ylabel='y (deg)',title=label)
        fig.colorbar(image,ax=ax,label='counts / 0.1 deg pixel')
    fig.savefig(ROOT/'cell62_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    positions=np.arange(len(coverage))
    for ax,p in zip(axes,(68,95)):
        ax.errorbar(positions,[r[f'coverage{p}'] for r in coverage],yerr=[r['coverage_mc_se'][0 if p==68 else 1] for r in coverage],fmt='o',capsize=3)
        ax.axhline(p/100,color='k',ls='--',label='nominal')
        ax.set_xticks(positions,[f"{r['shape']}\nN={r['expected_annulus_count']}" for r in coverage],rotation=40,ha='right',fontsize=8)
        ax.set(ylabel='Coverage (all attempts)',ylim=(0,1.04),title=f'{p}% interval, cell62 geometry; bars = 1 MC SE')
    fig.savefig(ROOT/'coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    signal=npz(ROOT/'signal_propagated_errors.npz'); energy=np.geomspace(.2,500,300)
    for col,name in enumerate(('pl','logpar')):
        for mode,color in [('legacy','C0'),('propagated','C1')]:
            fit=spectra['outputs'][mode]['metadata']['fits'][name+'_conservative']
            curve=energy**2*(F.pl_flux_tev(energy,**fit['parameters'],pivot_tev=3) if name=='pl' else F.logpar_flux_tev(energy,**fit['parameters'],pivot_tev=3))
            axes[0,col].loglog(energy,curve,color=color,label=f'{mode}: chi2/dof={fit["chi2"]:.1f}/{fit["ndof"]}')
            data=npz(ROOT/mode/'fit.npz')
            axes[1,col].plot(signal['cell_id'],data[name+'_conservative_pull'],'o',ms=3,color=color,label=mode)
        axes[0,col].set(title=name.upper(),xlabel='Energy (TeV)',ylabel='E^2 dN/dE (TeV cm^-2 s^-1)');axes[0,col].legend(fontsize=8)
        axes[1,col].axhline(0,color='k',lw=.5);axes[1,col].set(xlabel='Frozen cell ID',ylabel='Residual / marginal error');axes[1,col].legend()
    fig.suptitle('Diagnostic spectrum and pulls; calibration is a separate gate')
    fig.savefig(ROOT/'spectrum_pulls.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,group in zip(axes,('nhit','predE')):
        for mode,offset in [('legacy',.985),('propagated',1.015)]:
            points=[p for p in spectra['outputs'][mode]['sed'] if p['grouping']==group]
            positive=[p for p in points if p['e2_dnde']>0]
            ax.errorbar([p['effective_energy_tev']*offset for p in positive],[p['e2_dnde'] for p in positive],yerr=[p['e2_dnde_err'] for p in positive],fmt='o',ms=4,capsize=2,label=mode)
        ax.set(xscale='log',yscale='log',xlabel='Model-weighted median true energy (TeV)',ylabel='E^2 dN/dE (TeV cm^-2 s^-1)',title=f'Diagnostic SED grouped by {group}');ax.legend()
    fig.savefig(ROOT/'sed_comparison.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',required=True)
    parser.add_argument('--renderer',type=Path,required=True)
    args=parser.parse_args()
    manifest=load_json(ROOT/'analysis_manifest.json');background=load_json(ROOT/'background_all91.json')
    boot=load_json(ROOT/'bootstrap_cells.json');coverage=load_json(ROOT/'coverage.json')
    spectra=load_json(ROOT/'spectra_acceptance.json');toys=load_json(ROOT/'spectral_toys.json')
    uncertainty=load_json(ROOT/'uncertainty_acceptance.json');design=load_json(ROOT/'pseudoexperiment_design.json')
    require(sha256(ROOT/'pseudoexperiment_design.json')==uncertainty['design_sha256'],'Frozen toy design changed')
    selected=[r for r in background if r['required']]
    require([r['cell_id'] for r in selected]==manifest['selected_cell_ids'],'Background44 order')
    # Re-run all91 small fits from the final reviewed code, binding results to current source.
    _,cache,full=context()
    for row in background:
        i=row['cell_id']-1;mask=cache['training_mask'][i]
        try:
            current=cell_fit(full[mask],cache['counts_map'][i][mask],row['radius_deg'])
        except (RuntimeError,ValueError):
            require(row['status']=='FAIL' and not row['required'],'Recomputation failed required cell')
            continue
        require(row['status']=='PASS','Candidate status changed')
        for key in ('B_on','minimum','objective','scale'):
            require(np.isclose(current[key],row[key],rtol=1e-11,atol=1e-10),f'Recomputed background changed: cell{row["cell_id"]}/{key}')
    for path,expected in manifest['reuse_hashes'].items():
        require(sha256(path)==expected,f'Reused source changed: {path}')
    parts=manifest['parquet_dataset']['parts'];data_dir=OLD/'c/runs'/f'{OLD.name}_stage_c/obs_events'
    for part in parts: require(sha256(data_dir/part['path'])==part['sha256'],'Parquet changed after run')
    jobs=[]
    for job in args.jobs.split(','):
        line=subprocess.check_output(['sacct','-n','-X','-j',job,'--format=JobID,JobName%40,State,ExitCode,Elapsed,AllocCPUS,ReqMem,Timelimit,NodeList','-P'],text=True).strip()
        fields=line.splitlines()[0].split('|')
        require(fields[0]==job,'No actual Slurm accounting')
        jobs.append(dict(zip(('id','name','state','exit_code','elapsed','cpus','memory','limit','node'),fields)))
    scientific_jobs=[j for j in jobs if j['id']!='66091']
    require(all(j['state']=='COMPLETED' and j['exit_code']=='0:0' for j in scientific_jobs),'Production job failed; do not infer success from artifacts')
    write(ROOT/'slurm_jobs.json',dict(jobs=jobs,report_job=os.environ['SLURM_JOB_ID'],report_status='RUNNING_AT_GENERATION; final delivery receipt requires completion'))
    figures(background,boot,coverage,spectra,toys)
    fits=spectra['outputs']['propagated']['metadata'];preferred=fits['preferred_fit']['model']+'_conservative'
    preferred_fit=fits['fits'][preferred]
    numeric=all(r['status']=='PASS' and r['minimum']>=-r['tolerance'] and r['B_on']>0 for r in selected)
    calibration=uncertainty['result']=='PASS'
    gof=bool(preferred_fit['valid'] and preferred_fit['p_value'] is not None and preferred_fit['p_value']>=.01)
    acceptance=dict(run_tag=TAG,data_integrity='PASS',numerical_solver='PASS' if numeric else 'FAIL',
        background_validity='PASS_FINITE_POSITIVE_INTEGRAL_AND_DISK_NONNEGATIVITY' if numeric else 'FAIL',
        uncertainty_calibration=uncertainty['result'],coverage_scope='8 truth/count cases at cell62 geometry; not all44 independent coverage certification',
        spectral_goodness_of_fit='PASS_ASYMPTOTIC_DIAGNOSTIC' if gof else 'FAIL_ASYMPTOTIC_DIAGNOSTIC',
        scientific_status='DIAGNOSTIC_ONLY_UNCERTAINTY_NOT_CERTIFIED' if not calibration else 'DIAGNOSTIC_ONLY_GEOMETRY_AND_SPECTRAL_COVERAGE_LIMITS',
        diagnostic_spectrum_and_sed=True,legacy_reproduction=False,
        report_job=os.environ['SLURM_JOB_ID'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        final_code_background_recomputation='PASS_ALL91_SAME_STATUS_AND_NUMBERS',frozen_design_sha256=sha256(ROOT/'pseudoexperiment_design.json'))
    write(ROOT/'final_acceptance.json',acceptance)
    exposure=load_json(OLD/'theta_exposure.json');gti=load_json(OLD/'gti/gti_audit.json')
    def fmt(x): return f'{x:.7g}' if isinstance(x,(float,np.floating)) else str(x)
    rows=[]
    for r,b in zip(selected,boot):
        require(r['cell_id']==b['cell_id'],'Uncertainty cell order')
        rows.append([r['cell_id'],r['N_on'],r['annulus_counts'],fmt(r['old_B_on']),fmt(r['B_on']),fmt(r['old_disk_minimum']),fmt(r['minimum']),fmt(r['old_objective']),fmt(r['objective']),r['active'],fmt(b['sd']),str(b['interval68']),str(b['interval95'])])
    fit_rows=[]
    for mode in ('legacy','propagated'):
        for key,v in spectra['outputs'][mode]['metadata']['fits'].items():
            fit_rows.append([mode,key,v['valid'],str(v['parameters']),str(v['errors']),fmt(v['chi2']),v['ndof'],fmt(v['p_value'])])
    cell62=next(r for r in selected if r['cell_id']==62)
    uncertainty62=next(b for b in boot if b['cell_id']==62)
    body=[f'''---
title: "2022 年 1 月 v6 Crab：连续 6° 非负二次背景约束版"
subtitle: "独立约束版 · 数值验收、低统计校准和诊断谱分别判定"
date: "2026-10-15"
lang: "zh-CN"
toc: true
---

本轮完成 ETO Slurm 的独立约束分析。最终科学状态：**{acceptance['scientific_status']}**。44 个必选 cell 全部保留；受约束拟合成功不能证明背景准确。谱和 SED 是诊断产物，不能作为已校准的正式科学结果。

## 分项验收
''',table(['项目','结论'],[[k,v] for k,v in acceptance.items() if k in ('data_integrity','numerical_solver','background_validity','uncertainty_calibration','spectral_goodness_of_fit','scientific_status')]),
    '''
## 方法与允许变化

背景仍为 q=c0+c1*x+c2*y+c3*x²+c4*x*y+c5*y²。x/y 继续为旧 Crab tangent offsets（degree），q 单位为 counts/0.1° 网格像素。目标函数 Σ(z-q)²/max(z,1) 与原权重完全相同，增加连续 x²+y²≤36 上 q≥0 的约束，允许接触零。8°只保留覆盖诊断。没有使用 Poisson 拟合、pooling、exp 曲面、增加epsilon或事后裁零修补。

坐标按6°归一、密度按环区均值归一后用 SLSQP 求解；连续极小值由所有内部及圆周驻点的解析候选给出，网格不是证明。约束验收容差为 1e-10×环区均值，单位 counts/pixel；求解 ftol=1e-12 为归一目标尺度。它远小于单事件密度及统计误差，不给系数增加正偏置。保存目标函数、优化器状态、迭代数、约束违例与最小值位置。数值依据见 [SciPy SLSQP 文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。

环区归一化仍为 Σz/Σmax(q_pixel,0)，孔径 B_on 仍为 scale/step²×[πr²c0+πr⁴(c3+c5)/4]。容差影响的 denominator clipping delta 逐 cell 写入背景JSON；没有将解析环积分替换旧归一化分母。必选解须有限正 B_on 和有效环区归一化。旧严格正曲面门槛在专用入口适配为连续非负；旧算法入口不变。

唯一分支为双 Rayleigh+Scheme R+analytic B_on。selector SHA256=c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6。F(r_opt)=0.7129790300890827，响应只乘一次，containment=fixed1。保留 nv/Nhit、91账本、固定44顺序、原MC权重/primary分母/投放面积/单位。负 excess不删除。PL/LogPar pivot=3 TeV，LogPar自然对数，沿用旧分组与模型加权真能量中位数，不用旧半年参数作为目标。

## 来源、完整性和曝光
''',f'''复用旧一月目录 `{OLD}`。全部681 parquet分片及关键产物 SHA256 已重新计算；smoke 从全部 Stage C 重新生成网格并用旧 Stage E 球面距离逐事件重数 N_on，逐值一致。旧失败报告、模型、预测、时间friend、current/latest均保留。没有重跑推理或时间对齐。

历史完整性核验得到681预测文件、48,726,074输入、21,559,258 Stage C事件；原完整时间69,279,369行，464异常保留。GTI={gti['merged_interval_count']}段，间隙定义活时={gti['total_live_days']:.10f}天；旧源轨迹积分曝光={exposure['source_visible_live_days']:.10f}天。没有独立硬件deadtime记录，活时不等于源曝光。

全部来源和哈希见 [analysis_manifest.json](analysis_manifest.json)，兼容性依据沿用已核验的 nominal/Scheme R/2R PSF 数组。PSF全部91半径与旧文件一致。数据集canonical tree digest `{manifest['parquet_dataset']['canonical_tree_sha256']}`。

## 全部44个cell

下表 old/new minimum 都指连续6°有效圆盘，objective为未归一的旧weighted-LS目标。区间是待校准的 plug-in bootstrap normal 区间（下端截至0只用于区间，不修改曲面）。完整系数、最小值坐标、容差、scale、归一化残差及candidate失败见 [91-cell背景JSON](background_all91.json)。
''',table(['cell','N_on','N_ann','old B','new B','old min6','new min6','old objective','new objective','约束改变','boot SD','68%区间','95%区间'],rows),
    f'''
## Cell62与其他变化

Cell62原 analytic B_on={cell62['old_B_on']:.12g}，本次={cell62['B_on']:.12g}，N_on={cell62['N_on']}、N_ann={cell62['annulus_counts']}。连续6°最小值={cell62['minimum']:.12g}、容差={cell62['tolerance']:.12g}。Bootstrap SD={uncertainty62['sd']:.8g}、条件偏差={uncertainty62['conditional_bias']:.8g}。这些是给定Poisson像素模型的结果，不是背景准确性的证明。

必选44中约束改变cell {[r['cell_id'] for r in selected if r['active']]}；其余41保留原LS最优解。原先通过孔径正值检查的48、70在6°全域约束下也改变。归一化裁剪影响最大值={max(r['clipping_denominator_delta'] for r in selected):.6g} counts（分母原尺度）；归一化残差最大绝对值={max(abs(r['normalization_residual']) for r in selected):.6g} counts。非必选失败 {[r['cell_id'] for r in background if r['status']=='FAIL']} 仅属candidate诊断，不混作固定44失败。

![Cell62旧曲面与非负约束曲面；黑圈为孔径与环区边界，未裁零](cell62_comparison.png)

## 低统计、区间覆盖与偏差

生产前锁定 [伪实验设计](pseudoexperiment_design.json)，SHA256={sha256(ROOT/'pseudoexperiment_design.json')}。每44cell做512次plug-in Poisson bootstrap；重拟合时重新计算权重及原归一化。使用样本SD作为σB，报告max(0,Bhat−zσB)至Bhat+zσB；不使用旧Hessian。4类独立真值为constant、正倾斜二次、内部接触零和圆周接触零，各N_ann=10/1000；每场景256次外层、128次内层。失败进入覆盖率分母；bias/RMSE仅对有效估计计算，失败率单列。条件plug-in偏差、RMSE、边界率、失败率逐cell见 [bootstrap_cells.json](bootstrap_cells.json)；独立真值完整结果见 [coverage.json](coverage.json)。

覆盖试验仅验证cell62环区/孔径几何，不能证明全部44几何的普遍校准。68%/95%为名义水平，实际覆盖与1σ Monte Carlo误差如下；通过预设宽容下限也不等于证明精确名义覆盖。偏差门槛额外要求 |bias/true B|≤10%。不根据谱形、覆盖结果换真值或修正参数。
''',table(['真值','N_ann','true B','bias','RMSE','boundary','failure','cov68','cov95','MC SE68/95','预设下限68/95','偏差/覆盖门槛'],[[r['shape'],r['expected_annulus_count'],fmt(r['truth_B_on']),fmt(r['bias']),fmt(r['rmse']),fmt(r['boundary_rate']),fmt(r['failure_rate']),fmt(r['coverage68']),fmt(r['coverage95']),str(r['coverage_mc_se']),str(r['acceptance_lower']),r['pass_calibration']] for r in coverage]),
    '''
![独立真值的实际覆盖率，误差棒为1σ Monte Carlo误差](coverage.png)

## 谱拟合与误差口径

旧direct-expectation误差σ²=N_on+B_on，B_on已经作为背景项。传播版用σ²=N_on+Var_boot(Bhat)替换该背景项，没有重复叠加。ON与训练环区不重叠、cell互斥，使用独立Poisson统计假设；这不包含公共系统误差或MC/PSF/曝光不确定度。两个误差口径在完全相同44cell上分别拟合PL/LogPar、构建SED。内部兼容字段仍叫conservative，机器可读error_definition明确实际含义。
''',table(['误差','模型','Minuit valid','参数','误差','chi²','dof','p（渐近）'],fit_rows),
    '''
![同一44cell的PL/LogPar诊断谱及residual/error；两种误差比较](spectrum_pulls.png)

## SED与条件传播试验

沿用旧Nhit/predE分组、冻结谱形的单参数归一化及模型加权真能量中位数；负归一化在JSON/CSV保留，log图仅显示正值。SED误差条件于冻结谱形，不能解读为全系统误差。

![两种误差口径的SED；横向轻微错开仅用于辨认，原始代表能量见CSV](sed_comparison.png)

预设PL和LogPar真值各256次条件传播，使用独立Poisson ON与已保存每cell背景bootstrap样本；每次重拟合两种谱及SED。背景方差固定在观测数据估计，故是误差传播与偏差诊断，不是全44嵌套谱覆盖率认证。参数bias/RMSE、68/95覆盖和MC误差如下。分组SED误差、偏差及覆盖见 [spectral_toys.json](spectral_toys.json)。
''',table(['真模型','误差','参数','bias','RMSE','mean error','cov68','cov95','MCSE68/95','failure'],[[r['truth_model'],r['mode'],r['parameter'],fmt(r['bias']),fmt(r['rmse']),fmt(r['mean_reported_error']),fmt(r['coverage68']),fmt(r['coverage95']),f"{r['mc_se68']:.4g}/{r['mc_se95']:.4g}",fmt(r['failure_rate'])] for r in toys['parameters']]),
    '''
## Slurm与复现交付

以下为实际sacct记录；首次smoke路径失败保留，其余科学作业均须0:0。report当前作业在生成时尚未结束，最终状态由单独delivery验证记录。不存在因有产物而忽略非零退出码的成功判断。
''',table(['job','name','state','exit','elapsed','CPU','mem','limit'],[[j[k] for k in ('id','name','state','exit_code','elapsed','cpus','memory','limit')] for j in jobs]),
    f'''
入口：`apply/tools/january_v6_nonnegative.py`；批处理：`scripts/slurm/january_v6_nonnegative.sbatch`；条件谱传播：`apply/tools/january_v6_nonnegative_spectral_toys.py`。最终代码对全部91cell重算已与生产结果一致，来源提交 `{acceptance['source_commit']}`。旧算法、预测/时间配对和冻结选择不变；没有开展Pass5共同GTI、1R、v7或其他分支。

产物：[背景](background_all91.json)、[信号及传播误差NPZ](signal_propagated_errors.npz)、[背景bootstrap NPZ](background_bootstrap.npz)、[传播版谱JSON](propagated/fit.json)、[谱残差/pull NPZ](propagated/fit.npz)、[传播版SED CSV](propagated/sed.csv)、[旧误差对照](legacy/fit.json)、[分项验收](final_acceptance.json)、[作业账本](slurm_jobs.json)。日志副本在logs/，全部关键哈希由最终delivery_receipt.json另行绑定。

## 限制与结论

连续非负性、正积分和优化收敛仅说明数值有效，不能证明背景准确。低统计偏差、边界解、失败率、覆盖率和渐近χ²拟合优度分别判断。本次不以旧半年χ²/参数为目标，不删负excess、不因谱形重新选cell、不调孔径。所有普遍无偏或全44科学校准的主张均未成立；因此仅交付诊断谱和SED，科学验收保持未通过。完整数据和源码哈希由最终收据核验。
''']
    (ROOT/'report.md').write_text('\n'.join(body))
    subprocess.run([sys.executable,str(args.renderer),str(ROOT/'report.md'),'--output',str(ROOT/'report_202201_nonnegative6.html')],check=True)
    logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    for job in jobs:
        for suffix in ('out','err'):
            path=REPO/'logs/slurm'/f'{job["name"]}_{job["id"]}.{suffix}'
            if path.exists():shutil.copy2(path,logdir/path.name)
    print(json.dumps(acceptance),flush=True)

if __name__=='__main__':main()
