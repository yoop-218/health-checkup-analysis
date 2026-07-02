# -*- coding: utf-8 -*-
"""
건강검진 데이터 분석 유틸리티 함수 모음

analysis.ipynb에서 정의한 분석·시각화 함수를 재사용 가능한 형태로 분리한 모듈입니다.
원본 노트북은 전역 변수(df, num_cols, cat_vars)에 의존했으나,
이 모듈에서는 모두 함수 인자로 받아 어떤 데이터프레임에도 적용할 수 있도록 리팩터링했습니다.

사용 예:
    import pandas as pd
    from utils import run_2group, run_multigroup, posthoc, plot_box, run_chi2, plot_crosstab

    df = pd.read_csv('data.csv')
    num_cols = ['age', 'bmi', 'sbp', 'dbp', 'glucose', 'hdl', 'ldl', 'triglyceride']

    plot_box(df, 'metabolic_syndrome', num_cols)
    result = run_2group(df, 'metabolic_syndrome', num_cols)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_box(df, group_col, num_cols):
    """그룹별 박스플롯 — 연속형 변수들을 한 화면에 배치.

    Parameters
    ----------
    df : pandas.DataFrame
        분석 대상 데이터프레임.
    group_col : str
        그룹을 나누는 범주형 변수명.
    num_cols : list of str
        박스플롯을 그릴 연속형 변수명 목록.
    """
    n = len(num_cols)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
    for ax, y in zip(np.ravel(axes), num_cols):
        sns.boxplot(data=df, x=group_col, y=y,
                    hue=group_col, palette='colorblind', legend=False, ax=ax)
        ax.set_title(y)
        ax.set_xlabel('')
    # 남는 축 숨김
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    plt.suptitle(f'그룹: {group_col}', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_crosstab(df, cat_vars, target='metabolic_syndrome'):
    """교차표 시각화 일괄 — 각 범주형 변수별 target 비율을 누적 막대로 표시.

    Parameters
    ----------
    df : pandas.DataFrame
        분석 대상 데이터프레임.
    cat_vars : list of str
        비율을 볼 범주형 변수명 목록.
    target : str, default 'metabolic_syndrome'
        비율의 기준이 되는 이진/범주형 결과 변수명.
    """
    n = len(cat_vars)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, cat_vars):
        (pd.crosstab(df[c], df[target], normalize='index')
           .plot(kind='bar', stacked=True, colormap='Blues',
                 ax=ax, legend=False))
        ax.set_title(c)
        ax.set_xlabel('')
        ax.set_ylabel('비율')
        ax.tick_params(axis='x', rotation=0)
    plt.suptitle(f'변수별 {target} 비율', fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 집단 비교 검정 (가정 점검 후 검정 자동 선택)
# ─────────────────────────────────────────────────────────────
def run_2group(df, group_col, num_cols):
    """2집단 비교 — 정규성·등분산 점검 후 검정을 자동 선택.

    정규성 불충족 → Mann-Whitney U
    정규 + 등분산  → Student t
    정규 + 이분산  → Welch t
    효과크기는 Cohen's d (전체 표준편차 기준).

    Returns
    -------
    pandas.DataFrame
        변수별 검정 방법·통계량·p값·Cohen_d·유의 표시 (p 오름차순 정렬).
    """
    a, b = sorted(df[group_col].dropna().unique())
    rows = []
    for y in num_cols:
        g1 = df[df[group_col] == a][y].dropna()
        g2 = df[df[group_col] == b][y].dropna()

        normal = (stats.shapiro(g1).pvalue > 0.05) and (stats.shapiro(g2).pvalue > 0.05)
        equal_var = stats.levene(g1, g2).pvalue > 0.05

        if not normal:
            stat, p = stats.mannwhitneyu(g1, g2)
            method = 'Mann-Whitney'
        elif equal_var:
            stat, p = stats.ttest_ind(g1, g2, equal_var=True)
            method = 'Student t'
        else:
            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
            method = 'Welch t'

        d = (g1.mean() - g2.mean()) / df[y].std()
        rows.append([y, method, round(stat, 2), round(p, 4), round(d, 2)])

    out = pd.DataFrame(rows, columns=['변수', '검정', '통계량', 'p', 'Cohen_d'])
    out['유의'] = np.where(out['p'] < 0.05, '*', '')
    return out.sort_values('p')


def run_multigroup(df, group_col, num_cols):
    """3집단 이상 비교 — 가정 점검 후 검정을 자동 선택.

    정규성 불충족 → Kruskal-Wallis
    정규 + 등분산  → 일원분산분석(ANOVA)
    정규 + 이분산  → Welch ANOVA (Alexander-Govern)
    효과크기는 eta-squared.

    Returns
    -------
    pandas.DataFrame
        변수별 검정 방법·통계량·p값·eta2·유의 표시 (p 오름차순 정렬).
    """
    rows = []
    for y in num_cols:
        groups = [g[y].dropna().values for _, g in df.groupby(group_col)]

        normal = all(stats.shapiro(g).pvalue > 0.05 for g in groups)
        equal_var = stats.levene(*groups).pvalue > 0.05

        if not normal:
            stat, p = stats.kruskal(*groups)
            method = 'Kruskal'
        elif equal_var:
            stat, p = stats.f_oneway(*groups)
            method = 'ANOVA'
        else:
            res = stats.alexandergovern(*groups)
            stat, p = res.statistic, res.pvalue
            method = 'Welch ANOVA'

        all_vals = np.concatenate(groups)
        ss_between = sum(len(g) * (g.mean() - all_vals.mean()) ** 2 for g in groups)
        ss_total = ((all_vals - all_vals.mean()) ** 2).sum()
        eta2 = ss_between / ss_total

        rows.append([y, method, round(stat, 2), round(p, 4), round(eta2, 3)])

    out = pd.DataFrame(rows, columns=['변수', '검정', '통계량', 'p', 'eta2'])
    out['유의'] = np.where(out['p'] < 0.05, '*', '')
    return out.sort_values('p')


def posthoc(df, group_col, y, method):
    """사후검정 — 유의한 조합에서 '어느 그룹 쌍이 다른가'를 확인.

    정규(ANOVA 계열) → Tukey HSD
    비정규(Kruskal)  → Dunn (Bonferroni 보정)

    Parameters
    ----------
    df : pandas.DataFrame
    group_col : str
        그룹 변수명.
    y : str
        종속(연속형) 변수명.
    method : str
        run_multigroup 결과의 '검정' 값. 'Kruskal'이면 Dunn, 그 외는 Tukey.
    """
    print(f'[{group_col} × {y}]  ({method})')
    sub = df[[group_col, y]].dropna()
    if method == 'Kruskal':
        return sp.posthoc_dunn(sub, val_col=y, group_col=group_col,
                               p_adjust='bonferroni').round(3)
    else:
        res = pairwise_tukeyhsd(sub[y], sub[group_col])
        print(res)
        return res


def run_chi2(df, cat_vars, target='metabolic_syndrome'):
    """범주형 일괄 검정 — 기대빈도 점검 후 카이제곱/Fisher 자동 선택.

    기대빈도<5 & 2x2 → Fisher exact
    그 외            → 카이제곱
    효과크기는 Cramér's V.

    Returns
    -------
    pandas.DataFrame
        변수별 검정 방법·chi2·자유도·p값·Cramér's V·최소기대빈도·유의 표시.
    """
    rows = []
    for c in cat_vars:
        ct = pd.crosstab(df[c], df[target])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        min_exp = expected.min()

        if (expected < 5).any() and ct.shape == (2, 2):
            _, p = stats.fisher_exact(ct)
            method = 'Fisher'
        else:
            method = 'Chi2'

        n = ct.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rows.append([c, method, round(chi2, 2), dof,
                     round(p, 4), round(cramers_v, 3), round(min_exp, 1)])

    out = pd.DataFrame(rows, columns=['변수', '검정', 'chi2', 'dof',
                                      'p', 'CramersV', '최소기대빈도'])
    out['유의'] = np.where(out['p'] < 0.05, '*', '')
    return out.sort_values('p')
