// Shape of public/data.json, written by src/build_site.py in the analysis repo.

export type Gender = 'Male' | 'Female'

export interface Participant {
  gender: Gender
  age: number
  age_range: string
  time_spent_hours: number
  score: number
  score_pct: number
}

export interface Term {
  coef: number
  se: number
  pval: number
  ci_low: number
  ci_high: number
}

export interface GenderStats {
  gender: Gender
  n: number
  score_mean: number
  score_sd: number
  pct_above_chance: number
  hours_median: number
  age_median: number
  pearson_r: number
  pearson_p: number
  effect_per_10h_pts: number
  ci_low_pts: number
  ci_high_pts: number
  time_pval: number
  age_coef_pts_per_year: number
  age_pval: number
  age_spearman_p: number
  n_over_40: number
  age_pval_under40: number
}

export interface RobustnessRow {
  spec: string
  n: number
  effect_per_10h_pts: number
  ci_low_pts: number
  ci_high_pts: number
  pval: number
}

export interface SiteData {
  summary: {
    n: number
    gender_counts: Record<Gender, number>
    score_pct_mean: number
    time_hours_median: number
    pearson: { r: number; pval: number }
    ols: { r2: number; time_spent_hours: Term }
    gender_gap: { male_minus_female_pts: number; welch_p: number; mannwhitney_p: number; cohens_d: number }
    by_gender: GenderStats[]
    robustness: RobustnessRow[]
  }
  participants: Participant[]
  overall_model: { roc_auc: number }
}

export const GENDERS: Gender[] = ['Male', 'Female']
export const LABEL: Record<Gender, string> = { Male: 'Men', Female: 'Women' }
export const COLOR: Record<Gender, string> = { Male: 'var(--men)', Female: 'var(--women)' }

const MINUS = '−'

/** Fixed decimals with a real minus sign and no negative zero. */
export function num(x: number, digits = 2): string {
  const t = Math.abs(x).toFixed(digits)
  return x < 0 && Number(t) !== 0 ? MINUS + t : t
}

/** Like num() but always signed (+/−), except for zero. */
export function signed(x: number, digits = 1): string {
  const t = num(x, digits)
  return t.startsWith(MINUS) || Number(t) === 0 ? t : '+' + t
}

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`

export function pval(p: number): string {
  if (p < 0.001) return 'p < 0.001'
  return `p = ${p.toFixed(p < 0.01 ? 3 : 2)}`
}
