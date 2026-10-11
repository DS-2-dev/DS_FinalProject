import { useEffect, useState, type ReactNode } from 'react'

import { ScoreDots, TimeScatter } from '@/components/charts'
import { ForestPlot } from '@/components/forest-plot'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { COLOR, GENDERS, LABEL, num, pct, pval, signed, type GenderStats, type SiteData } from '@/data'

const REPO = 'https://github.com/DS-2-dev/DS_FinalProject'

function Section({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20">
      <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{eyebrow}</p>
      <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
      <div className="mt-6 space-y-6">{children}</div>
    </section>
  )
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <Card size="sm">
      <CardContent className="space-y-1">
        <p className="text-muted-foreground">{label}</p>
        <p className="text-3xl font-semibold tracking-tight">{value}</p>
        <p className="text-xs text-muted-foreground">{note}</p>
      </CardContent>
    </Card>
  )
}

function Swatch({ g }: { g: 'Male' | 'Female' }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="size-2.5 rounded-full" style={{ background: COLOR[g] }} />
      {LABEL[g]}
    </span>
  )
}

function Points({ items }: { items: ReactNode[] }) {
  return (
    <ul className="space-y-3 text-[15px] leading-relaxed text-muted-foreground">
      {items.map((item, i) => (
        <li key={i} className="flex gap-3">
          <span className="mt-2.5 h-px w-3 shrink-0 bg-foreground" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  )
}

export default function App() {
  const [data, setData] = useState<SiteData | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetch('./data.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError(true))
  }, [])

  if (error) return <p className="p-8 text-sm text-muted-foreground">Could not load data.json.</p>
  if (!data) return null

  const s = data.summary
  const gap = s.gender_gap
  const G = Object.fromEntries(s.by_gender.map((r) => [r.gender, r])) as Record<'Male' | 'Female', GenderStats>
  const main = s.robustness[0]

  const rows: [string, (g: GenderStats) => ReactNode][] = [
    ['Participants', (g) => g.n],
    ['Average score', (g) => <span className="font-semibold text-foreground">{pct(g.score_mean, 1)}</span>],
    ['Scored above chance', (g) => pct(g.pct_above_chance)],
    ['Median hours online / week', (g) => g.hours_median],
    ['Median age', (g) => g.age_median],
    ['Correlation, hours vs score', (g) => `${num(g.pearson_r)} (${pval(g.pearson_p)})`],
    ['Effect of +10 h/week (points)', (g) => `${signed(g.effect_per_10h_pts)} (${signed(g.ci_low_pts)} to ${signed(g.ci_high_pts)})`],
    ['Effect of age (points / year)', (g) => `${signed(g.age_coef_pts_per_year, 2)} (${pval(g.age_pval)})${g.gender === 'Male' ? '*' : ''}`],
  ]

  return (
    <div className="min-h-svh">
      <header className="sticky top-0 z-10 border-b bg-background/90 backdrop-blur">
        <nav className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4 text-sm sm:px-6">
          <span className="font-medium">AI image detection study</span>
          <div className="hidden gap-6 text-muted-foreground sm:flex">
            <a href="#overall" className="hover:text-foreground">Overall</a>
            <a href="#gender" className="hover:text-foreground">Men vs. women</a>
            <a href="#method" className="hover:text-foreground">Method</a>
            <a href={REPO} className="hover:text-foreground">GitHub</a>
          </div>
        </nav>
      </header>

      <main className="mx-auto max-w-5xl space-y-20 px-4 py-14 sm:px-6 sm:py-20">
        <div className="max-w-3xl">
          <Badge variant="outline" className="mb-5">Survey study · n = {s.n}</Badge>
          <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
            Does time online affect how well people spot AI images?
          </h1>
          <p className="mt-5 text-lg leading-relaxed text-muted-foreground text-pretty">
            {s.n} people ({s.gender_counts.Male} men, {s.gender_counts.Female} women) judged 20 images as AI-generated or
            human-made and reported how many hours a week they spend online. We tested whether internet exposure predicts
            their accuracy.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="Correlation, time vs score" value={num(s.pearson.r)} note={`${pval(s.pearson.pval)}: no relationship`} />
          <Stat label="Men vs. women" value={`${signed(gap.male_minus_female_pts)} pts`} note={`${pval(gap.welch_p)}: men scored higher`} />
          <Stat label="Average accuracy" value={pct(s.score_pct_mean)} note="Chance is 50%" />
          <Stat label="Median time online" value={`${s.time_hours_median} h`} note="Per week, self-reported" />
        </div>

        <Section id="overall" eyebrow="Overall" title="Time online makes no difference">
          <Points items={[
            <><span className="font-medium text-foreground">No relationship.</span> Pearson r = {num(s.pearson.r)} ({pval(s.pearson.pval)}). In the main regression, +10 hours a week changes the score by {signed(main.effect_per_10h_pts)} points (95% CI {signed(main.ci_low_pts)} to {signed(main.ci_high_pts)}).</>,
            <><span className="font-medium text-foreground">Only modestly better than chance.</span> Average accuracy was {pct(s.score_pct_mean)}; guessing gives 50%.</>,
          ]} />
          <Card>
            <CardHeader>
              <CardTitle>The result holds under every check</CardTitle>
              <CardDescription>Estimated change in score for +10 hours online per week, with 95% confidence intervals. Every interval crosses zero.</CardDescription>
            </CardHeader>
            <CardContent><ForestPlot rows={s.robustness} /></CardContent>
          </Card>
        </Section>

        <Section id="gender" eyebrow="Men vs. women" title={`Men scored ${num(gap.male_minus_female_pts, 1)} points higher`}>
          <Card className="py-0">
            <Table className="text-xs sm:text-sm">
              <TableHeader>
                <TableRow>
                  <TableHead className="pl-4" />
                  {GENDERS.map((g) => <TableHead key={g} className="text-right"><Swatch g={g} /></TableHead>)}
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map(([label, f]) => (
                  <TableRow key={label}>
                    <TableCell className="pl-4 whitespace-normal text-muted-foreground">{label}</TableCell>
                    {GENDERS.map((g) => <TableCell key={g} className="pr-4 text-right whitespace-normal tabular-nums">{f(G[g])}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Score distribution</CardTitle>
              <CardDescription>Each dot is one participant, stacked by score. The vertical line is the group average; the bar beneath it is its 95% confidence interval.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {GENDERS.map((g) => (
                <div key={g}>
                  <div className="flex items-baseline justify-between text-sm">
                    <Swatch g={g} />
                    <span className="text-muted-foreground tabular-nums">average <span className="font-medium text-foreground">{pct(G[g].score_mean, 1)}</span></span>
                  </div>
                  <ScoreDots gender={g} participants={data.participants} stats={G[g]} />
                </div>
              ))}
            </CardContent>
          </Card>

          <Points items={[
            <><span className="font-medium text-foreground">The one clear group difference.</span> Welch t-test {pval(gap.welch_p)}, Mann–Whitney {pval(gap.mannwhitney_p)}, Cohen’s d = {gap.cohens_d.toFixed(2)} (small to medium).</>,
            <><span className="font-medium text-foreground">Time online doesn’t matter within either gender.</span> Both correlations are about zero.</>,
            <><span className="font-medium text-foreground">*Age among men</span> looks significant, but it comes from {G.Male.n_over_40} men over 40. Refit on men 40 or younger, the effect disappears ({pval(G.Male.age_pval_under40)}), and the rank correlation is not significant ({pval(G.Male.age_spearman_p)}). Treat it as a hint, not a finding.</>,
          ]} />

          <div className="grid gap-3 md:grid-cols-2">
            {GENDERS.map((g) => (
              <Card key={g}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <Swatch g={g} />
                    <span className="text-sm font-normal text-muted-foreground tabular-nums">r = {num(G[g].pearson_r)}</span>
                  </CardTitle>
                  <CardDescription>Time online vs. score. Larger dots are older participants; the band is the 95% confidence interval of the fit.</CardDescription>
                </CardHeader>
                <CardContent><TimeScatter gender={g} participants={data.participants} /></CardContent>
              </Card>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">One participant reported 130 hours a week and is drawn at the right edge (▲). The fit and correlation include them.</p>
        </Section>

        <Section id="method" eyebrow="Context" title="Method and limits">
          <div className="grid gap-3 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">Classifiers <Badge variant="secondary">Exploratory</Badge></CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
                <p>A logistic model that flags low scorers (≤ 60%) from hours, age and gender reaches a cross-validated ROC AUC of <span className="text-foreground">{data.overall_model.roc_auc.toFixed(2)}</span>, barely above chance (0.50).</p>
                <p>Per age × gender models (groups with n ≥ 6) do no better.</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Caveats</CardTitle></CardHeader>
              <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
                <p>Self-reported hours and a convenience sample, mostly ages 16–25.</p>
                <p>Observational: associations, not causes.</p>
                <p>One hours answer was imputed with the median; no correction for multiple comparisons.</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Method</CardTitle></CardHeader>
              <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
                <p>Google Form: 20 images judged as AI or human, then age, gender and weekly hours online. Names were replaced by hashes.</p>
                <p>Main model: score ~ hours + gender + age, OLS with HC3 robust errors. Within each gender: score ~ hours + age.</p>
              </CardContent>
            </Card>
          </div>
        </Section>
      </main>

      <Separator />
      <footer className="mx-auto flex max-w-5xl flex-col gap-2 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:justify-between sm:px-6">
        <span>DS Final Project</span>
        <a href={REPO} className="hover:text-foreground">Code, data and notebooks on GitHub</a>
      </footer>
    </div>
  )
}
