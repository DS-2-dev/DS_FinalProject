import { num, pval, signed, type RobustnessRow } from '@/data'

/** Point estimate + 95% CI per row, on a shared symmetric axis centred on zero. */
export function ForestPlot({ rows }: { rows: RobustnessRow[] }) {
  const lim = Math.ceil(Math.max(...rows.flatMap((r) => [-r.ci_low_pts, r.ci_high_pts])))
  const pos = (v: number) => `${((v + lim) / (2 * lim)) * 100}%`
  const ticks = Array.from({ length: 2 * lim + 1 }, (_, i) => i - lim)

  return (
    <div className="text-sm">
      <div className="grid grid-cols-[minmax(0,1fr)] gap-y-1 sm:grid-cols-[13rem_minmax(0,1fr)_8.5rem] sm:gap-x-4">
        {rows.map((r, i) => (
          <div key={r.spec} className="contents">
            <div className="pt-2 font-medium sm:py-2.5 sm:font-normal">{r.spec}</div>
            <div className="relative h-9" title={`${signed(r.effect_per_10h_pts)} points, 95% CI ${signed(r.ci_low_pts)} to ${signed(r.ci_high_pts)}, n = ${r.n}, ${pval(r.pval)}`}>
              {ticks.map((t) => (
                <div key={t} className={t === 0 ? 'absolute inset-y-0 w-px bg-foreground' : 'absolute inset-y-0 w-px bg-border'} style={{ left: pos(t) }} />
              ))}
              <div
                className={i === 0 ? 'absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-foreground' : 'absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-muted-foreground'}
                style={{ left: pos(r.ci_low_pts), right: `calc(100% - ${pos(r.ci_high_pts)})` }}
              />
              <div
                className={i === 0 ? 'absolute top-1/2 size-3 -translate-1/2 rounded-full bg-foreground ring-2 ring-background' : 'absolute top-1/2 size-3 -translate-1/2 rounded-full bg-muted-foreground ring-2 ring-background'}
                style={{ left: pos(r.effect_per_10h_pts) }}
              />
            </div>
            <div className="pb-2 text-muted-foreground tabular-nums sm:py-2.5 sm:text-right">
              <span className="text-foreground">{signed(r.effect_per_10h_pts)}</span>{' '}
              ({signed(r.ci_low_pts)}, {signed(r.ci_high_pts)})
            </div>
          </div>
        ))}
        <div className="hidden sm:block" />
        <div className="relative mt-1 h-5 text-xs text-muted-foreground tabular-nums">
          {ticks.map((t) => (
            <span key={t} className="absolute -translate-x-1/2" style={{ left: pos(t) }}>{num(t, 0)}</span>
          ))}
        </div>
      </div>
      <p className="mt-2 text-xs text-muted-foreground sm:pl-[14rem]">Points per +10 hours online per week</p>
    </div>
  )
}
