import { useMemo } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  LabelList,
  Line,
  ReferenceLine,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'

import { ChartContainer, ChartTooltip, type ChartConfig } from '@/components/ui/chart'
import { COLOR, LABEL, num, pct, type Gender, type GenderStats, type Participant } from '@/data'

const config = {
  Male: { label: 'Men', color: 'var(--men)' },
  Female: { label: 'Women', color: 'var(--women)' },
} satisfies ChartConfig

const axisTick = { fontSize: 12 }

interface TipRow {
  gender: Gender
  score: number
  score_pct: number
  time_spent_hours?: number
  age?: number
}

function PointTooltip({ active, payload }: { active?: boolean; payload?: { payload: TipRow }[] }) {
  const row = active ? payload?.find((p) => p.payload?.gender)?.payload : undefined
  if (!row) return null
  return (
    <div className="grid min-w-36 gap-1 rounded-lg border bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <div className="flex items-center gap-2 text-muted-foreground">
        <span className="size-2 rounded-full" style={{ background: COLOR[row.gender] }} />
        {LABEL[row.gender]}
      </div>
      <div className="font-medium text-foreground tabular-nums">{row.score}/20 correct ({pct(row.score_pct)})</div>
      {row.time_spent_hours !== undefined && (
        <div className="text-muted-foreground tabular-nums">
          {num(row.time_spent_hours, 1)} h/week · age {row.age}
        </div>
      )}
    </div>
  )
}

/** One row of stacked dots per score value, with the group mean and its 95% CI. */
export function ScoreDots({ gender, participants, stats }: { gender: Gender; participants: Participant[]; stats: GenderStats }) {
  const points = useMemo(() => {
    const seen: Record<number, number> = {}
    return participants
      .filter((p) => p.gender === gender)
      .map((p) => ({ ...p, x: Math.round(p.score_pct * 100) / 100 }))
      .sort((a, b) => a.x - b.x)
      .map((p) => ({ ...p, y: (seen[p.x] = (seen[p.x] ?? 0) + 1) }))
  }, [gender, participants])
  const half = (1.96 * stats.score_sd) / Math.sqrt(stats.n)

  return (
    <ChartContainer config={config} className="aspect-auto h-44 w-full">
      <ScatterChart margin={{ top: 8, right: 12, bottom: 0, left: 12 }}>
        <XAxis type="number" dataKey="x" domain={[0.25, 0.9]} ticks={[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
          tickFormatter={(v) => pct(v)} tick={axisTick} tickLine={false} axisLine={false} />
        <YAxis type="number" dataKey="y" domain={[-1, 12]} hide />
        <ReferenceLine x={0.5} stroke="var(--border)" strokeWidth={1}
          label={{ value: 'chance', position: 'insideTopRight', fontSize: 11, fill: 'var(--muted-foreground)' }} />
        <ReferenceLine segment={[{ x: stats.score_mean, y: 0.4 }, { x: stats.score_mean, y: 12 }]} stroke="var(--foreground)" strokeWidth={1.5} />
        <ReferenceLine segment={[{ x: stats.score_mean - half, y: -0.3 }, { x: stats.score_mean + half, y: -0.3 }]}
          stroke="var(--foreground)" strokeWidth={4} strokeLinecap="round" />
        <ChartTooltip cursor={false} isAnimationActive={false} content={<PointTooltip />} />
        <Scatter data={points} fill={COLOR[gender]} stroke="var(--background)" strokeWidth={2} shape="circle" isAnimationActive={false} />
        <ZAxis range={[64, 64]} />
      </ScatterChart>
    </ChartContainer>
  )
}

const XMAX = 75

function linearFit(rows: Participant[]) {
  const n = rows.length
  const xs = rows.map((r) => r.time_spent_hours)
  const ys = rows.map((r) => r.score_pct)
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  xs.forEach((x, i) => {
    sxy += (x - mx) * (ys[i] - my)
    sxx += (x - mx) ** 2
  })
  const b = sxy / sxx
  const a = my - b * mx
  const s2 = ys.reduce((acc, y, i) => acc + (y - a - b * xs[i]) ** 2, 0) / (n - 2)
  return Array.from({ length: 31 }, (_, i) => {
    const x = (i / 30) * XMAX
    const yhat = a + b * x
    const half = 1.98 * Math.sqrt(s2 * (1 / n + (x - mx) ** 2 / sxx))
    return { x, yhat, band: [yhat - half, yhat + half] as [number, number] }
  })
}

/** Time online vs score for one gender: dots sized by age, linear fit and 95% band. */
export function TimeScatter({ gender, participants }: { gender: Gender; participants: Participant[] }) {
  const rows = useMemo(() => participants.filter((p) => p.gender === gender), [gender, participants])
  const fit = useMemo(() => linearFit(rows), [rows])
  const inside = rows.filter((r) => r.time_spent_hours <= XMAX).map((r) => ({ ...r, x: r.time_spent_hours, y: r.score_pct }))
  const outside = rows.filter((r) => r.time_spent_hours > XMAX).map((r) => ({ ...r, x: XMAX, y: r.score_pct }))

  return (
    <ChartContainer config={config} className="aspect-auto h-72 w-full">
      <ComposedChart data={fit} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis type="number" dataKey="x" domain={[0, XMAX]} ticks={[0, 25, 50, 75]} tick={axisTick}
          tickLine={false} axisLine={false} allowDataOverflow padding={{ left: 10, right: 10 }}
          label={{ value: 'Hours online per week', position: 'insideBottom', offset: -2, fontSize: 12, fill: 'var(--muted-foreground)' }}
          height={40} />
        <YAxis type="number" dataKey="y" domain={[0.25, 0.9]} ticks={[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
          tickFormatter={(v) => pct(v)} tick={axisTick} tickLine={false} axisLine={false} width={40} />
        <ZAxis type="number" dataKey="age" domain={[6, 74]} range={[40, 340]} />
        <Area dataKey="band" stroke="none" fill={COLOR[gender]} fillOpacity={0.12} isAnimationActive={false} tooltipType="none" activeDot={false} />
        <Line dataKey="yhat" stroke={COLOR[gender]} strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} tooltipType="none" />
        <Scatter data={inside} fill={COLOR[gender]} fillOpacity={0.85} stroke="var(--background)" strokeWidth={1.5} isAnimationActive={false} />
        {outside.length > 0 && (
          <Scatter data={outside} fill={COLOR[gender]} shape="triangle" isAnimationActive={false}>
            <LabelList dataKey="time_spent_hours" position="left" formatter={(v) => `${v} h`}
              style={{ fontSize: 11, fill: 'var(--muted-foreground)' }} />
          </Scatter>
        )}
        <ChartTooltip cursor={false} isAnimationActive={false} shared={false} content={<PointTooltip />} />
      </ComposedChart>
    </ChartContainer>
  )
}
