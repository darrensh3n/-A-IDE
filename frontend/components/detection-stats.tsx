"use client"

import { Card } from "@/components/ui/card"
import { Activity, Eye, AlertCircle, TrendingUp } from "lucide-react"

interface DetectionStatsProps {
  totalDetections: number
  criticalAlerts: number
  activeMonitors: number
  avgResponseTime: string
}

export function DetectionStats({
  totalDetections,
  criticalAlerts,
  activeMonitors,
  avgResponseTime,
}: DetectionStatsProps) {
  const stats = [
    {
      label: "Total Detections",
      value: totalDetections.toString(),
      icon: Eye,
      color: "text-primary",
    },
    {
      label: "Critical Alerts",
      value: criticalAlerts.toString(),
      icon: AlertCircle,
      color: "text-destructive",
    },
    {
      label: "Active Monitors",
      value: activeMonitors.toString(),
      icon: Activity,
      color: "text-chart-3",
    },
    {
      label: "Avg Response Time",
      value: avgResponseTime,
      icon: TrendingUp,
      color: "text-chart-4",
    },
  ]

  return (
    <Card className="overflow-hidden shadow-2xl border-0 bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl">
      <div className="border-b border-slate-200 dark:border-slate-700 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-slate-800 dark:to-slate-900 px-4 py-4">
        <h2 className="text-lg font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">System Statistics</h2>
      </div>

      <div className="divide-y divide-slate-200 dark:divide-slate-700 bg-slate-50/50 dark:bg-slate-950/50">
        {stats.map((stat, index) => (
          <div key={index} className="px-4 py-5 transition-all duration-200 hover:bg-white dark:hover:bg-slate-900">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={`rounded-xl bg-gradient-to-br p-3 shadow-sm ${
                  index === 0 ? "from-blue-500/10 to-cyan-500/10 border border-blue-200 dark:border-blue-800" :
                  index === 1 ? "from-red-500/10 to-rose-500/10 border border-red-200 dark:border-red-800" :
                  index === 2 ? "from-green-500/10 to-emerald-500/10 border border-green-200 dark:border-green-800" :
                  "from-purple-500/10 to-violet-500/10 border border-purple-200 dark:border-purple-800"
                }`}>
                  <stat.icon className={`h-5 w-5 ${stat.color}`} />
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{stat.label}</p>
                  <p className="text-2xl font-bold text-foreground mt-1">{stat.value}</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
