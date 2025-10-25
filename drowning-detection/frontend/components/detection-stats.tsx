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
    <Card className="overflow-hidden">
      <div className="border-b border-border bg-muted/50 px-4 py-3">
        <h2 className="font-semibold text-foreground">System Statistics</h2>
      </div>

      <div className="divide-y divide-border">
        {stats.map((stat, index) => (
          <div key={index} className="px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-muted p-2">
                  <stat.icon className={`h-4 w-4 ${stat.color}`} />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">{stat.label}</p>
                  <p className="text-xl font-semibold text-foreground">{stat.value}</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
