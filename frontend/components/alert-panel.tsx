"use client"

import { Card } from "@/components/ui/card"
import { AlertTriangle, CheckCircle, Clock, Bell } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export interface Alert {
  id: string
  type: "critical" | "warning" | "info"
  message: string
  time: string
  status: "active" | "resolved"
  detectionClass?: string
  confidence?: number
}

interface AlertPanelProps {
  alerts: Alert[]
  onClearAlert?: (id: string) => void
  onVoiceAlert?: (alert: Alert) => void
}

export function AlertPanel({ alerts, onClearAlert, onVoiceAlert }: AlertPanelProps) {
  const sortedAlerts = [...alerts].sort((a, b) => {
    if (a.status === "active" && b.status !== "active") return -1
    if (a.status !== "active" && b.status === "active") return 1
    return 0
  })

  return (
    <Card className="overflow-hidden shadow-2xl border-0 bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl">
      <div className="border-b border-slate-200 dark:border-slate-700 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-800 dark:to-slate-900 px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 p-2 shadow-lg">
              <Bell className="h-5 w-5 text-white" />
            </div>
            <h2 className="text-lg font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">Alert History</h2>
          </div>
          {alerts.filter((a) => a.status === "active").length > 0 && (
            <Badge variant="destructive" className="text-xs font-bold shadow-lg animate-pulse bg-gradient-to-r from-red-500 to-rose-600 border-0">
              {alerts.filter((a) => a.status === "active").length} Active
            </Badge>
          )}
        </div>
      </div>

      <div className="max-h-[400px] divide-y divide-slate-200 dark:divide-slate-700 overflow-y-auto bg-slate-50/50 dark:bg-slate-950/50">
        {sortedAlerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 py-12">
            <div className="rounded-full bg-green-100 dark:bg-green-900/30 p-4">
              <CheckCircle className="h-10 w-10 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-sm font-semibold text-slate-600 dark:text-slate-400">All Clear - No Alerts</p>
          </div>
        ) : (
          sortedAlerts.map((alert) => (
            <div key={alert.id} className="px-4 py-4 transition-all duration-200 hover:bg-white dark:hover:bg-slate-900 hover:shadow-sm">
              <div className="flex items-start gap-3">
                <div
                  className={`mt-0.5 rounded-xl p-2 shadow-sm ${
                    alert.type === "critical"
                      ? "bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-950/30 dark:to-rose-950/30 border border-red-200 dark:border-red-800"
                      : alert.type === "warning"
                        ? "bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-950/30 dark:to-amber-950/30 border border-orange-200 dark:border-orange-800"
                        : "bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-950/30 dark:to-cyan-950/30 border border-blue-200 dark:border-blue-800"
                  }`}
                >
                  {alert.type === "critical" ? (
                    <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400" />
                  ) : alert.type === "warning" ? (
                    <AlertTriangle className="h-5 w-5 text-orange-600 dark:text-orange-400" />
                  ) : (
                    <CheckCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  )}
                </div>
                <div className="flex-1 space-y-1">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1">
                      <p className="text-sm font-medium text-foreground">{alert.message}</p>
                      {alert.detectionClass && alert.confidence && (
                        <p className="mt-1 text-xs text-muted-foreground">
                          Class: {alert.detectionClass} • Confidence: {(alert.confidence * 100).toFixed(1)}%
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {alert.status === "active" && (
                        <Badge variant="destructive" className="text-xs">
                          Active
                        </Badge>
                      )}
                      {alert.status === "resolved" && onClearAlert && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs"
                          onClick={() => onClearAlert(alert.id)}
                        >
                          Clear
                        </Button>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {alert.time}
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  )
}
