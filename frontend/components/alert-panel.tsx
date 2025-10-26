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
    <Card className="overflow-hidden">
      <div className="border-b border-border bg-muted/50 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell className="h-4 w-4 text-foreground" />
            <h2 className="font-semibold text-foreground">Alert History</h2>
          </div>
          {alerts.filter((a) => a.status === "active").length > 0 && (
            <Badge variant="destructive" className="text-xs">
              {alerts.filter((a) => a.status === "active").length} Active
            </Badge>
          )}
        </div>
      </div>

      <div className="max-h-[400px] divide-y divide-border overflow-y-auto">
        {sortedAlerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-8">
            <CheckCircle className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">No alerts</p>
          </div>
        ) : (
          sortedAlerts.map((alert) => (
            <div key={alert.id} className="px-4 py-3">
              <div className="flex items-start gap-3">
                <div
                  className={`mt-0.5 rounded-full p-1 ${
                    alert.type === "critical"
                      ? "bg-destructive/10"
                      : alert.type === "warning"
                        ? "bg-chart-4/10"
                        : "bg-primary/10"
                  }`}
                >
                  {alert.type === "critical" ? (
                    <AlertTriangle className="h-4 w-4 text-destructive" />
                  ) : alert.type === "warning" ? (
                    <AlertTriangle className="h-4 w-4 text-chart-4" />
                  ) : (
                    <CheckCircle className="h-4 w-4 text-primary" />
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
