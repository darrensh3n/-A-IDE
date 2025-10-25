"use client"

import { useState, useCallback } from "react"
import { VideoMonitor } from "@/components/video-monitor"
import { DetectionStats } from "@/components/detection-stats"
import { AlertPanel, type Alert } from "@/components/alert-panel"

interface Detection {
  class: string
  confidence: number
  bbox: [number, number, number, number]
}

interface DetectionResult {
  detections: Detection[]
  image: string
  timestamp: string
}

export default function Home() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [totalDetections, setTotalDetections] = useState(0)
  const [criticalAlerts, setCriticalAlerts] = useState(0)

  const handleDetectionResult = useCallback((result: DetectionResult) => {
    // Filter detections to only count those with 80% confidence and above
    const highConfidenceDetections = result.detections.filter(detection => detection.confidence >= 0.8)
    setTotalDetections((prev) => prev + highConfidenceDetections.length)

    // Create alerts for each high-confidence detection
    highConfidenceDetections.forEach((detection) => {
      const isDrowning =
        detection.class.toLowerCase().includes("drowning") || detection.class.toLowerCase().includes("distress")
      const isPerson = detection.class.toLowerCase().includes("person")

      let alertType: "critical" | "warning" | "info" = "info"
      let message = `Detected: ${detection.class}`

      if (isDrowning) {
        alertType = "critical"
        message = `⚠️ CRITICAL: Drowning detected - ${detection.class}`
        setCriticalAlerts((prev) => prev + 1)
      } else if (isPerson && detection.confidence > 0.7) {
        alertType = "warning"
        message = `Person detected in water - monitoring`
      }

      const newAlert: Alert = {
        id: `${Date.now()}-${Math.random()}`,
        type: alertType,
        message,
        time: new Date().toLocaleTimeString(),
        status: alertType === "critical" ? "active" : "resolved",
        detectionClass: detection.class,
        confidence: detection.confidence,
      }

      setAlerts((prev) => [newAlert, ...prev].slice(0, 20)) // Keep last 20 alerts
    })
  }, [])

  const handleClearAlert = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((alert) => alert.id !== id))
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <svg className="h-6 w-6 text-primary-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground">Drowning Detection System</h1>
                <p className="text-sm text-muted-foreground">AI-Powered Water Safety Monitoring</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2">
                <div className="h-2 w-2 animate-pulse rounded-full bg-chart-3" />
                <span className="text-sm font-medium text-foreground">System Active</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <VideoMonitor onDetectionResult={handleDetectionResult} />
          </div>
          <div className="space-y-6">
            <AlertPanel alerts={alerts} onClearAlert={handleClearAlert} />
            <DetectionStats
              totalDetections={totalDetections}
              criticalAlerts={criticalAlerts}
              activeMonitors={1}
              avgResponseTime="1.2s"
            />
          </div>
        </div>
      </main>
    </div>
  )
}
