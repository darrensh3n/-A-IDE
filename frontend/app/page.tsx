"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Video, AlertTriangle, Activity, Shield, ArrowRight } from "lucide-react"

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-slate-950 dark:via-slate-900 dark:to-blue-950">
      {/* Header */}
      <header className="border-b border-border/40 bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500">
                <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                <h1 className="text-xl font-bold text-foreground">AIDE</h1>
                <p className="text-xs text-muted-foreground">AI-Powered Water Safety</p>
              </div>
            </div>
            <Link href="/dashboard">
              <Button size="lg" className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600">
                Go to Dashboard
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="flex flex-col items-center text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-blue-100 dark:bg-blue-950 px-4 py-2 text-sm font-medium text-blue-700 dark:text-blue-300">
            <Shield className="h-4 w-4" />
            AI-Powered Drowning Detection
          </div>
          
          <h1 className="mb-6 text-5xl font-bold tracking-tight text-foreground md:text-6xl lg:text-7xl">
            Drowning Detection
            <br />
            <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              Made Simple
            </span>
          </h1>
          
          <p className="mb-10 max-w-2xl text-lg text-muted-foreground md:text-xl">
            Real-time AI monitoring system powered by YOLOv8 to detect and alert potential drowning incidents.
            Keep swimmers safe with cutting-edge computer vision technology.
          </p>

          <div className="flex flex-col gap-4 sm:flex-row">
            <Link href="/dashboard">
              <Button size="lg" className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 sm:w-auto">
                <Video className="mr-2 h-5 w-5" />
                Start Monitoring
              </Button>
            </Link>
            <Button size="lg" variant="outline" className="w-full sm:w-auto">
              Learn More
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="mb-12 text-center">
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
            Powerful Features
          </h2>
          <p className="text-lg text-muted-foreground">
            Everything you need for comprehensive water safety monitoring
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <Card className="group overflow-hidden border-2 p-6 transition-all hover:border-blue-500 hover:shadow-xl">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 text-blue-600 transition-colors group-hover:bg-blue-500 group-hover:text-white dark:bg-blue-950 dark:text-blue-400">
              <Video className="h-6 w-6" />
            </div>
            <h3 className="mb-2 text-xl font-semibold text-foreground">Real-Time Detection</h3>
            <p className="text-muted-foreground">
              Live camera feed processing with instant detection and alerts for potential drowning scenarios.
            </p>
          </Card>

          <Card className="group overflow-hidden border-2 p-6 transition-all hover:border-cyan-500 hover:shadow-xl">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-cyan-100 text-cyan-600 transition-colors group-hover:bg-cyan-500 group-hover:text-white dark:bg-cyan-950 dark:text-cyan-400">
              <AlertTriangle className="h-6 w-6" />
            </div>
            <h3 className="mb-2 text-xl font-semibold text-foreground">Smart Alerts</h3>
            <p className="text-muted-foreground">
              Intelligent alert system with risk assessment and audio notifications for immediate response.
            </p>
          </Card>

          <Card className="group overflow-hidden border-2 p-6 transition-all hover:border-purple-500 hover:shadow-xl">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 text-purple-600 transition-colors group-hover:bg-purple-500 group-hover:text-white dark:bg-purple-950 dark:text-purple-400">
              <Activity className="h-6 w-6" />
            </div>
            <h3 className="mb-2 text-xl font-semibold text-foreground">Video Upload</h3>
            <p className="text-muted-foreground">
              Upload and analyze pre-recorded videos to detect drowning risks with detailed frame-by-frame analysis.
            </p>
          </Card>

          <Card className="group overflow-hidden border-2 p-6 transition-all hover:border-green-500 hover:shadow-xl">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 text-green-600 transition-colors group-hover:bg-green-500 group-hover:text-white dark:bg-green-950 dark:text-green-400">
              <Shield className="h-6 w-6" />
            </div>
            <h3 className="mb-2 text-xl font-semibold text-foreground">YOLOv8 Powered</h3>
            <p className="text-muted-foreground">
              State-of-the-art object detection with pose estimation for accurate drowning risk assessment.
            </p>
          </Card>

          <Card className="group overflow-hidden border-2 p-6 transition-all hover:border-orange-500 hover:shadow-xl">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-orange-100 text-orange-600 transition-colors group-hover:bg-orange-500 group-hover:text-white dark:bg-orange-950 dark:text-orange-400">
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="mb-2 text-xl font-semibold text-foreground">Statistics Dashboard</h3>
            <p className="text-muted-foreground">
              Track detection metrics, response times, and historical alerts with comprehensive analytics.
            </p>
          </Card>

          <Card className="group overflow-hidden border-2 p-6 transition-all hover:border-pink-500 hover:shadow-xl">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-pink-100 text-pink-600 transition-colors group-hover:bg-pink-500 group-hover:text-white dark:bg-pink-950 dark:text-pink-400">
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
            </div>
            <h3 className="mb-2 text-xl font-semibold text-foreground">Multi-Modal Alerts</h3>
            <p className="text-muted-foreground">
              Voice alerts, visual notifications, and alert history for comprehensive incident management.
            </p>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-20">
        <Card className="overflow-hidden border-2 bg-gradient-to-br from-blue-500 to-cyan-500 p-12 text-center">
          <h2 className="mb-4 text-3xl font-bold text-white md:text-4xl">
            Ready to Get Started?
          </h2>
          <p className="mb-8 text-lg text-blue-50">
            Launch the monitoring dashboard and start protecting swimmers with AI-powered detection.
          </p>
          <Link href="/dashboard">
            <Button size="lg" variant="secondary" className="bg-white text-blue-600 hover:bg-blue-50">
              <Video className="mr-2 h-5 w-5" />
              Launch Dashboard
            </Button>
          </Link>
        </Card>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/40 bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Shield className="h-4 w-4" />
              <span>AIDE - Drowning Detection System</span>
            </div>
            <div className="text-sm text-muted-foreground">
              Powered by YOLOv8 and Next.js
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
