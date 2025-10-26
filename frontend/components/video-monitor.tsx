"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Video, Camera, Loader2, Upload } from "lucide-react"

interface Detection {
  class: string
  confidence: number
  bbox: [number, number, number, number]
  frame?: number
}

interface DrowningAnalysis {
  drowning_risk: "none" | "low" | "medium" | "high"
  risk_score: number
  indicators: string[]
  people_detected: number
}

interface DetectionResult {
  detections: Detection[]
  image: string
  timestamp: string
  summary?: any
  total_frames?: number
  processed_frames?: number
  drowning_analysis?: DrowningAnalysis
}

interface VideoMonitorProps {
  onDetectionResult?: (result: DetectionResult) => void
}

export function VideoMonitor({ onDetectionResult }: VideoMonitorProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [videoSource, setVideoSource] = useState<"camera" | "upload" | null>(null)
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(false)
  const [isRealTimeRunning, setIsRealTimeRunning] = useState(false)
  const [fps, setFps] = useState(0)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const realTimeIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastProcessTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)
  const animationFrameRef = useRef<number | null>(null)





  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      })
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }

      setVideoSource("camera")
      setIsRealTimeEnabled(true) // Automatically enable real-time processing
      setError(null)
      
      // Start real-time processing immediately when camera starts
      setTimeout(() => {
        if (!isRealTimeRunning) {
          startRealTimeProcessing()
        }
      }, 1000) // Small delay to ensure video is ready
    } catch (error) {
      console.error("Error accessing camera:", error)
      setError("Failed to access camera. Please check permissions.")
    }
  }

  const stopCamera = () => {
    stopRealTimeProcessing()

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setVideoSource(null)
    setIsRealTimeEnabled(false)
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Check if it's a video file
    if (!file.type.startsWith('video/')) {
      setError('Please select a video file')
      return
    }

    setIsProcessing(true)
    setError(null)
    setVideoSource(null) // Clear camera source

    try {
      // First, display the uploaded video
      const videoUrl = URL.createObjectURL(file)
      if (videoRef.current) {
        videoRef.current.src = videoUrl
        videoRef.current.load()
        setVideoSource("upload") // Set a custom source type
      }

      // Then process the video for detection
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/api/detect-drowning', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Video processing failed')

      const result = await response.json()
      setDetectionResult(result)
      onDetectionResult?.(result)

    } catch (error) {
      console.error('Error processing video:', error)
      setError('Failed to process video. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return null

    const canvas = canvasRef.current
    const video = videoRef.current

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext("2d")
    if (!ctx) return null

    ctx.drawImage(video, 0, 0)

    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8)
    })
  }, [])

  const processFrame = useCallback(async () => {
    if (isProcessing) return

    const blob = await captureFrame()
    if (!blob) return

    setIsProcessing(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", blob, "frame.jpg")

    try {
      const startTime = performance.now()

      const response = await fetch("http://localhost:8000/api/detect-frame", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Detection failed")

      const result = await response.json()
      setDetectionResult(result)
      onDetectionResult?.(result)

      const endTime = performance.now()
      const processingTime = endTime - startTime
      lastProcessTimeRef.current = processingTime
      frameCountRef.current++

      if (frameCountRef.current % 5 === 0) {
        setFps(Math.round(1000 / processingTime))
      }
    } catch (error) {
      console.error("Error processing frame:", error)
      setError("Failed to process frame. Make sure your FastAPI backend is running on port 8000.")
    } finally {
      setIsProcessing(false)
    }
  }, [captureFrame, isProcessing, onDetectionResult])

  const startRealTimeProcessing = useCallback(() => {
    if (realTimeIntervalRef.current) return

    setIsRealTimeRunning(true)
    frameCountRef.current = 0

    // Process frames every 1 second for live detection (more responsive)
    realTimeIntervalRef.current = setInterval(() => {
      processFrame()
    }, 1000)
  }, [processFrame])

  const stopRealTimeProcessing = useCallback(() => {
    if (realTimeIntervalRef.current) {
      clearInterval(realTimeIntervalRef.current)
      realTimeIntervalRef.current = null
    }
    setIsRealTimeRunning(false)
    setFps(0)
    frameCountRef.current = 0
  }, [])

  const toggleRealTimeProcessing = useCallback(() => {
    if (isRealTimeRunning) {
      stopRealTimeProcessing()
    } else {
      startRealTimeProcessing()
    }
  }, [isRealTimeRunning, startRealTimeProcessing, stopRealTimeProcessing])

  useEffect(() => {
    if (isRealTimeEnabled && videoSource === "camera" && !isRealTimeRunning) {
      startRealTimeProcessing()
    } else if (!isRealTimeEnabled && isRealTimeRunning) {
      stopRealTimeProcessing()
    }
  }, [isRealTimeEnabled, videoSource, isRealTimeRunning, startRealTimeProcessing, stopRealTimeProcessing])

  // Auto-start camera on component mount
  useEffect(() => {
    startCamera()
  }, [])

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (realTimeIntervalRef.current) {
        clearInterval(realTimeIntervalRef.current)
      }
    }
  }, [])



  // For camera mode, use the original detection result drawing
  useEffect(() => {
    if (videoSource === "camera" && detectionResult && canvasRef.current && videoRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      const ctx = canvas.getContext("2d")
      if (!ctx) return

      canvas.width = video.videoWidth || video.width
      canvas.height = video.videoHeight || video.height

      ctx.drawImage(video, 0, 0)

      // Draw bounding boxes for all detections (live mode shows everything)
      detectionResult.detections
        .filter(detection => detection.confidence >= 0.3) // Lower threshold for live detection
        .forEach((detection) => {
          const [x1, y1, x2, y2] = detection.bbox
          const isDrowning = detection.class.toLowerCase().includes("drowning") ||
                            detection.class.toLowerCase().includes("distress")
          const isPerson = detection.class.toLowerCase().includes("person")
          
          // Color coding based on detection type and confidence
          let strokeColor = "#3b82f6" // Default blue
          let fillColor = "#3b82f6"
          
          if (isDrowning) {
            strokeColor = "#ef4444" // Red for drowning
            fillColor = "#ef4444"
          } else if (isPerson) {
            strokeColor = "#10b981" // Green for person
            fillColor = "#10b981"
          }
          
          // Make high confidence detections thicker
          const lineWidth = detection.confidence >= 0.7 ? 4 : 2
          
          ctx.strokeStyle = strokeColor
          ctx.lineWidth = lineWidth
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

          // Draw label background
          ctx.fillStyle = fillColor
          ctx.fillRect(x1, y1 - 25, 200, 25)

          // Draw label text
          ctx.fillStyle = "#ffffff"
          ctx.font = "14px sans-serif"
          ctx.fillText(`${detection.class} ${(detection.confidence * 100).toFixed(1)}%`, x1 + 5, y1 - 7)
        })
    }
  }, [detectionResult, videoSource])

  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border bg-muted/50 px-4 py-3">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-foreground">Live Monitor</h2>
          <div className="flex items-center gap-2">
            {videoSource === "camera" ? (
              <Button variant="outline" size="sm" onClick={stopCamera} disabled={isProcessing}>
                <Video className="mr-2 h-4 w-4" />
                Stop Camera
              </Button>
            ) : (
              <Button variant="outline" size="sm" onClick={startCamera} disabled={isProcessing}>
                <Camera className="mr-2 h-4 w-4" />
                Start Camera
              </Button>
            )}
            <div className="relative">
              <input
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={isProcessing}
              />
              <Button
                variant="outline"
                size="sm"
                disabled={isProcessing}
              >
                <Upload className="mr-2 h-4 w-4" />
                Upload Video
              </Button>
            </div>
          </div>
        </div>
      </div>

      {videoSource === "camera" && (
        <div className="border-b border-border bg-muted/30 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-sm font-medium text-foreground">
                <input
                  type="checkbox"
                  checked={isRealTimeEnabled}
                  onChange={(e) => setIsRealTimeEnabled(e.target.checked)}
                  disabled={isProcessing}
                  className="h-4 w-4"
                />
                Live Detection (1s intervals)
              </label>
              {isRealTimeEnabled && (
                <Button variant="outline" size="sm" onClick={toggleRealTimeProcessing} disabled={isProcessing}>
                  {isRealTimeRunning ? "Pause" : "Resume"}
                </Button>
              )}
            </div>
            {isRealTimeRunning && (
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <div className="h-2 w-2 animate-pulse rounded-full bg-chart-3" />
                  <span>Live-detecting every 1s</span>
                </div>
                {fps > 0 && <span>~{fps} FPS</span>}
                <span>{frameCountRef.current} frames</span>
              </div>
            )}
          </div>
        </div>
        )}

      {videoSource === "upload" && (
        <div className="border-b border-border bg-muted/30 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Video className="h-4 w-4" />
                <span>Uploaded Video - Click play to view</span>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.src = ""
                  videoRef.current.load()
                }
                setVideoSource(null)
                setDetectionResult(null)
              }}
            >
              Clear Video
            </Button>
          </div>
        </div>
      )}

      <div className="relative aspect-video bg-muted">
        {!videoSource && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
            <div className="rounded-full bg-muted-foreground/10 p-6">
              <Video className="h-12 w-12 text-muted-foreground" />
            </div>
            <div className="text-center">
              <p className="font-medium text-foreground">No Video Source</p>
              <p className="text-sm text-muted-foreground">Start camera or upload a video to begin monitoring</p>
            </div>
          </div>
        )}

        <video 
          ref={videoRef} 
          className="h-full w-full object-contain" 
          controls={videoSource === "upload"}
          autoPlay={videoSource === "camera"}
          muted
        />

        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-contain pointer-events-none" />

        {isProcessing && !isRealTimeRunning && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm font-medium text-foreground">Processing with YOLOv8...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute bottom-4 left-4 right-4 rounded-lg border border-destructive bg-destructive/10 p-3">
            <p className="text-sm font-medium text-destructive">{error}</p>
          </div>
        )}
      </div>

      {detectionResult && (
        <div className="border-t border-border bg-muted/30 px-4 py-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Live Detections (≥30%):</span>
                <span className="font-semibold text-foreground">
                  {detectionResult.detections.filter(d => d.confidence >= 0.3).length}
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                {new Date(detectionResult.timestamp).toLocaleTimeString()}
              </div>
            </div>
            
            {/* Drowning Risk Alert */}
            {detectionResult.drowning_analysis && detectionResult.drowning_analysis.drowning_risk !== "none" && (
              <div className={`rounded-lg border-2 p-3 ${
                detectionResult.drowning_analysis.drowning_risk === "high" 
                  ? "border-red-500 bg-red-50 dark:bg-red-950/20"
                  : detectionResult.drowning_analysis.drowning_risk === "medium"
                  ? "border-orange-500 bg-orange-50 dark:bg-orange-950/20"
                  : "border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20"
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">⚠️</span>
                    <div>
                      <div className={`font-bold text-sm ${
                        detectionResult.drowning_analysis.drowning_risk === "high" 
                          ? "text-red-600 dark:text-red-400"
                          : detectionResult.drowning_analysis.drowning_risk === "medium"
                          ? "text-orange-600 dark:text-orange-400"
                          : "text-yellow-600 dark:text-yellow-400"
                      }`}>
                        {detectionResult.drowning_analysis.drowning_risk.toUpperCase()} DROWNING RISK DETECTED
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Risk Score: {detectionResult.drowning_analysis.risk_score} • {detectionResult.drowning_analysis.people_detected} people detected
                      </div>
                    </div>
                  </div>
                </div>
                {detectionResult.drowning_analysis.indicators.length > 0 && (
                  <div className="mt-2 text-xs space-y-1">
                    {detectionResult.drowning_analysis.indicators.map((indicator, idx) => (
                      <div key={idx} className="text-muted-foreground">• {indicator}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Current Frame: {detectionResult.detections.filter(d => d.confidence >= 0.3).length} detections</span>
              <span>High Confidence: {detectionResult.detections.filter(d => d.confidence >= 0.7).length}</span>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
