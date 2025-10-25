"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Video, Camera, Loader2, Play, Pause, SkipForward, SkipBack } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"

interface Detection {
  class: string
  confidence: number
  bbox: [number, number, number, number]
  frame?: number
}

interface DetectionResult {
  detections: Detection[]
  image: string
  timestamp: string
  summary?: any
  total_frames?: number
  processed_frames?: number
}

interface VideoMonitorProps {
  onDetectionResult?: (result: DetectionResult) => void
}

export function VideoMonitor({ onDetectionResult }: VideoMonitorProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [videoSource, setVideoSource] = useState<"upload" | "camera" | null>(null)
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(false)
  const [isRealTimeRunning, setIsRealTimeRunning] = useState(false)
  const [fps, setFps] = useState(0)
  
  // Video playback states
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [showControls, setShowControls] = useState(false)
  const [allDetections, setAllDetections] = useState<Detection[]>([])
  const [currentFrameDetections, setCurrentFrameDetections] = useState<Detection[]>([])

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const realTimeIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastProcessTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)
  const animationFrameRef = useRef<number | null>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setVideoSource("upload")
    setIsProcessing(true)
    setError(null)
    setAllDetections([])
    setCurrentFrameDetections([])

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch("http://localhost:8000/api/detect-drowning", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Detection failed")

      const result = await response.json()
      setDetectionResult(result)
      
      // Store all detections for video playback
      if (result.detections) {
        setAllDetections(result.detections)
      }
      
      onDetectionResult?.(result)

      // Display the video
      if (videoRef.current) {
        videoRef.current.src = URL.createObjectURL(file)
        videoRef.current.load()
        
        // Set up video event listeners
        videoRef.current.addEventListener('loadedmetadata', () => {
          if (videoRef.current) {
            setDuration(videoRef.current.duration)
          }
        })
        
        videoRef.current.addEventListener('timeupdate', () => {
          if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime)
            updateCurrentFrameDetections(videoRef.current.currentTime)
          }
        })
        
        videoRef.current.addEventListener('play', () => setIsPlaying(true))
        videoRef.current.addEventListener('pause', () => setIsPlaying(false))
        videoRef.current.addEventListener('ended', () => setIsPlaying(false))
      }
    } catch (error) {
      console.error("Error processing video:", error)
      setError("Failed to process video. Make sure your FastAPI backend is running on port 8000.")
    } finally {
      setIsProcessing(false)
    }
  }

  // Helper function to update detections for current frame
  const updateCurrentFrameDetections = useCallback((currentTime: number) => {
    if (!allDetections.length || !videoRef.current) return
    
    // Calculate approximate frame number based on current time
    // Assuming 30 FPS for calculation (this could be made more accurate)
    const fps = 30
    const currentFrame = Math.floor(currentTime * fps)
    
    // Find detections for the current frame or nearby frames
    const frameDetections = allDetections.filter(detection => {
      if (detection.frame !== undefined) {
        return Math.abs(detection.frame - currentFrame) <= 2 // Allow 2 frame tolerance
      }
      return true // If no frame info, show all detections
    })
    
    setCurrentFrameDetections(frameDetections)
  }, [allDetections])

  // Video playback controls
  const togglePlayPause = useCallback(() => {
    if (!videoRef.current) return
    
    if (isPlaying) {
      videoRef.current.pause()
    } else {
      videoRef.current.play()
    }
  }, [isPlaying])

  const seekTo = useCallback((time: number) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = time
  }, [])

  const changePlaybackRate = useCallback((rate: number) => {
    if (!videoRef.current) return
    videoRef.current.playbackRate = rate
    setPlaybackRate(rate)
  }, [])

  const skipTime = useCallback((seconds: number) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds))
  }, [duration])

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

    // Process frames every 5 seconds (adjustable based on performance)
    realTimeIntervalRef.current = setInterval(() => {
      processFrame()
    }, 5000)
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

  // Animation loop for drawing bounding boxes
  const drawBoundingBoxes = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return

    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size to match video
    canvas.width = video.videoWidth || video.width
    canvas.height = video.videoHeight || video.height

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw bounding boxes for current frame (only 80% confidence and above)
    currentFrameDetections
      .filter(detection => detection.confidence >= 0.8)
      .forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox
        const isDrowning = detection.class.toLowerCase().includes("drowning") || 
                          detection.class.toLowerCase().includes("distress")

        // Draw bounding box
        ctx.strokeStyle = isDrowning ? "#ef4444" : "#3b82f6"
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        // Draw label background
        const labelText = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`
        ctx.fillStyle = isDrowning ? "#ef4444" : "#3b82f6"
        const labelWidth = ctx.measureText(labelText).width + 10
        ctx.fillRect(x1, y1 - 25, labelWidth, 25)

        // Draw label text
        ctx.fillStyle = "#ffffff"
        ctx.font = "14px sans-serif"
        ctx.fillText(labelText, x1 + 5, y1 - 7)
      })

    // Continue animation loop
    animationFrameRef.current = requestAnimationFrame(drawBoundingBoxes)
  }, [currentFrameDetections])

  useEffect(() => {
    if (videoSource === "upload" && currentFrameDetections.length > 0) {
      drawBoundingBoxes()
    } else if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [videoSource, currentFrameDetections, drawBoundingBoxes])

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

      // Draw bounding boxes (only 80% confidence and above)
      detectionResult.detections
        .filter(detection => detection.confidence >= 0.8)
        .forEach((detection) => {
          const [x1, y1, x2, y2] = detection.bbox
          const isDrowning = detection.class.toLowerCase().includes("drowning") ||
                            detection.class.toLowerCase().includes("distress")

          ctx.strokeStyle = isDrowning ? "#ef4444" : "#3b82f6"
          ctx.lineWidth = 3
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

          ctx.fillStyle = isDrowning ? "#ef4444" : "#3b82f6"
          ctx.fillRect(x1, y1 - 25, 200, 25)

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
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*,image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
            <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()} disabled={isProcessing}>
              <Upload className="mr-2 h-4 w-4" />
              Upload
            </Button>
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
          </div>
        </div>
      </div>

      {videoSource === "camera" && (
        <div className="border-b border-border bg-muted/30 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Switch
                  id="realtime-mode"
                  checked={isRealTimeEnabled}
                  onCheckedChange={setIsRealTimeEnabled}
                  disabled={isProcessing}
                />
                <Label htmlFor="realtime-mode" className="text-sm font-medium text-foreground">
                  Auto Detection (5s intervals)
                </Label>
              </div>
              {isRealTimeEnabled && (
                <Button variant="outline" size="sm" onClick={toggleRealTimeProcessing} disabled={isProcessing}>
                  {isRealTimeRunning ? (
                    <>
                      <Pause className="mr-2 h-4 w-4" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Resume
                    </>
                  )}
                </Button>
              )}
            </div>
            {isRealTimeRunning && (
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <div className="h-2 w-2 animate-pulse rounded-full bg-chart-3" />
                  <span>Auto-detecting every 5s</span>
                </div>
                {fps > 0 && <span>~{fps} FPS</span>}
                <span>{frameCountRef.current} frames</span>
              </div>
            )}
          </div>
        </div>
      )}

      {videoSource === "upload" && duration > 0 && (
        <div className="border-b border-border bg-muted/30 px-4 py-3">
          <div className="space-y-3">
            {/* Progress bar */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground w-12">
                {Math.floor(currentTime / 60)}:{(currentTime % 60).toFixed(0).padStart(2, '0')}
              </span>
              <Slider
                value={[currentTime]}
                onValueChange={([value]) => seekTo(value)}
                max={duration}
                step={0.1}
                className="flex-1"
              />
              <span className="text-xs text-muted-foreground w-12">
                {Math.floor(duration / 60)}:{(duration % 60).toFixed(0).padStart(2, '0')}
              </span>
            </div>
            
            {/* Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => skipTime(-10)}>
                  <SkipBack className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm" onClick={togglePlayPause}>
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Button variant="outline" size="sm" onClick={() => skipTime(10)}>
                  <SkipForward className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="flex items-center gap-2">
                <Label htmlFor="playback-rate" className="text-xs text-muted-foreground">
                  Speed:
                </Label>
                <select
                  id="playback-rate"
                  value={playbackRate}
                  onChange={(e) => changePlaybackRate(parseFloat(e.target.value))}
                  className="text-xs border rounded px-2 py-1"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={0.75}>0.75x</option>
                  <option value={1}>1x</option>
                  <option value={1.25}>1.25x</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </div>
            </div>
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
              <p className="text-sm text-muted-foreground">Upload a video or start camera to begin monitoring</p>
            </div>
          </div>
        )}

        <video 
          ref={videoRef} 
          className="h-full w-full object-contain" 
          controls={false}
          onMouseEnter={() => setShowControls(true)}
          onMouseLeave={() => setShowControls(false)}
        />

        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-contain pointer-events-none" />

        {/* Custom video controls overlay for uploaded videos */}
        {videoSource === "upload" && showControls && (
          <div className="absolute bottom-4 left-4 right-4 flex items-center justify-center">
            <div className="flex items-center gap-2 rounded-lg bg-black/80 px-4 py-2">
              <Button variant="ghost" size="sm" onClick={() => skipTime(-10)} className="text-white hover:bg-white/20">
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm" onClick={togglePlayPause} className="text-white hover:bg-white/20">
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => skipTime(10)} className="text-white hover:bg-white/20">
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

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
                <span className="text-sm text-muted-foreground">High Confidence Detections (≥80%):</span>
                <span className="font-semibold text-foreground">
                  {videoSource === "upload" 
                    ? currentFrameDetections.filter(d => d.confidence >= 0.8).length
                    : detectionResult.detections.filter(d => d.confidence >= 0.8).length
                  }
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                {new Date(detectionResult.timestamp).toLocaleTimeString()}
              </div>
            </div>
            
            {videoSource === "upload" && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <div className="flex items-center gap-4">
                  <span>Current Frame: {currentFrameDetections.filter(d => d.confidence >= 0.8).length} high-confidence</span>
                  {detectionResult.total_frames && (
                    <span>Total Frames: {detectionResult.total_frames}</span>
                  )}
                  {detectionResult.processed_frames && (
                    <span>Processed: {detectionResult.processed_frames}</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {allDetections.some(d => d.class.toLowerCase().includes("drowning")) && (
                    <span className="text-red-500 font-medium">⚠️ Drowning Detected</span>
                  )}
                </div>
              </div>
            )}
            
            {videoSource === "camera" && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Current Frame: {detectionResult.detections.filter(d => d.confidence >= 0.8).length} high-confidence</span>
                {detectionResult.detections.some(d => d.class.toLowerCase().includes("drowning") && d.confidence >= 0.8) && (
                  <span className="text-red-500 font-medium">⚠️ High-Confidence Drowning Detected</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}
