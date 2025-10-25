export interface Detection {
  class: string
  confidence: number
  bbox: [number, number, number, number]
}

export interface DetectionResult {
  detections: Detection[]
  image: string
  timestamp: string
}

export interface Alert {
  id: number
  type: "critical" | "warning" | "info"
  message: string
  time: string
  status: "active" | "resolved"
}
