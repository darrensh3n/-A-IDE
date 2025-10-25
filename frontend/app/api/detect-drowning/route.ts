import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file")

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Forward the request to your FastAPI backend
    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000/api/detect-drowning"

    const backendFormData = new FormData()
    backendFormData.append("file", file)

    const response = await fetch(fastApiUrl, {
      method: "POST",
      body: backendFormData,
    })

    if (!response.ok) {
      throw new Error("FastAPI backend error")
    }

    const result = await response.json()

    return NextResponse.json(result)
  } catch (error) {
    console.error("Error processing detection:", error)
    return NextResponse.json({ error: "Failed to process detection" }, { status: 500 })
  }
}
