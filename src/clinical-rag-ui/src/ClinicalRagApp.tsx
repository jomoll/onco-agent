import { Fragment, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Check, Loader2, MessageSquarePlus, Send, Square } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

const API_BASE = "/api"
const CONTEXT_CAPACITY = 65_000
const PATIENTS_ENDPOINT = `${API_BASE}/patients`
const MODELS_ENDPOINT = `${API_BASE}/models`
const CHAT_ENDPOINT = `${API_BASE}/chat`
const REPORTS_ENDPOINT = `${API_BASE}/reports`

interface PatientSummaryPayload {
  patient_id?: string
  fields?: Record<string, string>
  summary?: string
}

interface PatientSummary {
  id: string
  name: string
  description?: string
  details?: string
  dob?: string
  latest_report_date?: string
  report_counts?: Record<string, number>
  summary?: PatientSummaryPayload
}

interface AgentContextNode {
  report_type?: string
  report_date?: string
  section_name?: string
  snippet?: string
  citation_id?: string
  report_id?: string
  section_id?: string
  patient_id?: string
  test?: string
  value?: string
  unit?: string
  date?: string
  time?: string
  assessment_id?: string
}

interface AgentMetadata {
  final_answer?: string
  model?: string
  context_nodes?: AgentContextNode[]
  citations?: CitationMeta[]
  missing_information?: string[]
  analysis?: string
  required_information?: string[]
  actions?: Array<Record<string, unknown>>
  context_tokens_used?: number
}

interface CitationMeta {
  id: string
  label?: string
  type?: string
  date?: string
  snippet?: string
  aliases?: string[]
}

type MessageRole = "user" | "assistant" | "system"

interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  metadata?: AgentMetadata
}

interface ReportPreview {
  report_id: string
  patient_id: string
  report_type: string
  report_date?: string
  filename?: string
  content: string
}

type AgentEventPayload = Record<string, unknown>

interface AgentEvent {
  id: string
  type: string
  timestamp: number
  payload: AgentEventPayload
}

function createId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return Math.random().toString(36).slice(2)
}

function normaliseEvent(envelope: Record<string, unknown> | null | undefined): AgentEvent | null {
  if (!envelope || typeof envelope !== "object") {
    return null
  }
  const idRaw = envelope.id
  const id = typeof idRaw === "string" && idRaw ? idRaw : createId()
  const typeRaw = envelope.type
  const type = typeof typeRaw === "string" && typeRaw ? typeRaw : "unknown"
  const timestampRaw = envelope.timestamp
  let timestamp: number
  if (typeof timestampRaw === "number" && Number.isFinite(timestampRaw)) {
    timestamp = timestampRaw
  } else if (typeof timestampRaw === "string") {
    const parsed = Number(timestampRaw)
    timestamp = Number.isFinite(parsed) ? parsed : Date.now() / 1000
  } else {
    timestamp = Date.now() / 1000
  }
  const payloadRaw = envelope.payload
  const payload: AgentEventPayload =
    payloadRaw && typeof payloadRaw === "object" ? (payloadRaw as AgentEventPayload) : {}
  return { id, type, timestamp, payload }
}

export default function ClinicalRagApp() {
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [patientsLoading, setPatientsLoading] = useState(false)
  const [patientsError, setPatientsError] = useState<string | null>(null)
  const [patientSearchTerm, setPatientSearchTerm] = useState("")
  const [patientListOpen, setPatientListOpen] = useState(false)

  const [models, setModels] = useState<string[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [contextCapacity, setContextCapacity] = useState<number>(CONTEXT_CAPACITY)

  const [selectedPatientId, setSelectedPatientId] = useState<string>("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputText, setInputText] = useState("")
  const [isSending, setIsSending] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const [stepEvents, setStepEvents] = useState<AgentEvent[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [monitoringLoading, setMonitoringLoading] = useState(false)
  const [monitoringError, setMonitoringError] = useState<string | null>(null)
  const runIdRef = useRef<string | null>(null)
  const [selectedContextNode, setSelectedContextNode] = useState<AgentContextNode | null>(null)
  const [reportPreview, setReportPreview] = useState<ReportPreview | null>(null)
  const [reportDialogOpen, setReportDialogOpen] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const isPatientSelected = Boolean(selectedPatientId)
  const selectedPatient = useMemo(
    () => patients.find((patient) => patient.id === selectedPatientId) ?? null,
    [patients, selectedPatientId]
  )

  useEffect(() => {
    if (!selectedPatient) {
      setPatientSearchTerm("")
      return
    }
    setPatientListOpen(false)
    setPatientSearchTerm("")
  }, [selectedPatient])

  const filteredPatients = useMemo(() => {
    const term = patientSearchTerm.trim().toLowerCase()
    if (!term) {
      return patients
    }
    return patients
      .filter((patient) => {
        const haystack = `${patient.name ?? ""} ${patient.id}`.toLowerCase()
        return haystack.includes(term)
      })
      
  }, [patients, patientSearchTerm])

  const latestAssistantMessage = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "assistant") {
        return messages[index]
      }
    }
    return null
  }, [messages])

  const latestCitations = latestAssistantMessage?.metadata?.citations ?? []
  const latestContextNodes = latestAssistantMessage?.metadata?.context_nodes ?? []
  const latestCitationMap = useMemo(
    () => buildCitationIndex(latestContextNodes),
    [latestContextNodes]
  )
  const latestContextLookup = useMemo(() => {
    const map = new Map<string, AgentContextNode>()
    for (const node of latestContextNodes || []) {
      if (node?.citation_id) {
        map.set(node.citation_id, node)
      }
    }
    return map
  }, [latestContextNodes])

  useEffect(() => {
    setPatientsLoading(true)
    setPatientsError(null)
    fetch(PATIENTS_ENDPOINT)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text())
        }
        return response.json()
      })
      .then((data: PatientSummary[] | { patients: PatientSummary[] }) => {
        const list = Array.isArray(data) ? data : data.patients ?? []
        setPatients(list)
      })
      .catch((error: Error) => {
        console.error("Failed to load patients", error)
        setPatientsError("Failed to load patient list.")
      })
      .finally(() => setPatientsLoading(false))
  }, [])

  useEffect(() => {
    setMessages([])
    setChatError(null)
    setStepEvents([])
    setRunId(null)
    runIdRef.current = null
    setMonitoringError(null)
    setMonitoringLoading(false)
  }, [selectedPatientId])

  useEffect(() => {
    setSelectedContextNode(null)
    setReportPreview(null)
    setReportDialogOpen(false)
    setReportError(null)
  }, [latestAssistantMessage?.id])

  useEffect(() => {
    setModelsLoading(true)
    setModelsError(null)
    fetch(MODELS_ENDPOINT)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text())
        }
        return response.json()
      })
      .then((data: { models?: string[]; default?: string; context_tokens?: number }) => {
        const list = Array.isArray(data?.models) ? data.models.filter(Boolean) : []
        setModels(list)
        const initial =
          (data?.default && list.includes(data.default) ? data.default : list[0]) ?? ""
        setSelectedModel(initial)
        if (typeof data?.context_tokens === "number" && data.context_tokens > 0) {
          setContextCapacity(data.context_tokens)
        } else {
          setContextCapacity(CONTEXT_CAPACITY)
        }
      })
      .catch((error: Error) => {
        console.error("Failed to load models", error)
        setModelsError("Failed to load models.")
      })
      .finally(() => setModelsLoading(false))
  }, [])

  const handleSend = useCallback(async () => {
    const trimmed = inputText.trim()
    if (!trimmed || isSending || !isPatientSelected) {
      if (!isPatientSelected) {
        setChatError("Select a patient before asking a question.")
      }
      return
    }

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputText("")
    setIsSending(true)
    setChatError(null)
    setStepEvents([])
    setMonitoringError(null)
    setRunId(null)
    runIdRef.current = null

    const streamUrl = `${CHAT_ENDPOINT}/stream`
    const previousMessages = messages.slice(-10)

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const response = await fetch(streamUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: selectedPatientId || null,
          question: trimmed,
          model: selectedModel || null,
          history: previousMessages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(await response.text())
      }

      if (!response.body) {
        throw new Error("Streaming response did not include a body.")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let streamFailed = false

      while (true) {
        const { value, done } = await reader.read()
        if (value) {
          buffer += decoder.decode(value, { stream: !done })
        } else if (done) {
          buffer += decoder.decode(new Uint8Array(), { stream: false })
        }

        let newlineIndex = buffer.indexOf("\n")
        while (newlineIndex !== -1) {
          const line = buffer.slice(0, newlineIndex).trim()
          buffer = buffer.slice(newlineIndex + 1)

          if (line) {
            let parsed: Record<string, unknown> | null = null
            try {
              parsed = JSON.parse(line)
            } catch (parseError) {
              console.warn("Failed to parse stream line", parseError, line)
            }

            if (parsed) {
              if (typeof parsed.run_id === "string") {
                runIdRef.current = parsed.run_id
                setRunId(parsed.run_id)
              }

              const eventRecord = normaliseEvent(parsed)
              if (eventRecord) {
                setStepEvents((prev) => [...prev, eventRecord])

                if (eventRecord.type === "run_completed") {
                  const payload = eventRecord.payload
                  const metadata =
                    payload["metadata"] && typeof payload["metadata"] === "object"
                      ? (payload["metadata"] as AgentMetadata)
                      : ({} as AgentMetadata)
                  const answerFromPayload =
                    typeof payload["answer"] === "string" ? (payload["answer"] as string) : ""
                  const answer = answerFromPayload || metadata.final_answer || ""
                  const assistantMessage: ChatMessage = {
                    id: eventRecord.id,
                    role: "assistant",
                    content: answer,
                    createdAt: new Date().toISOString(),
                    metadata,
                  }
                  setMessages((prev) => [...prev, assistantMessage])
                }

                if (eventRecord.type === "run_failed") {
                  const failureMessage =
                    typeof eventRecord.payload["message"] === "string"
                      ? (eventRecord.payload["message"] as string)
                      : "An error occurred while running the agent. Please check the server logs."
                  setChatError(failureMessage)
                  setMessages((prev) => [
                    ...prev,
                    {
                      id: createId(),
                      role: "assistant",
                      content:
                        "Sorry, I couldn’t generate a response. Please try again, take a look in the monitoring log below, or contact Johannes.",
                      createdAt: new Date().toISOString(),
                    },
                  ])
                  streamFailed = true
                }
              }
            }
          }

          newlineIndex = buffer.indexOf("\n")
        }

        if (done || streamFailed) {
          if (streamFailed) {
            try {
              await reader.cancel()
            } catch {
              // ignore cancellation errors
            }
          }
          break
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setChatError("Request cancelled.")
        return
      }
      console.error("Chat stream failed", error)
      setChatError(
        "An error occurred while fetching the answer. Please retry or check the server logs."
      )
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "assistant",
          content:
            "Sorry, I couldn’t generate a response. Please try again or contact support.",
          createdAt: new Date().toISOString(),
        },
      ])
    } finally {
      abortControllerRef.current = null
      setIsSending(false)
      if (runIdRef.current) {
        setRunId(runIdRef.current)
      }
    }
  }, [inputText, isSending, isPatientSelected, selectedPatientId, selectedModel, messages])

  const handleAbortRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }, [])

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        handleSend().catch((error) => console.error("Send failed", error))
      }
    },
    [handleSend]
  )

  const handleClearChat = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setMessages([])
    setChatError(null)
    setStepEvents([])
    setRunId(null)
    runIdRef.current = null
    setMonitoringError(null)
    setMonitoringLoading(false)
    setSelectedContextNode(null)
    setReportPreview(null)
    setReportDialogOpen(false)
    setReportError(null)
  }, [])

  const refreshMonitoring = useCallback(async () => {
    const targetRunId = runIdRef.current ?? runId
    if (!targetRunId) {
      return
    }

    setMonitoringLoading(true)
    setMonitoringError(null)
    try {
      const response = await fetch(`${API_BASE}/monitor/runs/${targetRunId}`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data = (await response.json()) as { events?: Array<Record<string, unknown>> }
      const eventsArray = Array.isArray(data.events) ? data.events : []
      const normalisedEvents = eventsArray
        .map((event) => normaliseEvent(event))
        .filter((event): event is AgentEvent => Boolean(event))
      setStepEvents(normalisedEvents)
      setRunId(targetRunId)
      runIdRef.current = targetRunId
    } catch (error) {
      console.error("Monitoring fetch failed", error)
      setMonitoringError(
        error instanceof Error
          ? error.message
          : "Unable to refresh monitoring."
      )
    } finally {
      setMonitoringLoading(false)
    }
  }, [runId])

  const handleOpenReport = useCallback(async (node: AgentContextNode) => {
    if (!node.report_id) {
      return
    }
    setSelectedContextNode(node)
    setReportDialogOpen(true)
    setReportError(null)
    setReportPreview(null)
    setReportLoading(true)
    try {
      const response = await fetch(`${REPORTS_ENDPOINT}/${encodeURIComponent(node.report_id)}`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data = (await response.json()) as ReportPreview
      setReportPreview({
        report_id: data.report_id,
        patient_id: data.patient_id,
        report_type: data.report_type,
        report_date: data.report_date,
        filename: data.filename,
        content: data.content ?? "",
      })
    } catch (error) {
      console.error("Failed to load report", error)
      setReportError(error instanceof Error ? error.message : "Failed to load report.")
    } finally {
      setReportLoading(false)
    }
  }, [])

  const closeReportDialog = useCallback(() => {
    setReportDialogOpen(false)
    setSelectedContextNode(null)
    setReportPreview(null)
    setReportError(null)
  }, [])

  const highlightedReportContent = useMemo<ReactNode[]>(() => {
    if (!reportPreview) {
      return []
    }
    return highlightReportContent(reportPreview.content, selectedContextNode?.snippet)
  }, [reportPreview, selectedContextNode])

  const reportCountEntries = useMemo(() => {
    if (!selectedPatient?.report_counts) {
      return []
    }
    return Object.entries(selectedPatient.report_counts)
      .map(([type, count]) => ({
        type: type.replace(/_/g, " "),
        count,
      }))
      .sort((a, b) => {
        const diff = (b.count ?? 0) - (a.count ?? 0)
        if (diff !== 0) {
          return diff
        }
        return a.type.localeCompare(b.type)
      })
  }, [selectedPatient])
  const summaryFieldEntries = useMemo(() => {
    const fields = selectedPatient?.summary?.fields
    if (!fields) {
      return []
    }
    return Object.entries(fields)
  }, [selectedPatient])
  const hasSummaryFields = summaryFieldEntries.length > 0

  const contextUsage = useMemo(() => {
    const usedTokens = Number(latestAssistantMessage?.metadata?.context_tokens_used) || 0
    const capacity = contextCapacity || CONTEXT_CAPACITY
    const percent = capacity ? Math.min(100, Math.round((usedTokens / capacity) * 100)) : 0
    return { used: usedTokens, capacity, percent }
  }, [latestAssistantMessage, contextCapacity])

  return (
    <div className="min-h-screen bg-muted/40">
      <header className="border-b bg-background">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-4">
          <div>
            <h1 className="text-lg font-semibold">Clinical RAG Agent</h1>
            <p className="text-sm text-muted-foreground">
              Interactive health record for myeloma patients. Now supports follow-up questions. Agent assumes the questions are asked at the point of knowledge cutoff ('Date as of').
            </p>
          </div>
          <div />
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl gap-4 px-4 py-6 md:grid-cols-[260px_1fr] xl:grid-cols-[260px_1fr_300px]">
        <aside className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Patient</CardTitle>
              <CardDescription>Search by name or ID to select a patient.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <div className="relative">
                  <Input
                    value={patientSearchTerm}
                    onChange={(event) => {
                      setPatientSearchTerm(event.target.value)
                      setPatientListOpen(true)
                    }}
                    onFocus={() => setPatientListOpen(true)}
                    onBlur={() => {
                      // Delay closing to allow click handlers to run.
                      setTimeout(() => setPatientListOpen(false), 150)
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && filteredPatients.length > 0) {
                        event.preventDefault()
                        const first = filteredPatients[0]
                        setSelectedPatientId(first.id)
                        const displayName = first.name?.trim() || first.id
                        setPatientSearchTerm(`${displayName} (${first.id})`)
                        setPatientListOpen(false)
                      }
                    }}
                    placeholder="Search patients…"
                    disabled={patientsLoading || !patients.length}
                  />
                  {patientListOpen && (
                    <div className="absolute z-20 mt-1 w-full rounded-md border bg-popover shadow-xl">
                      <div className="max-h-64 overflow-y-auto divide-y">
                        {filteredPatients.length ? (
                          filteredPatients.map((patient) => {
                            const isActive = patient.id === selectedPatientId
                            return (
                              <button
                                key={patient.id}
                                type="button"
                                className={cn(
                                  "flex w-full items-center justify-between px-3 py-2 text-left text-sm",
                                  isActive ? "bg-muted font-semibold" : "hover:bg-muted/60"
                                )}
                                onMouseDown={(event) => event.preventDefault()}
                                onClick={() => {
                                  setSelectedPatientId(patient.id)
                                  setPatientSearchTerm("")
                                  setPatientListOpen(false)
                                }}
                              >
                                <div>
                                  <p className="font-medium">{patient.name || patient.id}</p>
                                  <p className="text-xs text-muted-foreground">{patient.id}</p>
                                  {(!patient.name?.trim() || patient.name === patient.id) && (
                                    <p className="text-[11px] text-amber-600">
                                      Patient is currently being added—will be available soon.
                                    </p>
                                  )}
                                </div>
                                {isActive && <Check className="h-4 w-4 text-primary" />}
                              </button>
                            )
                          })
                        ) : (
                          <p className="px-3 py-2 text-sm text-muted-foreground">No matches.</p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              {patientsLoading && (
                <p className="flex items-center text-sm text-muted-foreground">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading patients…
                </p>
              )}
              {patientsError && <p className="text-sm text-destructive">{patientsError}</p>}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Selected patient</CardTitle>
            <CardDescription>Currently active case.</CardDescription>
          </CardHeader>
          <CardContent>
            {selectedPatient ? (
              <div className="space-y-2 text-sm">
                <p className="font-semibold text-foreground">{selectedPatient.name || selectedPatient.id}</p>
                <p className="text-muted-foreground">{selectedPatient.id}</p>
                {selectedPatient.description && (
                  <p className="text-muted-foreground">{selectedPatient.description}</p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No patient selected.</p>
            )}
          </CardContent>
        </Card>
          {chatError && (
            <Card className="border-destructive/50">
              <CardHeader>
                <CardTitle className="text-destructive">Error</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive/80">{chatError}</p>
              </CardContent>
            </Card>
          )}
          <Card className="flex h-[48.5vh] flex-col">
            <CardHeader>
              <CardTitle>Available Reports by Type</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              {!selectedPatient && (
                <p className="text-sm text-muted-foreground">Select a patient to view report counts.</p>
              )}
              {selectedPatient?.latest_report_date && (
                <p className="text-muted-foreground">
                  Latest report: {formatIsoDate(selectedPatient.latest_report_date)}
                </p>
                )}
              {selectedPatient && !reportCountEntries.length && (
                <p className="text-sm text-muted-foreground">No reports found for this patient.</p>
              )}
              {reportCountEntries.length > 0 && (
                <ScrollArea className="h-full rounded-md border bg-background">
                  <div className="divide-y text-sm">
                    {reportCountEntries.map((entry) => (
                      <div key={entry.type} className="flex items-center justify-between px-3 py-2">
                        <span className="font-medium">{entry.type}</span>
                        <span className="text-muted-foreground">{entry.count}</span>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
          <Card>
          <CardHeader>
            <CardTitle>LLM Model</CardTitle>
            <CardDescription>Select the model.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Select
              value={selectedModel}
              onValueChange={setSelectedModel}
              disabled={modelsLoading || !models.length}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="gpt-oss-120b (OpenAI)" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {modelsLoading && (
              <p className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading models…
              </p>
            )}
            {modelsError && <p className="text-sm text-destructive">{modelsError}</p>}
          </CardContent>
        </Card>
        </aside>

        <section className="flex flex-col space-y-4">
          <Card className="flex h-[90vh] flex-col">
            <CardHeader>
              <CardTitle>Conversation</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col overflow-hidden gap-3">
              <Textarea
                rows={4}
                value={inputText}
                onChange={(event) => setInputText(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your question. Press Ctrl/⌘ + Enter (e.g. Does the patient meet the CAR-T eligibility criteria?)."
                disabled={!isPatientSelected}
              />
              {!isPatientSelected && (
                <p className="text-sm text-muted-foreground">
                  Select a patient to enable asking questions.
                </p>
              )}
              <ScrollArea className="flex-1 rounded-md border bg-background">
                <div className="space-y-4 p-4">
                  {messages.length === 0 && (
                    <p className="text-sm text-muted-foreground">
                      Ask a question above to get started.
                    </p>
                  )}
                  {messages.map((message) => (
                    <MessageBubble key={message.id} message={message} />
                  ))}
                </div>
              </ScrollArea>
              <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border bg-muted/30 px-3 py-2">
                <div className="flex items-center gap-3">
                  <ContextUsageIndicator {...contextUsage} />
                  <div className="flex flex-col leading-tight text-sm text-muted-foreground">
                    <span className="text-[0.65rem] uppercase tracking-wide">Context window</span>
                    <span className="text-xs">
                      {formatTokenCount(contextUsage.used)} / {formatTokenCount(contextUsage.capacity)} tokens used
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {isSending && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Thinking…
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleAbortRequest}
                        className="text-muted-foreground"
                      >
                        <Square className="mr-1 h-4 w-4" /> Interrupt
                      </Button>
                    </div>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={isSending}
                    onClick={handleClearChat}
                  >
                    <MessageSquarePlus className="mr-2 h-4 w-4" /> New conversation
                  </Button>
                  <Button
                    onClick={() => handleSend().catch(console.error)}
                    disabled={isSending || !inputText.trim() || !isPatientSelected}
                  >
                    Send <Send className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="flex h-[70vh] flex-col">
            <CardHeader>
              <CardTitle>Agent Monitoring (dev)</CardTitle>
              <CardDescription>Live steps and audit trail for the latest run for development.</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              <ScrollArea className="h-full rounded-md border bg-background p-3">
                <AgentEventTimeline events={stepEvents} />
              </ScrollArea>
              {monitoringError && <p className="mt-3 text-sm text-destructive">{monitoringError}</p>}
              {runId && (
                <p className="mt-3 text-xs text-muted-foreground">
                  Run ID: <span className="font-mono">{runId}</span>
                </p>
              )}
            </CardContent>
            <CardFooter className="flex items-center justify-between gap-3">
              <span className="text-xs text-muted-foreground">
                {isSending
                  ? "Live stream active…"
                  : stepEvents.length
                    ? "Latest agent steps"
                    : "No events yet"}
              </span>
              <div className="flex items-center gap-2">
                {monitoringLoading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => refreshMonitoring().catch(console.error)}
                  disabled={!runId || monitoringLoading}
                >
                  Refresh monitoring
                </Button>
              </div>
            </CardFooter>
          </Card>
        </section>

        <aside className="space-y-4 hidden xl:block">
          <Card>
            <CardHeader>
              <CardTitle>Baseline summary</CardTitle>
              <CardDescription>Auto-generated AI overview at diagnosis (can contain mistakes). Currently not visible to the model.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {!selectedPatient && (
                <p className="text-muted-foreground">Select a patient to view their baseline summary.</p>
              )}
              {selectedPatient && (
                <div className="space-y-2">
                  {selectedPatient.dob && (
                    <div className="rounded-md border p-2">
                      <p className="text-xs font-semibold uppercase text-muted-foreground">Date of Birth</p>
                      <p className="text-sm text-foreground">{formatIsoDate(selectedPatient.dob)}</p>
                    </div>
                  )}
                  {selectedPatient.latest_report_date && (
                    <div className="rounded-md border p-2">
                      <p className="text-xs font-semibold uppercase text-muted-foreground">Latest Report</p>
                      <p className="text-sm text-foreground">{formatIsoDate(selectedPatient.latest_report_date)}</p>
                    </div>
                  )}
                </div>
              )}
              {selectedPatient?.summary?.summary && (
                <p className="text-muted-foreground">{selectedPatient.summary.summary}</p>
              )}
              {selectedPatient && !hasSummaryFields && !selectedPatient.summary?.summary && (
                <p className="text-muted-foreground">
                  AI summary generation is in progress and will be available soon.
                </p>
              )}
              {hasSummaryFields && (
                <div className="space-y-2">
                  {summaryFieldEntries.map(([label, value]) => (
                    <div key={label} className="rounded-md border p-2">
                      <p className="text-xs font-semibold uppercase text-muted-foreground">{label}</p>
                      <p className="text-sm text-foreground">{value || "not reported"}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
          {latestCitations.length > 0 && (
            <Card className="flex h-[90vh] flex-col">
              <CardHeader>
                <CardTitle>Context snippets</CardTitle>
                <CardDescription>From the latest assistant response. Click on a snippet to open the full report.</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 overflow-hidden p-0">
                <div className="h-full overflow-y-auto pr-4">
                  <div className="space-y-3 p-4 text-sm">
                    {latestCitations.map((citation) => {
                      const indexValue = latestCitationMap.get(citation.id)
                      const badge = indexValue ? `[${indexValue}]` : "[?]"
                      const textSnippet = citation.snippet || "(no snippet provided)"
                      const headerLabel = [citation.type, citation.date].filter(Boolean).join(" • ")
                      const fallbackLabel = citation.label || "Citation"
                      const nodeForCitation =
                        (citation.id && latestContextLookup.get(citation.id)) || null
                      return (
                        <button
                          type="button"
                          key={citation.id}
                          className={cn(
                            "w-full rounded-lg border p-3 text-left space-y-1.5 transition",
                            nodeForCitation ? "hover:bg-muted/60" : "cursor-default"
                          )}
                          onClick={() => nodeForCitation && handleOpenReport(nodeForCitation)}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-baseline gap-2">
                              <div className="font-semibold leading-tight">
                                {headerLabel || fallbackLabel}
                              </div>
                              <span className="text-xs text-muted-foreground">{badge}</span>
                            </div>
                          </div>
                          <p className="text-xs text-muted-foreground">{textSnippet}</p>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </aside>
      </main>

      <Dialog open={reportDialogOpen} onOpenChange={(open) => (open ? null : closeReportDialog())}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Report preview</DialogTitle>
            <DialogDescription>
              {selectedContextNode?.report_type ?? reportPreview?.report_type ?? "Report"}{" "}
              {reportPreview?.report_date ? `• ${reportPreview.report_date}` : ""}
            </DialogDescription>
          </DialogHeader>
          {reportLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading report…
            </div>
          ) : reportError ? (
            <p className="text-sm text-destructive">{reportError}</p>
          ) : reportPreview ? (
            <div className="max-h-[60vh] overflow-y-auto pr-2">
              <pre className="whitespace-pre-wrap text-sm leading-relaxed">
                {highlightedReportContent.length > 0
                  ? highlightedReportContent
                  : reportPreview.content}
              </pre>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Select a context snippet to view the report.</p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

function AgentEventTimeline({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) {
    return <p className="text-sm text-muted-foreground">No events yet.</p>
  }

  return (
    <ul className="space-y-3">
      {events.map((event) => {
        const summary = describeEvent(event)
        return (
          <li key={event.id} className="rounded-md border bg-muted/30 p-3 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium text-foreground">{formatEventLabel(event.type)}</span>
              <time className="text-[10px] uppercase text-muted-foreground">
                {formatTimestamp(event.timestamp)}
              </time>
            </div>
            {summary && <p className="mt-1 text-muted-foreground">{summary}</p>}
            <details className="mt-2">
              <summary className="cursor-pointer text-[10px] text-muted-foreground">Show details</summary>
              <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-background p-2 font-mono text-[11px] leading-snug">
                {JSON.stringify(event.payload, null, 2)}
              </pre>
            </details>
          </li>
        )
      })}
    </ul>
  )
}

function formatEventLabel(type: string): string {
  return type
    .split("_")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : ""))
    .join(" ")
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000)
  if (Number.isNaN(date.getTime())) {
    return ""
  }
  return date.toLocaleTimeString()
}

function safeString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null
}

function describeEvent(event: AgentEvent): string | null {
  const payload = event.payload
  switch (event.type) {
    case "run_started": {
      const question = safeString(payload["question"])
      return question ? `Question: ${question}` : "Run started."
    }
    case "plan_ready":
      return "Plan prepared."
    case "execution_step": {
      const stepIndex = typeof payload["step_index"] === "number" ? (payload["step_index"] as number) : null
      return stepIndex !== null ? `Executing plan step ${stepIndex + 1}.` : "Executing plan step."
    }
    case "tool_started": {
      const toolName = safeString(payload["tool_name"]) ?? safeString(payload["tool"])
      return toolName ? `Started tool ${toolName}.` : "Tool execution started."
    }
    case "tool_finished": {
      const toolName = safeString(payload["tool_name"]) ?? safeString(payload["tool"])
      const summary = safeString(payload["summary"])
      if (summary) {
        return toolName ? `Finished tool ${toolName}: ${summary}` : `Finished tool: ${summary}`
      }
      return toolName ? `Finished tool ${toolName}.` : "Tool execution finished."
    }
    case "run_failed": {
      const message = safeString(payload["message"])
      return message ? `Error: ${message}` : "Run failed."
    }
    case "run_completed":
      return "Assistant response generated."
    default:
      return null
  }
}

function ContextUsageIndicator({
  percent,
}: {
  used: number
  capacity: number
  percent: number
}) {
  const normalized = Math.min(100, Math.max(0, percent))
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <div
        className="relative h-6 w-6 text-foreground"
        aria-label={`Context usage ${normalized}%`}
      >
        <svg viewBox="0 0 36 36" className="h-6 w-6">
          <path
            className="text-muted"
            stroke="currentColor"
            strokeWidth="4"
            strokeLinecap="round"
            fill="none"
            d="M18 2a16 16 0 1 1 0 32a16 16 0 1 1 0-32"
            opacity={0.2}
          />
          <path
            className="text-primary"
            stroke="currentColor"
            strokeWidth="4"
            strokeLinecap="round"
            fill="none"
            d={`M18 2a16 16 0 0 1 0 32`}
            strokeDasharray={`${normalized}, 100`}
          />
        </svg>
      </div>
      <span className="text-sm font-semibold text-foreground">{normalized}%</span>
    </div>
  )
}

function formatTokenCount(value: number): string {
  if (value >= 1_000_000) {
    return `${Math.round(value / 10_000) / 100}M`
  }
  if (value >= 1_000) {
    return `${Math.round(value / 100) / 10}k`
  }
  return `${value}`
}

function formatIsoDate(raw?: string | null): string {
  if (!raw) {
    return "-"
  }
  const trimmed = raw.trim()
  const isoMatch = /^\d{4}-\d{2}-\d{2}$/.test(trimmed)
  if (isoMatch) {
    const [year, month, day] = trimmed.split("-")
    return `${day}.${month}.${year}`
  }
  return trimmed
}

function renderInlineFormatting(text: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const regex = /\*\*([^*]+)\*\*|\*([^*]+)\*/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index))
    }
    const boldText = match[1] || match[2]
    nodes.push(
      <strong key={`bold-${match.index}-${boldText}`}>{boldText}</strong>
    )
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex))
  }
  return nodes.length ? nodes : [text]
}

function renderFormattedText(text: string): ReactNode {
  const lines = text.split(/\n/)
  return lines.map((line, idx) => (
    <Fragment key={`line-${idx}`}>
      {renderInlineFormatting(line)}
      {idx < lines.length - 1 ? <br /> : null}
    </Fragment>
  ))
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function highlightReportContent(content: string, snippet?: string): ReactNode[] {
  if (!snippet || !snippet.trim()) {
    return [content]
  }
  const tokens = snippet.trim().split(/\s+/).filter(Boolean)
  if (!tokens.length) {
    return [content]
  }
  const pattern = tokens.map((token) => escapeRegExp(token)).join("\\s+")
  if (!pattern) {
    return [content]
  }
  const regex = new RegExp(pattern, "gi")
  const nodes: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(content.slice(lastIndex, match.index))
    }
    const matchedText = content.slice(match.index, match.index + match[0].length)
    nodes.push(
      <mark key={`hit-${match.index}`} className="bg-amber-200 px-1 py-0.5 rounded">
        {matchedText}
      </mark>
    )
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < content.length) {
    nodes.push(content.slice(lastIndex))
  }

  return nodes.length ? nodes : [content]
}

function buildCitationIndex(
  nodes?: AgentContextNode[] | null,
  citations?: CitationMeta[] | null
): Map<string, number> {
  const map = new Map<string, number>()
  if (Array.isArray(nodes)) {
    nodes.forEach((node, idx) => {
      if (node.citation_id && !map.has(node.citation_id)) {
        map.set(node.citation_id, idx + 1)
      }
    })
  }
  if (Array.isArray(citations)) {
    citations.forEach((citation) => {
      if (!citation.id) {
        return
      }
      if (!map.has(citation.id)) {
        const nextIndex = map.size + 1
        map.set(citation.id, nextIndex)
      }
      if (Array.isArray(citation.aliases)) {
        citation.aliases.forEach((alias) => {
          if (alias && !map.has(alias)) {
            const mappedIndex = map.get(citation.id)
            if (mappedIndex) {
              map.set(alias, mappedIndex)
            }
          }
        })
      }
      if (citation.label === "Source" || citation.label === citation.id) {
        citation.label = "Unknown source"
      }
    })
  }
  return map
}

function renderCitationTag(id?: string, map?: Map<string, number>) {
  if (!id || !map) {
    return null
  }
  const index = map.get(id)
  if (!index) {
    return <span className="text-xs text-destructive">[?]</span>
  }
  return (
    <span className="text-xs text-muted-foreground">[{index}]</span>
  )
}

function replaceCitationTags(text: string | null, map?: Map<string, number>) {
  if (!text || !map) {
    return text
  }
  return text.replace(/\[([^\[\]]+)\]/g, (match, id) => {
    const index = map.get(id)
    return index ? `[${index}]` : match
  })
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isAssistant = message.role !== "user"
  const metadata = message.metadata
  const citationMap = useMemo(
    () => buildCitationIndex(metadata?.context_nodes, metadata?.citations),
    [metadata?.context_nodes, metadata?.citations]
  )
  const rawContent =
    (isAssistant && metadata?.final_answer?.trim()) ||
    (typeof message.content === "string" ? message.content.trim() : "")
  const bodyContent = rawContent ? replaceCitationTags(rawContent, citationMap) : null

  return (
    <div className={cn("flex flex-col gap-2", isAssistant ? "items-start" : "items-end")}>
      <div
        className={cn(
          "w-full max-w-2xl rounded-lg border p-4 text-sm shadow-sm",
          isAssistant ? "bg-card" : "bg-primary text-primary-foreground"
        )}
      >
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{isAssistant ? "Assistant" : "You"}</span>
          <time dateTime={message.createdAt}>{new Date(message.createdAt).toLocaleTimeString()}</time>
        </div>
        {bodyContent ? (
          <div className="mt-3 whitespace-pre-wrap text-sm leading-relaxed">
            {renderFormattedText(bodyContent)}
          </div>
        ) : null}
        {metadata?.context_nodes?.length ? (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
              Context snippets ({metadata.context_nodes.length})
            </summary>
            <ul className="mt-2 space-y-2 text-xs text-muted-foreground">
              {metadata.context_nodes.slice(0, 5).map((node, index) => (
                <li key={`${node.section_name ?? node.report_type}-${index}`}>
                  <span className="font-medium flex items-baseline gap-1">
                    <span>{node.report_type ?? node.section_name ?? "Section"}:</span>
                    {renderCitationTag(node.citation_id, citationMap)}
                  </span>
                  <span className="ml-1">{node.snippet ?? "(no snippet provided)"}</span>
                </li>
              ))}
              {metadata.context_nodes.length > 5 && <li>… additional snippets available on the server.</li>}
            </ul>
          </details>
        ) : null}
      </div>
    </div>
  )
}
