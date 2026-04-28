import json

from src.agent_tools import FullContextTool, ReportsRAGTool

def test_reports_rag_tool_returns_matches():
    tool = ReportsRAGTool()
    result = tool(query="Myelom", report_type="doctor_letter")
    data = json.loads(result.content)
    nodes = data.get("context_nodes")
    assert isinstance(nodes, list) and len(nodes) > 0
    assert any(node.get("patient_id") == tool.patient_id for node in nodes)


def test_full_context_tool_returns_summaries():
    tool = FullContextTool()
    result = tool(query="Pneumonie", chunk_count=2)
    payload = result.raw_output or {}
    nodes = payload.get("context_nodes")
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    assert all("summary" in (node.get("section_name") or "").lower() for node in nodes)
