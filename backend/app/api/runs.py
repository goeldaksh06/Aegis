from fastapi import APIRouter, Query

from app.database.db import list_runs
from app.models.schemas import RunRecordOut

router = APIRouter()


@router.get("/runs", response_model=list[RunRecordOut])
async def get_runs(limit: int = Query(default=20, ge=1, le=100)) -> list[RunRecordOut]:
	records = await list_runs(limit=limit)
	return [
		RunRecordOut(
			id=record.id,
			created_at=record.created_at,
			prompt=record.prompt,
			status=record.status,
			agent=record.agent,
			model=record.model,
			provider=record.provider,
			risk_level=record.risk_level,
			risk_score=record.risk_score,
			latency_ms=record.latency_ms,
			error=record.error,
		)
		for record in records
	]
