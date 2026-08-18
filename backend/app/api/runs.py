from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth.dependencies import get_current_user
from app.database.db import User, get_run, list_agent_steps, list_runs
from app.models.schemas import AgentStepOut, RunDetailOut, RunRecordOut

router = APIRouter()


def _run_out(record) -> RunRecordOut:
	return RunRecordOut(
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
		cost_usd=record.cost_usd,
		moderation_blocked=record.moderation_blocked,
		conversation_id=record.conversation_id,
	)


@router.get("/runs", response_model=list[RunRecordOut])
async def get_runs(
	limit: int = Query(default=20, ge=1, le=100),
	current_user: User = Depends(get_current_user),
) -> list[RunRecordOut]:
	"""Personal mission history — always scoped to the authenticated user's own id.

	The user id comes from the verified JWT (get_current_user), never from a query param,
	so one user can never list another user's missions by guessing/passing an id.
	"""
	records = await list_runs(limit=limit, user_id=current_user.id)
	return [_run_out(record) for record in records]


@router.get("/runs/{run_id}", response_model=RunDetailOut)
async def get_run_detail(run_id: str, current_user: User = Depends(get_current_user)) -> RunDetailOut:
	record = await get_run(run_id, user_id=current_user.id)
	if record is None:
		# Same 404 whether the run doesn't exist or belongs to someone else — does not leak
		# which case it is.
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found.")

	steps = await list_agent_steps(run_id)
	step_outs = [
		AgentStepOut(
			agent=step.agent,
			step_index=step.step_index,
			model=step.model,
			provider=step.provider,
			duration_ms=step.duration_ms,
			input_tokens=step.input_tokens,
			output_tokens=step.output_tokens,
			cost_usd=step.cost_usd,
			retrieved_count=step.retrieved_count,
			status=step.status,
			error=step.error,
		)
		for step in steps
	]

	total_duration_ms = sum((s.duration_ms or 0.0) for s in step_outs) or None
	total_tokens = sum((s.input_tokens or 0) + (s.output_tokens or 0) for s in step_outs) or None
	total_cost_usd = sum((s.cost_usd or 0.0) for s in step_outs) or None

	return RunDetailOut(
		**_run_out(record).model_dump(),
		steps=step_outs,
		total_duration_ms=total_duration_ms,
		total_tokens=total_tokens,
		total_cost_usd=total_cost_usd,
	)
