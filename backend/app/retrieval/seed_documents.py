from __future__ import annotations

from app.retrieval.indexer import DocumentIndexer
from app.retrieval.retriever import Retriever

# A small set of realistic reference documents so the RAG pipeline has something real to
# retrieve from out of the box, instead of an empty index that always returns zero results.
# These are authored reference material (not sourced from any real company). Coverage
# deliberately spans all four scenario domains offered as one-click chips in App.tsx
# (supply chain, cyber incident, market shock, natural disaster) — an earlier version only
# covered supply chain, which meant three of the four scenario chips always showed
# retrieved_count: 0 despite the RAG pipeline working correctly; that was a real, visible
# inconsistency, not a hypothetical one.
SEED_DOCUMENTS: list[tuple[str, str]] = [
    (
        "supply-chain-contingency-playbook",
        """
        Supply Chain Contingency Playbook (Internal Reference)

        Purpose: standard response procedure when a primary supplier or shipping lane becomes
        unavailable without warning.

        Trigger conditions: a primary supplier misses a scheduled check-in by more than 24
        hours, or a shipping lane is confirmed disrupted (port closure, carrier suspension, or
        customs blockage) affecting more than 10% of inbound volume.

        Immediate response (first 48 hours): activate the secondary logistics hub, reroute any
        in-transit shipments where possible, and open communication with all backup vendors on
        the approved alternate-supplier list. Do not wait for confirmation of the root cause
        before activating contingency routing — the cost of an unnecessary reroute is far lower
        than the cost of a missed fulfillment window.

        Escalation thresholds: if fulfillment capacity is projected to drop below 80% of
        committed volume for the current quarter, escalate to executive leadership within 24
        hours with a revised delivery timeline. If projected capacity drops below 60%, customer
        communications must go out before day 3 of the disruption.

        Historical note: the 2025 Q2 regional freight slowdown showed that disruptions
        communicated to customers within 72 hours had 40% fewer cancelled orders than
        disruptions where communication was delayed past a week.
        """,
    ),
    (
        "port-disruption-response-policy",
        """
        Port Disruption Response Policy

        Scope: applies to any confirmed closure, congestion event, or capacity reduction at a
        primary shipping port used for inbound or outbound freight.

        Standard mitigations, in order of preference:
        1. Reroute through the pre-approved secondary port with existing customs clearance
           agreements — this is the fastest option and should be attempted first.
        2. Shift the highest-priority SKUs (top 20% by revenue contribution) to air freight if
           rerouting adds more than 10 days to transit time.
        3. Activate domestic backup suppliers for any SKU where the primary supplier's region is
           directly affected by the port disruption.

        Cost guidance: air freight typically costs 4-6x standard ocean freight per unit, so it
        should only be used for revenue-critical SKUs or when contractual delivery penalties
        exceed the freight cost premium.

        Typical timelines: port congestion events historically resolve within 10-19 days from
        first disruption; permanent rerouting decisions should not be made before day 5 unless
        the port closure is confirmed indefinite by official notice.
        """,
    ),
    (
        "vendor-risk-management-guidelines",
        """
        Vendor Risk Management Guidelines

        Vendor risk tiers: Tier 1 vendors supply more than 15% of any single product category
        and require a documented backup vendor at all times. Tier 2 vendors supply 5-15% and
        require an identified (but not necessarily contracted) backup. Tier 3 vendors below 5%
        do not require a backup on file.

        Compliance review cadence: Tier 1 vendor compliance status (certifications, regulatory
        standing, financial health indicators) must be reviewed quarterly. A vendor moving from
        "compliant" to "under review" status is itself an escalation trigger, independent of any
        delivery disruption, since compliance issues frequently precede delivery issues by 2-6
        weeks.

        Communication SLA during an active disruption: a Tier 1 vendor that goes unresponsive
        must be escalated to the vendor management team within 24 hours. Standard practice is to
        attempt contact through at least two channels (primary account contact and emergency
        contact on file) before treating the vendor as non-responsive for planning purposes.
        """,
    ),
    (
        "q3-regional-freight-incident-report",
        """
        Incident Report: Q3 Regional Freight Delay (Internal Post-Mortem)

        Summary: a Tier 1 supplier in the affected region became unresponsive following a
        regional port disruption, creating a 14-day gap in inbound inventory for three product
        lines.

        Impact: on-time fulfillment SLA dropped from a 96% baseline to 61% during the affected
        weeks. Estimated revenue at risk was calculated at $4.1-5.3M based on order backlog,
        before mitigation.

        Response taken: contingency logistics was activated on day 2, two domestic backup
        suppliers were qualified within 24 hours, and the highest-revenue SKUs were shifted to
        air freight, adding approximately $900K in expedited freight cost. This limited the
        actual revenue impact to approximately 17-22% of the at-risk estimate.

        Lesson recorded: the fastest-recovering response path combined immediate backup-supplier
        qualification with selective air freight for only the top-revenue SKUs, rather than
        either extreme (full air freight for everything, or waiting for the primary supplier to
        resume before acting).
        """,
    ),
    (
        "data-breach-incident-response-plan",
        """
        Data Breach Incident Response Plan (Internal Reference)

        Purpose: standard procedure for a suspected or confirmed unauthorized access to customer
        or company data, including database exposure, credential leaks, or unauthorized system
        access.

        Immediate response (first 2 hours): isolate affected systems from the network without
        powering them off (preserves forensic evidence), rotate all credentials with access to
        the affected system, and activate the incident response team including legal and
        communications. Do not attempt to determine root cause before containment — containment
        always comes first.

        Notification thresholds: if personal data (names, emails, payment details) is confirmed
        exposed, regulatory notification requirements typically apply within 72 hours in most
        jurisdictions. Legal counsel must confirm the exact jurisdiction-specific deadline before
        any public disclosure, but internal breach-response preparation should begin immediately,
        not after that confirmation.

        Customer communication guidance: breaches disclosed proactively within 24-48 hours of
        confirmation, with clear guidance on protective steps (password resets, credit monitoring
        offers), consistently result in lower customer attrition than breaches where disclosure
        is delayed pending full investigation completion.

        Historical note: the average cost of a data breach increases significantly for each week
        containment is delayed — internal analysis of past incidents showed cost roughly doubling
        between a same-day containment and a two-week containment window.
        """,
    ),
    (
        "cybersecurity-severity-classification",
        """
        Cybersecurity Incident Severity Classification

        Severity 1 (Critical): confirmed data exfiltration, active ransomware, or production
        system compromise with customer-facing impact. Requires executive notification within 1
        hour and continuous incident command until resolved.

        Severity 2 (High): suspected but unconfirmed breach, or a contained compromise with no
        evidence of data exfiltration. Requires security team lead notification within 4 hours
        and a full assessment within 24 hours.

        Severity 3 (Moderate): a vulnerability discovered before exploitation, or an isolated
        phishing attempt that did not result in credential compromise. Standard triage timelines
        apply, no emergency escalation required.

        Forensics guidance: system logs and access records must be preserved before any remediation
        action that could overwrite them (e.g. rebuilding a compromised server). Rebuilding before
        preserving logs is one of the most common mistakes that prevents determining the actual
        scope of a breach after the fact.
        """,
    ),
    (
        "market-volatility-response-playbook",
        """
        Market Volatility Response Playbook (Internal Reference)

        Purpose: standard response when a sudden market event (competitor failure, sector-wide
        stock drop, major economic announcement) creates material uncertainty for the business.

        Immediate response (first 24 hours): avoid reactive public statements before internal
        exposure analysis is complete. Quantify direct exposure (contracts, receivables, or
        partnerships tied to the affected party or sector) before assessing indirect exposure
        (customer confidence, investor sentiment, competitor positioning).

        Decision framework: a market shock does not require an immediate strategic pivot unless
        direct financial exposure exceeds 5% of quarterly revenue, or the event signals a
        structural (not cyclical) shift in the market. Overreacting to short-term volatility with
        major strategic changes has historically cost more than the volatility itself in most
        internal post-mortems.

        Stakeholder communication: investor and board communication should distinguish clearly
        between "direct exposure" (quantified) and "market sentiment risk" (qualitative) — conflating
        the two in early communications has historically led to overcorrection by stakeholders.
        """,
    ),
    (
        "competitor-failure-case-study",
        """
        Case Study: Competitor Bankruptcy Market Response (Internal Post-Mortem)

        Summary: a major competitor filed for bankruptcy following sustained losses, triggering a
        sector-wide stock price drop of approximately 15% within 48 hours, including for companies
        with no direct financial exposure to the failed competitor.

        Analysis: the company's direct financial exposure was assessed at under 1% of quarterly
        revenue (no material contracts or receivables tied to the failed competitor), meaning the
        stock impact was driven by sector sentiment contagion, not fundamentals.

        Response taken: no major strategic pivot was made. Instead, a targeted investor
        communication clarified the lack of direct exposure within 48 hours, and normal business
        operations continued unchanged. Stock price recovered to within 3% of its pre-event level
        within three weeks.

        Lesson recorded: for sentiment-driven (not fundamentals-driven) market shocks, rapid,
        specific, quantified communication recovers investor confidence faster than a broad
        strategic response — a broad response can itself signal that management believes there is
        a real problem, prolonging the impact.
        """,
    ),
    (
        "business-continuity-severe-weather-plan",
        """
        Business Continuity Plan: Severe Weather / Natural Disaster (Internal Reference)

        Purpose: standard response when severe weather (hurricane, flood, wildfire, earthquake) is
        projected to affect a facility with more than 48 hours of advance warning.

        Pre-event response (48-24 hours before impact): activate the facility's business
        continuity plan, confirm alternate facility capacity can absorb at least 60% of the
        primary facility's throughput, and begin proactively shifting non-time-sensitive
        operations to unaffected facilities. Employee safety planning (evacuation routes,
        remote-work activation) takes priority over operational continuity in all cases.

        Inventory and asset protection: for facilities with physical inventory, prioritize moving
        or elevating the highest-value 20% of inventory by revenue contribution if full relocation
        is not feasible in the available time window.

        Recovery timeline guidance: based on prior events, facilities that pre-activated their
        continuity plan more than 24 hours before impact returned to full operational capacity
        30-40% faster than facilities that began response planning after the event occurred.

        Communication: customer-facing communication about expected delays should go out as soon
        as the continuity plan is activated, not after impact is confirmed — advance notice
        measurably reduces customer cancellation rates in prior events.
        """,
    ),
    (
        "hurricane-impact-incident-report",
        """
        Incident Report: Distribution Center Hurricane Impact (Internal Post-Mortem)

        Summary: a category 3 hurricane made landfall near a primary distribution center with 60
        hours of advance warning. The business continuity plan was activated 50 hours before
        landfall.

        Impact: the facility was closed for 6 days. Approximately 70% of throughput was
        successfully shifted to two alternate facilities before landfall; the remaining 30% of
        capacity was unavailable during the closure window, resulting in an estimated $2.8M in
        delayed fulfillment, of which roughly $1.9M was recovered through expedited shipping once
        the facility reopened.

        Response taken: the top 20% of inventory by revenue was relocated to an inland facility
        before landfall based on continuity-plan guidance; that inventory sustained zero losses,
        while unrelocated inventory sustained partial water damage.

        Lesson recorded: the pre-activation timeline (50 hours before landfall) was inside the
        continuity plan's recommended window and is credited with the successful 70% throughput
        shift; post-mortem recommended lowering the activation trigger threshold from 48 to 60
        hours for category 3+ storms specifically, since coordination with alternate facilities
        took longer than planned.
        """,
    ),
]


def index_seed_documents(indexer: DocumentIndexer, retriever: Retriever) -> int:
    """Index the built-in reference documents into the shared retrieval runtime.

    Called once per process (see app_container.get_retrieval_runtime) so the RAG pipeline
    has real material to retrieve from instead of an empty index. Returns the number of
    chunks indexed, mainly for logging/debugging.
    """
    total_chunks = 0

    for source, text in SEED_DOCUMENTS:
        chunks = indexer.index_text(text=text, source=source)
        retriever.add_chunks(chunks)
        total_chunks += len(chunks)

    return total_chunks
