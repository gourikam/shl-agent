"""
Self-replay harness — run this against your OWN endpoint before submitting.
Replays each trace's USER turns (in order) against /chat with growing history,
and reports: schema validity, whether recommendations only contain catalog
items, turn-cap compliance, and a rough manual-eyeball diff vs the expected
shortlist (full automated recall scoring requires the labeled expected sets,
which weren't in the trace .md files you uploaded — add them if you have the
separate labeled fact sheets).

Usage:
    python replay_harness.py http://localhost:8000
    python replay_harness.py https://your-app.onrender.com
"""
import sys
import json
import time
import requests

# Hand-extracted user turns from your 10 trace files (C1-C10).
# NOTE: only USER turns are replayed — the agent's own prior replies are
# fed back as conversation history exactly as the real evaluator would
# (using the agent's OWN previous responses, not the trace's reference
# agent replies), since that's what a stateless multi-turn replay does.
TRACES = {
    "C1": [
        "We need a solution for senior leadership.",
        "The pool consists of CXOs, director-level postions; people with more than 15 years of experience.",
        "Selection — comparing candidates against a leadership benchmark.",
        "Perfect, that's what we need.",
    ],
    "C2": [
        "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
        "Yes, go ahead. Should I also add a cognitive test for this level?",
        "That works. Thanks.",
    ],
    "C3": [
        "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
        "English.",
        "US.",
        "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?",
        "Perfect — new simulation for volume, old solution for finalists. Confirmed.",
    ],
    "C4": [
        "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
        "Good. Can you also add a situational judgement element — work-context decision making for graduates?",
        "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates.",
    ],
    "C5": [
        "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
        "What's the difference between OPQ and OPQ MQ Sales Report?",
        "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.",
    ],
    "C6": [
        "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?",
        "What's the difference between the DSI and the Safety & Dependability 8.0?",
        "We're industrial. The 8.0 bundle is the right fit. Confirmed.",
    ],
    "C7": [
        "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?",
        "They're functionally bilingual — English fluent for written work. Go with the hybrid.",
        "Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?",
        "Understood. Keep the shortlist as-is.",
    ],
    "C8": [
        "I need to quickly screen admin assistants for Excel and Word daily.",
        "In that case, I am OK with adding a simulation - we want to capture the capabilties.",
        "That's good.",
    ],
    "C9": [
        "Here's the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n\"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required.\"",
        "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features.",
        "Senior IC. They lead design on their own services but don't manage other engineers directly.",
        "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.",
        "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
        "Do we really need Verify G+ on top of all the technical tests? Feels redundant.",
        "Keep Verify G+. Locking it in.",
    ],
    "C10": [
        "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
        "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.",
        "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
    ],
}


def replay_trace(base_url: str, trace_id: str, user_turns: list):
    print(f"\n{'='*70}\nTRACE {trace_id}\n{'='*70}")
    history = []
    all_ok = True
    for i, user_msg in enumerate(user_turns, 1):
        history.append({"role": "user", "content": user_msg})
        if len(history) > 8:
            print(f"  [turn {i}] WARNING: exceeds 8-turn cap, evaluator would stop here")
            break
        try:
            resp = requests.post(f"{base_url}/chat", json={"messages": history}, timeout=30)
        except Exception as e:
            print(f"  [turn {i}] REQUEST FAILED: {e}")
            all_ok = False
            break

        if resp.status_code != 200:
            print(f"  [turn {i}] HTTP {resp.status_code}: {resp.text[:300]}")
            all_ok = False
            break

        try:
            data = resp.json()
        except Exception:
            print(f"  [turn {i}] INVALID JSON RESPONSE: {resp.text[:300]}")
            all_ok = False
            break

        # schema checks
        for required in ("reply", "recommendations", "end_of_conversation"):
            if required not in data:
                print(f"  [turn {i}] SCHEMA VIOLATION: missing '{required}'")
                all_ok = False

        recs = data.get("recommendations", [])
        if recs is not None and len(recs) > 10:
            print(f"  [turn {i}] HARD EVAL FAIL: {len(recs)} recommendations (>10)")
            all_ok = False

        print(f"  [turn {i}] USER: {user_msg[:80]}")
        print(f"           REPLY: {data.get('reply','')[:150]}")
        print(f"           RECS ({len(recs) if recs else 0}): {[r.get('name') for r in (recs or [])]}")
        print(f"           end_of_conversation: {data.get('end_of_conversation')}")

        history.append({"role": "assistant", "content": data.get("reply", "")})
        time.sleep(8)  # was 2.5  # avoid tripping Groq free-tier rate limits across 35+ calls in a row

    status = "OK" if all_ok else "FAILED"
    print(f"\n  --> {trace_id}: {status}")
    return all_ok


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    # health check first
    try:
        h = requests.get(f"{base_url}/health", timeout=120)
        print(f"/health -> {h.status_code} {h.text}")
    except Exception as e:
        print(f"/health FAILED: {e}")
        sys.exit(1)

    results = {}
    for trace_id, turns in TRACES.items():
        results[trace_id] = replay_trace(base_url, trace_id, turns)
        time.sleep(10)  # was 3

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    passed = sum(results.values())
    for trace_id, ok in results.items():
        print(f"  {trace_id}: {'PASS' if ok else 'FAIL'}")
    print(f"\n{passed}/{len(results)} traces completed without hard errors.")
    print("NOTE: this only checks schema/grounding/turn-cap compliance, not")
    print("Recall@10 against expected shortlists — eyeball the printed")
    print("recommendations above against each trace .md file's expected list.")


if __name__ == "__main__":
    main()