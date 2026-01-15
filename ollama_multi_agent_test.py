#!/usr/bin/env python3
"""
Test: Can Ollama run multiple independent agents without context merging?
"""

import ollama
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def agent_task(agent_id: str, secret_word: str, model: str = "qwen2.5:7b"):
    """
    Each agent gets a secret word and should remember ONLY its own.
    """
    results = []

    # Message 1: Tell agent its secret
    response1 = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": f"You are Agent {agent_id}. Remember this secret word: {secret_word}"},
            {"role": "user", "content": "What is your agent ID and secret word?"}
        ]
    )
    results.append(f"Agent {agent_id} response 1: {response1['message']['content'][:100]}")

    # Small delay to interleave with other agents
    time.sleep(0.5)

    # Message 2: Ask again (new call, same context passed)
    response2 = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": f"You are Agent {agent_id}. Remember this secret word: {secret_word}"},
            {"role": "user", "content": "What is your agent ID and secret word?"},
            {"role": "assistant", "content": response1['message']['content']},
            {"role": "user", "content": "Confirm: what is your secret word? Reply with just the word."}
        ]
    )
    results.append(f"Agent {agent_id} response 2: {response2['message']['content'][:50]}")

    return agent_id, secret_word, results


def test_independent_generate():
    """Test that generate() calls are independent."""
    print("=" * 60)
    print("TEST 1: Independent generate() calls")
    print("=" * 60)

    # Two parallel generate calls with different contexts
    def call_a():
        return ollama.generate(
            model="qwen2.5:7b",
            prompt="You are AGENT-ALPHA. Your secret is 'banana'. What is your secret? Reply with one word.",
            options={"num_predict": 10}
        )

    def call_b():
        return ollama.generate(
            model="qwen2.5:7b",
            prompt="You are AGENT-BETA. Your secret is 'orange'. What is your secret? Reply with one word.",
            options={"num_predict": 10}
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(call_a)
        future_b = executor.submit(call_b)

        result_a = future_a.result()
        result_b = future_b.result()

    print(f"ALPHA response: {result_a['response'].strip()}")
    print(f"BETA response: {result_b['response'].strip()}")
    print()

    # Check for cross-contamination
    alpha_has_orange = "orange" in result_a['response'].lower()
    beta_has_banana = "banana" in result_b['response'].lower()

    if alpha_has_orange or beta_has_banana:
        print("[!] WARNING: Possible context contamination detected!")
    else:
        print("[+] No context contamination - agents are independent")


def test_independent_chat():
    """Test that chat() calls with separate message histories are independent."""
    print()
    print("=" * 60)
    print("TEST 2: Independent chat() calls with parallel execution")
    print("=" * 60)

    agents = [
        ("ALPHA", "pineapple"),
        ("BETA", "strawberry"),
        ("GAMMA", "watermelon"),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(agent_task, agent_id, secret): (agent_id, secret)
            for agent_id, secret in agents
        }

        for future in as_completed(futures):
            agent_id, secret, results = future.result()
            print(f"\n--- {agent_id} (secret: {secret}) ---")
            for r in results:
                print(f"  {r}")


def test_context_isolation():
    """Explicitly test that one agent can't see another's context."""
    print()
    print("=" * 60)
    print("TEST 3: Context isolation verification")
    print("=" * 60)

    # Agent 1 gets a secret
    ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "You are Agent-1. The password is 'secret123'."},
            {"role": "user", "content": "Remember the password."}
        ]
    )

    # Agent 2 (fresh context) tries to get Agent 1's secret
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "user", "content": "What password did the previous agent receive? If you don't know, say 'I have no context from other agents'."}
        ]
    )

    print(f"Agent-2 (fresh context) response: {response['message']['content'][:200]}")

    if "secret123" in response['message']['content']:
        print("\n[!] FAIL: Context leaked between agents!")
    else:
        print("\n[+] PASS: Contexts are isolated - Agent-2 has no knowledge of Agent-1's secret")


def main():
    print("Ollama Multi-Agent Independence Test")
    print("Model: qwen2.5:7b")
    print()

    test_independent_generate()
    test_independent_chat()
    test_context_isolation()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Ollama's architecture:
- Each API call (generate/chat) is STATELESS
- Context is passed explicitly via 'messages' array (chat) or 'prompt' (generate)
- Multiple concurrent calls share the MODEL but not CONTEXT
- No automatic context merging between calls

For multi-agent systems:
- Each agent maintains its own message history
- Pass the full conversation to each call
- Agents cannot see each other's contexts
- Safe for parallel execution
""")


if __name__ == "__main__":
    main()
