# Full outputs — think_budget + json_schema sweep

Worker image `lia-vllm-chat-template-kwargs-868f031` (schema gate loosened: structured_outputs allowed when reasoning_parser is active).

All 3 runs emit valid `{"fouls":[...]}` JSON. `</think>` does not appear because the worker strips the think block server-side when `supports_thinking` is true; only post-`</think>` content (the JSON) is streamed.

## budget=None

- finish_reason: `stop`
- elapsed: 67.5s
- prompt_tokens: 27731
- completion_tokens: 1378

```json
{
  "fouls": [
    {
      "start": "00:16",
      "end": "00:21",
      "description": "Ashley Young commits a foul on Sadio Mané by sliding in and taking him down."
    },
    {
      "start": "01:28",
      "end": "01:32",
      "description": "Fred commits a foul on Jordan Henderson with a sliding tackle."
    },
    {
      "start": "05:25",
      "end": "05:30",
      "description": "Chris Smalling commits a foul on Sadio Mané inside the penalty area, resulting in a penalty kick."
    }
  ]
}
```

## budget=10

- finish_reason: `stop`
- elapsed: 31.5s
- prompt_tokens: 27731
- completion_tokens: 284

```json
{
  "fouls": [
    {
      "start": "16.0",
      "end": "21.0",
      "description": "A Manchester United player, number 27, commits a foul by tripping a Liverpool player, number 8, who falls to the ground."
    },
    {
      "start": "81.0",
      "end": "92.0",
      "description": "A Manchester United player, number 23, commits a foul by tripping a Liverpool player, number 10, who falls to the ground."
    },
    {
      "start": "100.0",
      "end": "106.0",
      "description": "A replay shows a Manchester United player, number 23, committing a foul by tripping a Liverpool player, number 10."
    },
    {
      "start": "106.0",
      "end": "110.0",
      "description": "A wide shot shows a Manchester United player, number 23, committing a foul by tripping a Liverpool player, number 10."
    }
  ]
}
```

## budget=100

- finish_reason: `stop`
- elapsed: 32.1s
- prompt_tokens: 27731
- completion_tokens: 328

```json
{
    "fouls": [
        {
            "start": "00:16",
            "end": "00:21",
            "description": "A Manchester United player tackles a Liverpool player from behind, causing him to fall."
        },
        {
            "start": "01:21",
            "end": "01:32",
            "description": "A Manchester United player slides in and trips a Liverpool player, leading to a free kick."
        },
        {
            "start": "01:40",
            "end": "01:49",
            "description": "A replay shows a Manchester United player tripping a Liverpool player near the sideline."
        },
        {
            "start": "07:15",
            "end": "07:22",
            "description": "A Manchester United player commits a foul on a Liverpool player, resulting in a free kick."
        }
    ]
}
```
