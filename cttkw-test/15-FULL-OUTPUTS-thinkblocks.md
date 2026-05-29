# budget + schema + think_blocks separation

Sweep on worker image `lia-vllm-chat-template-kwargs-87edc21` (v2 plumbing) via `/v1/chat/completions` against the same soccer mp4 (`[557, 224, 448, 3]`, fps=1, 0..557s).

Verifies that even with `--reasoning-parser qwen3` + `thinking_token_budget` + `response_format=json_schema` enabled, the worker's manual `<think>` partition (PR #80's `_extract_visible_text_and_think_blocks` via `assistant_think_prefix`) still extracts the reasoning into `think_blocks`. vLLM does not strip `<think>` from the raw completion output (`output.outputs[0].text`), so the existing partition logic continues to work alongside the parser-enforced budget + guided decoding.

## Compare

| budget | completion | think_blocks chars | text chars (JSON) | finish |
|---|---|---|---|---|
| None | 1532 | **4276** | 418 | `stop` |
| 10 | 362 | **42** | 1110 | `stop` |
| 100 | 332 | **356** | 841 | `stop` |

## budget=None

- finish_reason: `stop`
- elapsed: 87.8s
- prompt_tokens: 27731
- completion_tokens: 1532
- think_blocks: 1 item(s), 4276 chars total

### `think_blocks[0]` (raw reasoning, separately extracted)

```text
The user wants me to identify fouls in the video.

1.  **Scan the video for fouls:** I'll watch the video and look for instances where a player commits a foul, leading to a stoppage in play or a free kick.
2.  **Identify the first foul:** Around 00:16, a Liverpool player (Sadio Mané, #10) is dribbling the ball. A Manchester United player (Ashley Young, #18) slides in and tackles him from behind/side, bringing him down. The referee blows the whistle. This is clearly a foul.
3.  **Identify the second foul:** Around 01:28, a Liverpool player (Jordan Henderson, #14) is running with the ball. A Manchester United player (Fred, #17) slides in and tackles him. The referee blows the whistle. This is another foul.
4.  **Identify the third foul:** Around 05:25, a Liverpool player (Sadio Mané, #10) is dribbling towards the goal. A Manchester United player (Chris Smalling, #12) slides in and tackles him inside the penalty area. The referee blows the whistle and points to the penalty spot. This is a penalty foul.
5.  **Format the output:** I will create a JSON object with a "fouls" array, containing objects for each identified foul with start and end timestamps and a description.

*   **Foul 1:** 00:16 - 00:21. Ashley Young tackles Sadio Mané.
*   **Foul 2:** 01:28 - 01:32. Fred tackles Jordan Henderson.
*   **Foul 3:** 05:25 - 05:30. Chris Smalling tackles Sadio Mané in the box.

Let's refine the timestamps.
*   Foul 1 starts around 00:16 when the tackle happens and ends around 00:21 when the players are on the ground and the referee is involved.
*   Foul 2 starts around 01:28 when the tackle happens and ends around 01:32 when the referee signals.
*   Foul 3 starts around 05:25 when the tackle happens and ends around 05:30 when the referee signals the penalty.

Wait, looking closer at the sequence around 01:28. The player in red (Henderson) is tackled by a player in blue (Fred). The referee blows the whistle. This is a foul.
Looking at the sequence around 05:25. The player in red (Mané) is tackled by a player in blue (Smalling) inside the box. The referee blows the whistle and points to the spot. This is a penalty foul.

So I have three distinct fouls.

Let's double check the timestamps.
- 00:16: Mané is tackled by Young.
- 01:28: Henderson is tackled by Fred.
- 05:25: Mané is tackled by Smalling.

The request asks for "every foul". I will include these three.

One more check. Are there any other fouls?
- 00:00-00:15: Just general play and a close up of Ibrahimovic.
- 00:22-01:27: Replays of the first foul and general play.
- 01:33-05:24: Replays of the second foul, general play, a free kick, a shot saved by De Gea.
- 05:31-end: Replay of the penalty foul and general play.

So, the three identified fouls are the main ones.

Let's refine the descriptions.
1.  Ashley Young commits a foul on Sadio Mané.
2.  Fred commits a foul on Jordan Henderson.
3.  Chris Smalling commits a foul on Sadio Mané, resulting in a penalty.

Final check of timestamps.
- Foul 1: 00:16 to 00:21.
- Foul 2: 01:28 to 01:32.
- Foul 3: 05:25 to 05:30.

The output format requires a JSON object.
```json
{
    "fouls": [
        {
            "start": "00:16",
            "end": "00:21",
            "description": "Ashley Young commits a foul on Sadio Mané."
        },
        {
            "start": "01:28",
            "end": "01:32",
            "description": "Fred commits a foul on Jordan Henderson."
        },
        {
            "start": "05:25",
            "end": "05:30",
            "description": "Chris Smalling commits a foul on Sadio Mané, resulting in a penalty."
        }
    ]
}
```
Wait, looking at the video again, the segment from 01:28 to 01:32 is actually a replay of the foul that happened earlier? No, let's look at the timestamps on the screen.
- At 00:16, the clock says 28:25.
- At 01:28, the clock says 29:37. This is a different foul.
- At 05:25, the clock says 33:33. This is a different foul.

Okay, so they are distinct events.

Let's re-verify the players.
- 00:16: Mané (Liverpool #10) vs Young (Man Utd #18). Correct.
- 01:28: Henderson (Liverpool #14) vs Fred (Man Utd #17). Correct.
- 05:25: Mané (Liverpool #10) vs Smalling (Man Utd #12). Correct.

The timestamps seem accurate. I will proceed with generating the JSON.
```

### `text` (clean JSON answer, post-`</think>`)

```json
{
  "fouls": [
    {
      "start": "00:16",
      "end": "00:21",
      "description": "Ashley Young commits a foul on Sadio Mané."
    },
    {
      "start": "01:28",
      "end": "01:32",
      "description": "Fred commits a foul on Jordan Henderson."
    },
    {
      "start": "05:25",
      "end": "05:30",
      "description": "Chris Smalling commits a foul on Sadio Mané, resulting in a penalty."
    }
  ]
}
```

## budget=10

- finish_reason: `stop`
- elapsed: 52.2s
- prompt_tokens: 27731
- completion_tokens: 362
- think_blocks: 1 item(s), 42 chars total

### `think_blocks[0]` (raw reasoning, separately extracted)

```text
The user wants me to identify fouls in the
```

### `text` (clean JSON answer, post-`</think>`)

```json
{
  "fouls": [
    {
      "start": "16.0",
      "end": "21.0",
      "description": "A Manchester United player, Ashley Young, commits a foul by tripping a Liverpool player, Sadio Mané, as he attempts to dribble the ball."
    },
    {
      "start": "82.0",
      "end": "92.0",
      "description": "A Manchester United player, Ashley Young, commits a foul by tripping a Liverpool player, Sadio Mané, as he attempts to dribble the ball."
    },
    {
      "start": "100.0",
      "end": "106.0",
      "description": "A Manchester United player, Ashley Young, commits a foul by tripping a Liverpool player, Sadio Mané, as he attempts to dribble the ball."
    },
    {
      "start": "106.0",
      "end": "110.0",
      "description": "A Manchester United player, Ashley Young, commits a foul by tripping a Liverpool player, Sadio Mané, as he attempts to dribble the ball."
    },
    {
      "start": "435.0",
      "end": "441.0",
      "description": "A Manchester United player, Ashley Young, commits a foul by tripping a Liverpool player, Sadio Mané, as he attempts to dribble the ball."
    }
  ]
}
```

## budget=100

- finish_reason: `stop`
- elapsed: 37.3s
- prompt_tokens: 27731
- completion_tokens: 332
- think_blocks: 1 item(s), 356 chars total

### `think_blocks[0]` (raw reasoning, separately extracted)

```text
The user wants me to identify fouls in the video.

1.  **Scan the video for fouls:** I'll watch the video and look for instances where a player commits a foul, leading to a stoppage in play or a free kick.
2.  **Identify the first foul:** Around 00:16, a Liverpool player (Sadio Mané, #10) is dribbling the ball. A Manchester United player (Ashley Young, #
```

### `text` (clean JSON answer, post-`</think>`)

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
            "start": "05:24",
            "end": "05:32",
            "description": "A Manchester United player commits a foul on a Liverpool player inside the penalty area, resulting in a penalty kick."
        }
    ]
}
```
