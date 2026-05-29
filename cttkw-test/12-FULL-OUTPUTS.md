# Full model output per thinking_token_budget

Raw `text` field captured via SSE `delta.content` stream from the chat completions endpoint, for each sweep run in `11-soccer-sweep-result.json`.

Note: the worker may suppress the streamed reasoning block when `supports_thinking` is on (think block accumulated server-side, only post-`</think>` content streamed). That's why `budget=None` has empty text — model never emitted `</think>` within max_tokens=4096.

## budget=None

- finish_reason: `length`
- elapsed: 162.1s
- prompt_tokens: 27512
- completion_tokens: 4096
- text length: 0 chars
- `</think>` present in text: False

_(empty text — model never closed the reasoning block before max_tokens)_

## budget=10

- finish_reason: `stop`
- elapsed: 37.3s
- prompt_tokens: 27512
- completion_tokens: 428
- text length: 1279 chars
- `</think>` present in text: True
- `</think>` at char index: 1145

```text


The video shows a soccer match between Liverpool and Manchester United. I need to find every instance of a foul.

1.  **00:15 - 00:21**: A Liverpool player (Sadio Mané) is dribbling the ball. A Manchester United player (Ashley Young) slides in and tackles him from behind, causing Mané to fall. The referee blows the whistle for a foul.
2.  **01:21 - 01:32**: A Manchester United player (Paul Pogba) is dribbling the ball. A Liverpool player (Jordan Henderson) slides in and tackles him, causing Pogba to fall. The referee blows the whistle for a foul.
3.  **01:40 - 01:49**: A replay of the previous foul is shown. Paul Pogba is tackled by Jordan Henderson.
4.  **04:33 - 04:44**: A Manchester United player (Marcus Rashford) is dribbling the ball. A Liverpool player (Joe Gomez) slides in and tackles him, causing Rashford to fall. The referee blows the whistle for a foul.

So, there are three distinct fouls shown in the video, with one being replayed. I will list the timestamps for each unique foul event.

-   **Foul 1**: 00:15 - 00:21
-   **Foul 2**: 01:21 - 01:32
-   **Foul 3**: 04:33 - 04:44

I will format the output as requested.
</think>

The following start and end timestamps mark a foul in the video:
- **00:15 - 00:21**
- **01:21 - 01:32**
- **04:33 - 04:44**
```

## budget=100

- finish_reason: `stop`
- elapsed: 36.5s
- prompt_tokens: 27512
- completion_tokens: 508
- text length: 1198 chars
- `</think>` present in text: True
- `</think>` at char index: 984

```text


1.  **00:15 - 00:21**: A foul occurs when a Manchester United player slides in and trips a Liverpool player who is dribbling down the right side of the field.
2.  **01:21 - 01:32**: A foul is committed when a Manchester United player slides in from behind and trips a Liverpool player in the midfield.
3.  **01:40 - 01:49**: A foul happens when a Manchester United player slides in and trips a Liverpool player near the sideline.
4.  **05:25 - 05:32**: A foul occurs when a Manchester United player slides in and trips a Liverpool player inside the penalty area, leading to a penalty decision.
5.  **07:15 - 07:21**: A foul is committed when a Manchester United player slides in and trips a Liverpool player near the sideline.
6.  **07:41 - 07:49**: A foul occurs when a Manchester United player slides in and trips a Liverpool player near the sideline.
7.  **09:06 - 09:12**: A foul happens when a Manchester United player slides in and trips a Liverpool player near the sideline.
</think>

The following start and end timestamps mark a foul in the video:
- **00:15 - 00:21**
- **01:21 - 01:32**
- **01:40 - 01:49**
- **05:25 - 05:32**
- **07:15 - 07:21**
- **07:41 - 07:49**
- **09:06 - 09:12**
```
