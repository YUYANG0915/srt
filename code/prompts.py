from typing import Any


DEFAULT_PROMPT = """Answer the visual question.

Return JSON with this schema:
{{
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


DIRECT_PROMPT = """Answer the visual question.

If the question cannot be answered definitively from the image, say what is missing and ask a concise clarification question.

Return JSON with this schema:
{{
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


EVIDENCE_PROMPT = """Look at the image carefully before answering.

Step 1: List all objects in the image that are relevant to the question. For each object, describe its visible attributes, approximate position, and why it may or may not be the target.

Step 2: Based only on the visual evidence you listed, decide whether the question can be answered definitively. If more than one relevant candidate remains, ask a concise clarification question instead of guessing.

Return JSON with this schema:
{{
  "relevant_candidates": [
    {{
      "name": string,
      "attributes": string,
      "position": string,
      "target_status": "possible" | "unlikely" | "target"
    }}
  ],
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


STRUCTURED_PRIOR_PROMPT = """Look at the image carefully before answering.

Follow this decision process strictly:

Step 1: Identify the noun phrase in the question that refers to a visual object or person.

Step 2: List all visually plausible referents for that noun phrase. For each candidate, describe its visible attributes, approximate position, and whether it is a possible referent.

Step 3: Decide whether the referent is visually unique.
- If exactly one plausible referent remains, answer the question about that referent.
- If multiple plausible referents remain, do not choose one by salience, centrality, or prominence. Ask a concise clarification question or enumerate the possible answers.
- If no plausible referent exists or the needed visual evidence is missing, answer with uncertain.

Step 4: Make sure the final policy follows Step 3.

Return JSON with this schema:
{{
  "referent_phrase": string,
  "relevant_candidates": [
    {{
      "name": string,
      "attributes": string,
      "position": string,
      "target_status": "possible" | "unlikely" | "target"
    }}
  ],
  "is_referent_unique": boolean,
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


VISUAL_SKETCHPAD_PROMPT = """You are using a Sketchpad-style visual reasoning process.

The image already contains a visual sketchpad artifact: all candidate objects for the question's referent phrase have been marked with colored bounding boxes and labels such as person_1, person_2, car_1, etc.

Follow the Sketchpad process:

THOUGHT 0: Identify the referent phrase in the question and decide what visual evidence is needed.
ACTION 0: Inspect every marked candidate box that could refer to that phrase.
OBSERVATION 0: For each marked candidate, record its label, visible attributes, position, and whether it is a plausible referent.
THOUGHT 1: Apply the referent-uniqueness prior:
- If exactly one marked candidate is a plausible referent, answer about that candidate.
- If multiple marked candidates are plausible referents, do not choose by salience, centrality, size, or prominence. Ask a concise clarification question or enumerate the possible answers.
- If no marked candidate is plausible or the needed visual evidence is missing, answer with uncertain.
ANSWER: The final policy must follow THOUGHT 1.

Return JSON with this schema:
{{
  "thought_0": string,
  "action_0": string,
  "observation_0": [
    {{
      "mark_id": string,
      "visible_attributes": string,
      "position": string,
      "is_plausible_referent": boolean
    }}
  ],
  "thought_1": string,
  "referent_phrase": string,
  "referent_uniqueness": "unique" | "ambiguous" | "missing",
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


MARKED_NO_PRIOR_PROMPT = """Answer the visual question using the image.

The image may contain colored bounding boxes and labels such as person_1, person_2, car_1, etc. These labels are visual aids that identify objects in the image. You may refer to them if useful.

Return JSON with this schema:
{{
  "noticed_marks": boolean,
  "mark_notes": string,
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


TASK3_EVIDENCE_PROMPT = """Look at the image carefully before answering.

This question may involve people participating in an activity. Some participants may be visible directly, while others may be implied by the camera viewpoint or first-person body/equipment cues.

Step 1: List the directly visible participants in the activity.
Step 2: List any viewpoint cues, such as first-person hands, arms, skis, snowboard, bicycle handlebars, surfboard, or a camera-wearer perspective.
Step 3: Based only on the visual evidence you listed, answer the question. If the image only supports an inference rather than a certain count, make that uncertainty explicit.

Return JSON with this schema:
{{
  "visible_participants": [
    {{
      "description": string,
      "activity": string,
      "position": string
    }}
  ],
  "viewpoint_cues": [
    {{
      "cue": string,
      "why_relevant": string
    }}
  ],
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


TASK3_MARKED_NO_PRIOR_PROMPT = """Answer the visual question using the image.

The image may contain colored bounding boxes and labels. These labels are visual aids that mark visible participants and possible viewpoint cues, but they do not by themselves tell you the answer.

Return JSON with this schema:
{{
  "noticed_marks": boolean,
  "mark_notes": string,
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


TASK3_VISUAL_SKETCHPAD_PROMPT = """You are using a Sketchpad-style visual reasoning process.

The image already contains a visual sketchpad artifact: visible participants and possible first-person viewpoint cues may be marked with colored boxes and labels.

Follow the process strictly:

THOUGHT 0: Identify the activity and the participant-count question.
ACTION 0: Inspect every marked visible participant and every marked viewpoint cue.
OBSERVATION 0: Record directly visible participants and any first-person cues, such as visible hands, arms, skis, snowboard, bicycle handlebars, surfboard, or camera-wearer perspective.
THOUGHT 1: Apply the camera-holder / viewpoint participation prior:
- If the scene has strong first-person participation cues and the question asks how many people are doing the activity, include the likely camera wearer as an implied participant, while stating that this is an inference.
- If the scene is a normal third-person view without first-person participation cues, count only directly visible participants.
- If the viewpoint cues are weak or incompatible with the activity, answer uncertain instead of inventing a hidden participant.
ANSWER: The final answer must follow THOUGHT 1.

Return JSON with this schema:
{{
  "thought_0": string,
  "action_0": string,
  "observation_0": {{
    "visible_participants": [
      {{
        "mark_id": string,
        "description": string,
        "activity_match": boolean
      }}
    ],
    "viewpoint_cues": [
      {{
        "mark_id": string,
        "cue": string,
        "supports_camera_holder_participation": boolean
      }}
    ]
  }},
  "thought_1": string,
  "hidden_camera_holder_likely": boolean,
  "estimated_total_participants": number,
  "policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "answer": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


PROCEDURAL_DEFAULT_PROMPT = """Answer the visual question.

Choose the most likely current step of the visible process.

Use one label from this set when possible:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return JSON with this schema:
{{
  "step_label": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


PROCEDURAL_EVIDENCE_PROMPT = """Look at the image carefully before answering.

Step 1: Identify the visible actor, tool, manipulated object, and any visible object-state cues.
Step 2: Based only on those visible cues, infer the most likely current step of the process.
Step 3: If more than one step remains plausible, choose the best-supported one and make the uncertainty explicit.

Use one label from this set when possible:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return JSON with this schema:
{{
  "actor": string,
  "tool": string,
  "object": string,
  "state_cues": [string],
  "step_label": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


PROCEDURAL_MARKED_NO_PRIOR_PROMPT = """Answer the visual question using the image.

The image may contain colored bounding boxes and labels that mark the actor, tool, object, or state-relevant regions. These labels are only visual aids and do not by themselves determine the answer.

Use one label from this set when possible:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return JSON with this schema:
{{
  "noticed_marks": boolean,
  "mark_notes": string,
  "step_label": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


PROCEDURAL_VISUAL_SKETCHPAD_PROMPT = """You are using a Sketchpad-style visual reasoning process.

The image may contain a visual sketchpad artifact with marked actor, tool, object, and state-relevant regions.

Follow the process strictly:

THOUGHT 0: Identify the visible activity domain and what kind of process step is being asked about.
ACTION 0: Inspect each marked actor, tool, object, and state cue.
OBSERVATION 0: Record what each marked region contributes to the process interpretation.
THOUGHT 1: Apply the SRT prior.
- Sequence: identify actor, tool, object, and current object state.
- Relation: check actor-tool, tool-object, and object-state compatibility.
- Timeline: choose the process step that best fits the visible stage and a plausible human procedure order.
- If two steps remain equally plausible, answer uncertain instead of over-claiming.

Use one label from this set when possible:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return JSON with this schema:
{{
  "thought_0": string,
  "action_0": string,
  "observation_0": {{
    "actor": string,
    "tool": string,
    "object": string,
    "state_cues": [string]
  }},
  "thought_1": string,
  "step_label": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


PROCEDURAL_SRT_GENERIC_PROMPT = """Use SRT-base to answer the visual question.

S: identify the visible actor, tool, object, and object-state cues.
R: check whether the actor-tool, tool-object, and object-state relations support the same process step.
T: decide which visible step best matches the current frame, not a hypothetical earlier or later step.

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_PROCESS_PROMPT = """Use SRT-base to answer the visual question.

Route the image through the most relevant process check:

1. Heat-state check:
- if the food is visibly on or in cooking equipment and the frame shows an in-progress heated state, prefer `heating_cooking`
- if the food only appears finished and ready to present, do not automatically choose `heating_cooking`

2. Assembly-vs-serving check:
- if ingredients are being combined, spread, or placed onto a base before final presentation, treat this as pre-serving preparation
- if the food is already plated, displayed, handed over, or clearly in a ready-to-serve state, prefer `plating_serving`

3. Tool-action check:
- visible cutting tool and cutting action -> `cutting`
- visible liquid transfer -> `pouring`
- visible stirring/combining in container -> `mixing`
- visible rinsing/cleaning food or utensils -> `washing`

Choose the nearest current stage shown in the frame, not a broader activity label.

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_PROCESS_V2_PROMPT = """Use SRT-base to answer the visual question.

General rule:
- extract only the minimal visible state needed for the current stage decision
- check only the stage-relevant cues
- choose the current visible stage, not a broader activity description

Route the frame through these boundary checks in order:

1. Heat-context boundary:
- choose `heating_cooking` if the food is still visibly on or in cooking equipment, or still in the immediate cooking context, and the frame most naturally reflects the cooking stage itself
- this includes cases where the food already looks mostly cooked but is still shown on the oven tray, grill, stovetop, or active cooking surface rather than clearly transferred to a serving context
- choose `plating_serving` only if the food is already presented, displayed, handed over, plated, or clearly positioned for serving rather than still belonging to the cooking setup

2. Assembly boundary:
- choose `mixing` if ingredients are visibly being combined, spread, stirred, or assembled before final cooking or before a final serving-ready presentation
- choose `plating_serving` if the frame is better described as presenting, displaying, or handing over a prepared item rather than actively combining ingredients

3. Explicit action boundary:
- visible slicing/chopping/dividing action -> `cutting`
- visible liquid transfer -> `pouring`
- visible washing/rinsing/cleaning with water or soap -> `washing`

4. Tie-break rule:
- if a frame lies between two stages, choose the nearer stage indicated by the visible physical context
- prefer `heating_cooking` over `plating_serving` when the food is still on the cooking device or tray inside the cooking scene
- prefer `plating_serving` over `mixing` when the food is already presented to a viewer/customer/diner or arranged as a ready item

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_PROCESS_V3_PROMPT = """Use SRT-base to answer the visual question.

General rule:
- identify the current visible stage of the whole scene, not just the smallest local hand motion
- prefer the stage that best describes the visible outcome of the frame

Route the frame through these checks in order:

1. Serving / presentation-first check:
- choose `plating_serving` if the scene is mainly showing a finished or nearly finished food item being presented, displayed, plated, arranged for serving, handed to someone, or shown in a diner/customer/table context
- this includes celebration dessert scenes, table presentation, buffet/display layout, plated items, or completed food arranged as the visible outcome
- when a local spoon/hand/topping action appears inside an otherwise presentation-centered scene, keep `plating_serving` if the whole frame is better described as serving/presentation than construction

2. Heat-context boundary:
- choose `heating_cooking` if the food is still visibly on or in cooking equipment, or still belongs to the immediate cooking setup, and the frame most naturally reflects the cooking stage itself
- choose `plating_serving` instead if the item is detached from active cooking and is now being presented or displayed as a result

3. Assembly boundary:
- choose `mixing` if ingredients are visibly being combined, spread, stirred, or assembled before final cooking or before a clearly ready-to-serve presentation
- choose `plating_serving` if the frame is better described as presenting the completed item rather than constructing it

4. Explicit action boundary:
- visible slicing/chopping/dividing action -> `cutting`
- visible liquid transfer where the whole frame is best described by the transfer itself -> `pouring`
- visible washing/rinsing/cleaning with water or soap -> `washing`

5. Tie-break rule:
- if a frame lies between two stages, choose the nearer stage indicated by the visible physical context
- prefer the whole-scene stage over a small local motion when the scene already shows a finished outcome

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_ACTION_PRIORITY_PROMPT = """Use SRT-base with action-priority process rules.

Goal:
- decide the current visible process step
- do not let the general meaning "finished food / food for eating" override a visible action

Apply these checks in order:

1. Explicit tool-action priority:
- if a knife, cutter, or cutting utensil is visibly contacting food and dividing/slicing/chopping/carving it, choose `cutting`
- if a stirring utensil, hand, or tool is visibly combining ingredients inside a bowl/container or pile, choose `mixing`
- if liquid is visibly moving from a source into a target, choose `pouring`
- if water/soap is visibly cleaning food, hands, dishes, or tools, choose `washing`

2. Heat-state boundary:
- if food is visibly on/in a grill, oven, stove, pan, or active cooking surface and the current frame belongs to the cooking setup, choose `heating_cooking`
- do not choose `heating_cooking` only because the food must have been cooked earlier

3. Serving / presentation boundary:
- choose `plating_serving` when there is no stronger explicit action and the scene shows finished food being plated, displayed, served, held for eating, sold, or presented as a ready result

Tie-break rules:
- direct physical contact between tool and food beats broad scene context
- a cake/pizza/hot dog can still be `cutting` if the visible current action is slicing/dividing it
- a ready-looking food item is `plating_serving` only when no explicit action is currently changing/dividing/combining it

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_AUTO_ROUTER_ACTION_PROMPT = """Use automatic routed SRT to answer the visual question.

First choose the process-boundary family that controls the current decision:

1. `active_heat_vs_ready_result`
- use this when the main decision is whether food is still in an active cooking/heating setup or already a ready result
- cues: grill, oven, stove, pan, cooking tray, active heat context

2. `assembly_vs_serving`
- use this when the main decision is whether food is still being assembled/prepared for presentation or already served/displayed/ready-to-eat
- cues: plating, display counter, table, handed food, buffet, ready dessert, customer/diner-facing presentation

3. `explicit_action`
- use this when a direct visible action should override broad scene context
- cues: knife/cutter contacting food, visible slicing/dividing/chopping/carving, stirring/mixing in a container, liquid transfer, washing/rinsing

Then apply only the rule for the chosen family:

For `active_heat_vs_ready_result`:
- choose `heating_cooking` if food is visibly on/in a grill, oven, stove, pan, or active cooking surface
- choose `plating_serving` if food is detached from cooking equipment and shown as a ready/presented result

For `assembly_vs_serving`:
- choose `plating_serving` if the scene mainly shows finished or nearly finished food being plated, displayed, served, held for eating, sold, or presented as a ready result
- choose `mixing` only if the main visible event is ingredient combination, spreading, stirring, or assembly before presentation

For `explicit_action`:
- knife/cutter visibly contacting and dividing/slicing/chopping/carving food -> `cutting`
- tool/hand visibly combining or stirring ingredients inside a bowl/container/pile -> `mixing`
- visible liquid transfer from source to target -> `pouring`
- visible water/soap cleaning food, hands, dishes, or tools -> `washing`

Important priority:
- if the chosen family is `explicit_action`, direct tool-food contact beats broad serving/presentation context
- if there is no clear explicit action, do not force `cutting` or `mixing`; use the heat or serving boundary instead
- answer the current visible stage, not a hidden before/after story

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_BOUNDARY_CLASSIFIER_PROMPT = """Classify the process-boundary family for this cooking/food scene.

Choose exactly one:
- `explicit_action`
- `active_heat_vs_ready_result`
- `assembly_vs_serving`

Use this priority order:

1. Choose `explicit_action` if a direct visible action determines the current step:
- knife/cutter visibly contacting food
- food visibly being sliced, chopped, divided, carved, stirred, mixed, poured, washed, or rinsed
- tool-food contact is more important than the fact that the food may also look ready to eat

2. Else choose `active_heat_vs_ready_result` if the main boundary is active cooking/heating vs finished result:
- grill, oven, stove, pan, cooking tray, or active cooking surface is visually central

3. Else choose `assembly_vs_serving`:
- food is being plated, displayed, served, held for eating, sold, presented, or appears ready-to-eat
- no stronger explicit action or active heat cue controls the decision

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_ROUTER_PROMPT = """Use SRT-base to answer the visual question.

First identify which boundary family best describes the current frame.

Choose one boundary family:
- `active_heat_vs_ready_result`
- `assembly_vs_serving`
- `explicit_action`
- `other`

Family definitions:

1. `active_heat_vs_ready_result`
- use this when the main uncertainty is whether the food is still in the cooking stage or already in a ready-to-serve result state
- examples: food on oven tray, grill, stovetop, hot plate, cooking pan

2. `assembly_vs_serving`
- use this when the main uncertainty is whether ingredients are still being assembled/spread/combined or whether the item is already presented as a serving/display result
- examples: topping placement, sauce spreading, buffet/display arrangement, plated dessert

3. `explicit_action`
- use this when a direct visible action determines the answer
- cutting, pouring, washing, stirring/mixing

Then apply only the matching short rule:

For `active_heat_vs_ready_result`:
- choose `heating_cooking` if the food is still visibly on or in cooking equipment or still belongs to the active cooking setup
- choose `plating_serving` only if the food is already presented, plated, displayed to a diner/customer, or clearly detached from the cooking setup

For `assembly_vs_serving`:
- choose `mixing` if the frame shows ingredients being combined, spread, or assembled before final presentation
- choose `plating_serving` if the frame is better described as display, presentation, plating, buffet arrangement, handoff, or ready-to-serve result

For `explicit_action`:
- slicing/chopping/dividing -> `cutting`
- liquid transfer -> `pouring`
- rinsing/cleaning with water -> `washing`
- stirring/combining in container -> `mixing`

For `other`:
- choose the nearest visible current stage using only the minimal state needed for the decision

Final rule:
- answer the current visible stage only
- do not replace it with the broader activity name
- if the frame is between two stages, choose the nearer visible stage

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_ROUTER_V2_PROMPT = """Use SRT-base to answer the visual question.

First identify which boundary family best describes the current frame.

Choose one boundary family:
- `active_heat_vs_ready_result`
- `assembly_vs_serving`
- `explicit_action`
- `other`

Family definitions:

1. `active_heat_vs_ready_result`
- use this when the main uncertainty is whether the food is still in the cooking stage or already in a ready-to-serve result state
- examples: food on oven tray, grill, stovetop, hot plate, cooking pan

2. `assembly_vs_serving`
- use this when the main uncertainty is whether ingredients are still being assembled/spread/combined or whether the item is already presented as a serving/display result
- examples: topping placement, sauce spreading, buffet/display arrangement, plated dessert

3. `explicit_action`
- use this when a direct visible action determines the answer
- cutting, pouring, washing, stirring/mixing

Then apply only the matching short rule:

For `active_heat_vs_ready_result`:
- choose `heating_cooking` if the food is still visibly on or in cooking equipment or still belongs to the active cooking setup
- choose `plating_serving` only if the food is already presented, plated, displayed to a diner/customer, or clearly detached from the cooking setup

For `assembly_vs_serving`:
- choose `plating_serving` if the frame is best described as a completed food item being presented, displayed, plated, arranged for pickup, handed over, or shown as a ready-to-serve result
- cues for `plating_serving` include a whole or nearly finished item, diner/customer-facing presentation, serving tray or buffet layout, table/counter display, or a scene whose main function is presentation rather than construction
- choose `mixing` only if the main visible event is still ingredient combination, spreading, topping placement, or assembly before the item has reached a presentation-ready state
- if both are present, prefer `plating_serving` when the visible scene centers on the finished item as an outcome, not the local hand motion

For `explicit_action`:
- slicing/chopping/dividing -> `cutting`
- liquid transfer -> `pouring`
- rinsing/cleaning with water -> `washing`
- stirring/combining in container -> `mixing`

For `other`:
- choose the nearest visible current stage using only the minimal state needed for the decision

Final rule:
- answer the current visible stage only
- do not replace it with the broader activity name
- if the frame is between two stages, choose the nearer visible stage

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_ROUTER_ORACLE_PROMPT = """Use SRT-base to answer the visual question.

The boundary family has already been identified for you:
- {family_hint}

Use that family directly instead of re-classifying the scene.

Family definitions:

1. `active_heat_vs_ready_result`
- the main uncertainty is whether the food is still in the cooking stage or already in a ready-to-serve result state

2. `assembly_vs_serving`
- the main uncertainty is whether ingredients are still being assembled/spread/combined or whether the item is already presented as a serving/display result

3. `explicit_action`
- a direct visible action determines the answer

Then apply only the matching short rule:

For `active_heat_vs_ready_result`:
- choose `heating_cooking` if the food is still visibly on or in cooking equipment or still belongs to the active cooking setup
- choose `plating_serving` only if the food is already presented, plated, displayed to a diner/customer, or clearly detached from the cooking setup

For `assembly_vs_serving`:
- choose `mixing` if ingredients are visibly being combined, spread, or assembled before final presentation
- choose `plating_serving` if the frame is better described as display, presentation, plating, buffet arrangement, handoff, or ready-to-serve result

For `explicit_action`:
- slicing/chopping/dividing -> `cutting`
- liquid transfer -> `pouring`
- rinsing/cleaning with water -> `washing`
- stirring/combining in container -> `mixing`

Final rule:
- answer the current visible stage only
- do not replace it with the broader activity name
- if the frame is between two stages, choose the nearer visible stage

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_TARGETED_ROUTER_ORACLE_V2_PROMPT = """Use SRT-base to answer the visual question.

The boundary family has already been identified for you:
- {family_hint}

Use that family directly instead of re-classifying the scene.

Family definitions:

1. `active_heat_vs_ready_result`
- the main uncertainty is whether the food is still in the cooking stage or already in a ready-to-serve result state

2. `assembly_vs_serving`
- the main uncertainty is whether ingredients are still being assembled/spread/combined or whether the item is already presented as a serving/display result

3. `explicit_action`
- a direct visible action determines the answer

Then apply only the matching short rule:

For `active_heat_vs_ready_result`:
- choose `heating_cooking` if the food is still visibly on or in cooking equipment or still belongs to the active cooking setup
- choose `plating_serving` only if the food is already presented, plated, displayed to a diner/customer, or clearly detached from the cooking setup

For `assembly_vs_serving`:
- choose `plating_serving` if the frame is best described as a completed food item being presented, displayed, plated, arranged for pickup, handed over, or shown as a ready-to-serve result
- cues for `plating_serving` include a whole or nearly finished item, diner/customer-facing presentation, serving tray or buffet layout, table/counter display, or a scene whose main function is presentation rather than construction
- choose `mixing` only if the main visible event is still ingredient combination, spreading, topping placement, or assembly before the item has reached a presentation-ready state
- if both are present, prefer `plating_serving` when the visible scene centers on the finished item as an outcome, not the local hand motion

For `explicit_action`:
- slicing/chopping/dividing -> `cutting`
- liquid transfer -> `pouring`
- rinsing/cleaning with water -> `washing`
- stirring/combining in container -> `mixing`

Final rule:
- answer the current visible stage only
- do not replace it with the broader activity name
- if the frame is between two stages, choose the nearer visible stage

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_GPT_VERIFY_PROMPT = """Use a lightweight verification-style SRT process.

Verify the following in order:

1. Is the frame showing active heat application or food still on a cooking surface?
2. Is the frame showing a completed, ready-to-serve result instead?
3. Is the frame showing pre-cooking assembly, such as adding or spreading ingredients onto a base?
4. Is there explicit evidence for cutting, pouring, mixing, or washing?

Decision rules:
- prefer the most directly visible current step
- do not infer a hidden earlier step if the frame already shows a later one
- do not infer a hidden later step if the frame still shows active cooking
- if the image is between two stages, choose the nearer visible stage

Use one label from this set:
- washing
- cutting
- mixing
- pouring
- heating_cooking
- plating_serving

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_SEMANTIC_BOUNDARY_PROMPT = """Use SRT-base with explicit semantic boundary definitions.

Label meanings:
- `washing`: cleaning ingredients, hands, dishes, or tools with water/soap
- `cutting`: slicing, chopping, carving, or visibly dividing food with a cutting action
- `mixing`: stirring, combining, spreading, or assembling ingredients before final cooking or final serving
- `pouring`: visibly transferring a liquid or flowable substance from one container/source to another target
- `heating_cooking`: food is visibly being cooked, heated, grilled, baked, fried, or managed on active cooking equipment
- `plating_serving`: food is already plated, displayed, handed over, or in a ready-to-serve presentation state

Critical boundary rules:
- ingredient placement onto raw dough or a base before cooking is `mixing`, not `plating_serving`
- food resting as a completed cooked result ready to present is `plating_serving`, not `heating_cooking`
- food still actively on the grill, stove, or oven as the current visible process is `heating_cooking`

Return exactly one-line JSON and nothing else:
{{"step_label":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


PROCEDURAL_SRT_OVERLOAD_PROMPT = """Use a detailed multi-step SRT reasoning process.

Step 1: Identify every visible human actor or partial actor.
Step 2: Identify every possible tool, utensil, container, tray, cooking surface, and serving surface.
Step 3: Identify every visible food object and every ingredient-like component.
Step 4: Describe the current physical state of each food object.
Step 5: Enumerate all plausible process steps among washing, cutting, mixing, pouring, heating_cooking, plating_serving.
Step 6: For each plausible step, argue for and against it.
Step 7: Decide which step best matches the visible frame itself.

Important:
- avoid collapsing distinct stages
- prefer the visible current step over broader task descriptions
- if the frame is ambiguous, still choose the best-supported label

Return JSON with this schema:
{{
  "actors": [string],
  "tools": [string],
  "objects": [string],
  "state_cues": [string],
  "candidate_steps": [string],
  "step_label": string,
  "confidence": number,
  "rationale": string
}}

Question: {question}
"""


SRT_SPATIAL_DEFAULT_PROMPT = """Judge whether the spatial statement is true in the image.

Return JSON with this schema:
{{
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_EVIDENCE_PROMPT = """Look at the image carefully before judging the spatial statement.

Step 1: Identify the subject object, reference object, and spatial relation in the statement.
Step 2: Describe the visual evidence relevant to that relation.
Step 3: Judge whether the statement is true, false, or uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "reference": string,
  "relation": string,
  "visual_evidence": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_PROMPT = """You are using an SRT spatial reasoning process.

SRT means Sequence -> Relation -> Timeline/consistency.

S: Sequence
1. Identify the subject object in the statement.
2. Identify the reference object.
3. Identify the queried spatial relation.
4. Choose the required spatial frame:
   - image_plane for left/right/above/below
   - depth for in front of/behind
   - orientation for facing/looking/toward
   - containment for inside/in/contains/consists of
   - contact_support for on/under/touching/attached to
   - uncertain if the relation or objects are unclear

R: Relation
Inspect the relation between the subject and reference object using the selected frame. Do not use the wrong frame. For example, do not answer an in-front-of relation using only image-plane vertical position.

T: Timeline / consistency
Before finalizing, check whether your final true/false answer follows the Sequence and Relation steps. If the needed visual evidence is missing or the relation cannot be judged, answer uncertain.

Return JSON with this schema:
{{
  "sequence": {{
    "subject": string,
    "reference": string,
    "relation": string,
    "spatial_frame": "image_plane" | "depth" | "orientation" | "containment" | "contact_support" | "uncertain"
  }},
  "relation": {{
    "subject_visible": boolean,
    "reference_visible": boolean,
    "visual_evidence": string,
    "relation_holds": boolean
  }},
  "timeline_check": {{
    "is_consistent": boolean,
    "explanation": string
  }},
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_V2_PROMPT = """You are using an SRT spatial reasoning process.

SRT means Sequence -> Relation -> Timeline/consistency.

Important rule: this process is a visual verification procedure, not a taxonomy quiz.
The final answer should be based on the directly visible relation between the subject and the reference object. Do not change a visually clear true/false judgement to uncertain only because the relation category is hard to name.

S: Sequence
1. Parse the statement into subject, reference object, and queried relation.
2. Choose the main visual test needed for that relation:
   - image_plane: compare 2D image positions such as left/right/above/below.
   - depth: use occlusion, perspective, scene layout, and contextual depth cues for in front of/behind/far away.
   - orientation: inspect visible facing direction or body/object orientation.
   - containment: check whether the subject lies within the visible boundary or interior of the reference.
   - contact_support: check touching, support, attachment, or surface relation.
   - proximity: check near/next to/beside/adjacent relations without requiring physical contact.
   - mixed: use more than one cue when the relation naturally needs multiple cues.
   - uncertain: use only if the objects or relation cannot be identified.

R: Relation
3. State the visual evidence that supports the relation and the evidence that argues against it.
4. Decide relation_holds:
   - true if the visual evidence clearly supports the statement.
   - false if the visual evidence clearly contradicts the statement.
   - null only if one object is missing, heavily occluded, or the relation genuinely cannot be judged.

T: Timeline / consistency
5. The final answer must follow relation_holds:
   - relation_holds true -> answer "true"
   - relation_holds false -> answer "false"
   - relation_holds null -> answer "uncertain"

Return JSON with this schema:
{{
  "sequence": {{
    "subject": string,
    "reference": string,
    "relation": string,
    "spatial_frame": "image_plane" | "depth" | "orientation" | "containment" | "contact_support" | "proximity" | "mixed" | "uncertain",
    "visual_test": string
  }},
  "relation": {{
    "subject_visible": boolean,
    "reference_visible": boolean,
    "evidence_for": string,
    "evidence_against": string,
    "relation_holds": boolean | null
  }},
  "timeline_check": {{
    "is_consistent": boolean,
    "explanation": string
  }},
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_LITE_PROMPT = """You are using an SRT spatial reasoning process.

SRT means Sequence -> Relation -> Timeline/consistency.
Use it as a short visual verification procedure. Do not overthink a clearly visible relation.

S: Sequence
1. Parse the statement into subject, reference object, and queried spatial relation.
2. Write the concrete visual test for this relation in one sentence. The test should describe what you need to check in the image, not an abstract category label.

R: Relation
3. Inspect the image and describe:
   - evidence_for: visible cues supporting the statement
   - evidence_against: visible cues contradicting the statement
4. Decide relation_holds:
   - true if evidence_for is clearly stronger
   - false if evidence_against is clearly stronger
   - null only if the objects are missing, too occluded, or genuinely impossible to judge

T: Timeline / consistency
5. The final answer must follow relation_holds:
   - true -> "true"
   - false -> "false"
   - null -> "uncertain"

Return JSON with this schema:
{{
  "sequence": {{
    "subject": string,
    "reference": string,
    "relation": string,
    "visual_test": string
  }},
  "relation": {{
    "subject_visible": boolean,
    "reference_visible": boolean,
    "evidence_for": string,
    "evidence_against": string,
    "relation_holds": boolean | null
  }},
  "timeline_check": {{
    "is_consistent": boolean,
    "explanation": string
  }},
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_MINIMAL_PROMPT = """Look at the image carefully before judging the spatial statement.

Use a minimal SRT process:

S: Sequence
Identify the subject, reference object, and spatial relation in the statement.

R: Relation
Describe only the visual evidence needed to judge this relation. Mention both supporting and contradicting cues if they exist.

T: Timeline / consistency
Before finalizing, check whether the answer is directly supported by the visual evidence you just wrote. If the evidence supports true, answer true. If it supports false, answer false. Use uncertain only when the image genuinely does not show enough evidence.

Return JSON with this schema:
{{
  "sequence": {{
    "subject": string,
    "reference": string,
    "relation": string
  }},
  "relation_evidence": {{
    "visual_evidence": string,
    "supports": "true" | "false" | "uncertain"
  }},
  "timeline_check": {{
    "is_consistent": boolean,
    "explanation": string
  }},
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_VERIFY_PROMPT = """Look at the image carefully before judging the spatial statement.

Use SRT as a lightweight verification process:

S: Sequence
Identify the subject object, reference object, and spatial relation in the statement.

R: Relation
Describe the visual evidence relevant to that relation. Focus on what is visible in the image.

T: Timeline / consistency
Make the final answer directly follow the visual evidence. Do not add extra assumptions. Use uncertain only when the image genuinely does not provide enough evidence.

Return JSON with this schema:
{{
  "subject": string,
  "reference": string,
  "relation": string,
  "visual_evidence": string,
  "consistency_note": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_ADAPTIVE_PROMPT = """Look at the image carefully before judging the spatial statement.

Use relation-adaptive SRT: Sequence -> Relation -> Timeline/consistency.
The goal is not to produce a long chain. The goal is to apply the right visual check for the queried relation.

S: Sequence
Identify the subject object, reference object, and spatial relation in the statement.

R: Relation
Use the relation-specific check below:
- left/right/left side/right side/above/below: compare 2D image positions of the subject and reference.
- next to/beside/by/adjacent/alongside/at the side of: check whether the objects are in the same local region and close enough; physical contact is not required.
- on/on top of/under/beneath/touching/attached/covering: check visible contact, support, overlap, or surface relation.
- in/inside/within/into/enclosed by/contains/surrounding/part of/consists of: check containment, boundary, interior, or part-whole relation.
- in front of/behind/back/far away/away from: check depth cues such as occlusion, scale, ground plane, perspective, and scene layout.
- facing/facing away from/parallel/perpendicular: check visible body, head, front-facing side, or object orientation. If orientation is not visible enough, answer uncertain instead of guessing.
- across from/opposite to: check whether the subject and reference face or occupy opposing sides of a shared space.

Then write one concise visual_evidence sentence saying which cues support or contradict the statement.

T: Timeline / consistency
Give the final answer directly from the relation-specific evidence.
Use "uncertain" only if the needed objects or relation cues are genuinely not visible.

Return JSON with this schema:
{{
  "subject": string,
  "reference": string,
  "relation": string,
  "relation_family": "image_position" | "proximity" | "contact_support" | "containment" | "depth" | "orientation" | "opposition" | "uncertain",
  "visual_evidence": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_CONTACT_MICRO_PROMPT = """Look at the image carefully before judging the spatial statement.

Use a contact/support micro-prior. These relations are easy to confuse if you rely only on coarse proximity.

S: Sequence
Identify the subject object, the reference object, and the exact relation phrase.

R: Relation
Use the relation-specific visual check that matches the phrase:
- on / on top of: check whether the subject is resting on the upper surface of the reference, with visible support or stable placement.
- under / beneath: check whether the subject is lower than the reference and visually under it; do not require contact unless the image shows it.
- touching: check direct physical contact at the visible boundary.
- attached to: check direct attachment, connection, or fastening.
- covering: check whether the subject lies over or obscures the surface of the reference.
- at the edge of: check whether the subject is located near the visible boundary or rim of the reference.
- beside / next to / by / adjacent / alongside: these are only local-neighborhood checks. Do not upgrade proximity into contact or support.

Then write:
1. the subtype test you used,
2. one concise evidence_for sentence,
3. one concise evidence_against sentence.

T: Timeline / consistency
Answer "true" only if the visual evidence matches the exact subtype test.
Answer "false" if the visible evidence contradicts that subtype.
Answer "uncertain" only if the needed contact/support cues are genuinely not visible.

Return JSON with this schema:
{{
  "subject": string,
  "reference": string,
  "relation": string,
  "relation_subtype": "support_on" | "lower_under" | "contact_touching" | "attachment" | "covering" | "edge_boundary" | "local_proximity" | "uncertain",
  "evidence_for": string,
  "evidence_against": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_VSR_RULES_PROMPT = """Judge whether the spatial statement is true in the image.

Use a VSR-calibrated SRT process. VSR statements should be judged by visible spatial geometry, not by whether the sentence sounds physically typical or commonsense-plausible.

S: Sequence
1. Parse the statement into subject, relation phrase, and reference object.
2. Find visible instances of the subject and reference. If there are multiple instances, judge whether at least one subject-reference pair satisfies the statement.
3. Select the exact visual test for the relation phrase.

R: Relation
Use these relation rules:
- X under/beneath Y: X is visually lower than Y, or Y rests on / is supported by X. This can be true even when X is much larger than Y.
- X on/on top of Y: X rests on Y's visible upper surface or is supported by Y.
- X above/below Y: compare vertical image position, not physical plausibility.
- X behind/in front of Y: use depth cues, occlusion, ground plane, scale, and scene layout. Do not reduce this to image up/down.
- X inside/in/within/into/enclosed by Y: X is within the visible boundary, opening, or interior region of Y.
- X contains/surrounding/has as part/consists of Y: Y is inside, enclosed by, or part of X. Partial visibility can still support this if the boundary relation is clear.
- X at the edge/side of Y: X is near Y's visible boundary, rim, side, or outer region.
- X next to/beside/by/adjacent/alongside/near/close to Y: X and Y are in the same local region with small spatial separation; contact is not required.
- X facing/facing away from Y: inspect visible front/back direction, head/body orientation, gaze, or object-facing side.
- X opposite/across from Y: inspect opposing sides or facing positions across a shared space.

T: Timeline / consistency
Before answering, check for these failure modes:
- Do not answer false merely because the relation is unusual in the real world.
- Do not swap subject and reference.
- Do not use 2D left/right/up/down when the relation requires depth or containment.
- If visual evidence is enough for a true/false judgment, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "visual_test": string,
  "evidence_for": string,
  "evidence_against": string,
  "subject_reference_pair_found": boolean,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_ORIENTATION_POSE_PROMPT = """Judge whether the orientation statement is true in the image.

Use an orientation-specific SRT process. Orientation relations should be judged from visible pose, front/back side, head/body direction, gaze, or object-facing side. Do not require the two objects to be interacting.

S: Sequence
1. Parse the statement into subject, orientation relation, and reference object.
2. Locate the subject and reference.
3. Identify the subject's visible front/back/side direction before judging the relation.

R: Relation
Use these rules:
- X facing Y: X's front, head, gaze, or functional front side points toward Y.
- X facing away from Y: X's front/head/gaze points away from Y, or the visible back of X is oriented toward Y.
- X parallel to Y: X and Y have similar visible orientation axes.
- X perpendicular to Y: X and Y have visible orientation axes near a right angle.

Evidence rules:
- For people or animals, use head/body direction and visible back/front cues.
- For vehicles or objects, use the object's functional front/back side when visible.
- If the subject's orientation is not visible, answer uncertain.
- Do not answer false just because the subject is not actively looking at or interacting with the reference.

T: Timeline / consistency
Make the answer follow the orientation evidence only. Do not substitute proximity, left/right position, or physical contact for orientation.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "subject_orientation_cues": string,
  "reference_position_relative_to_subject": string,
  "evidence_for": string,
  "evidence_against": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_PROXIMITY_STRICT_PROMPT = """Judge whether the proximity statement is true in the image.

Use a strict proximity SRT process. Proximity relations should not be treated as true merely because both objects appear in the same image.

S: Sequence
1. Parse the statement into subject, proximity relation, and reference object.
2. Locate the closest visible subject-reference pair.
3. Identify whether they share a local region.

R: Relation
Use these rules:
- X by / next to / beside / adjacent to / alongside Y: X and Y should be spatially close, with only a small gap or direct adjacency in the local scene.
- For by / next to / beside, side-by-side adjacency is required. If X is on top of Y, inside Y, supported by Y, or vertically stacked with Y, do not count that as by/next to/beside unless there is also clear side-by-side placement.
- X at the side of Y: X should be near a visible side boundary of Y, not just somewhere elsewhere in the image.
- X near / close to Y: X and Y should be close relative to object size and scene scale.
- If another object, large gap, or separate region clearly separates X and Y, answer false.
- Do not answer true just because X and Y are both salient or both visible.

T: Timeline / consistency
Answer true only when local-neighborhood evidence is clear. Answer false when the objects are separated or merely co-present.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "closest_pair_description": string,
  "local_region_evidence": string,
  "separation_evidence": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_ENCLOSURE_OPEN_PROMPT = """Judge whether the containment statement is true in the image.

Use an enclosure-focused SRT process. In VSR, surrounding/contains-style statements are judged by visible enclosure geometry, not by whether the container is fully sealed.

S: Sequence
1. Parse the statement into subject, containment relation, and reference object.
2. Locate the subject and reference.
3. Judge whether the reference lies inside, within the opening of, or is visibly enclosed by the subject's boundary region.

R: Relation
Use these rules:
- X surrounding Y: true if Y is visibly enclosed, wrapped by, or placed within X's opening/boundary region, even if X is open on one side.
- X contains / has as a part Y: true if Y is inside X, partly enclosed by X, or clearly included as a visible part of X.
- X inside / in / within Y: true if X lies inside Y's interior region or within Y's visible opening.
- Do not require full 360-degree closure.
- Do not answer false just because much of Y remains visible.
- Answer false when Y is merely on top of X, beside X, or touching X without being in its interior/opening/boundary region.

T: Timeline / consistency
Base the answer on visible enclosure only.
If the opening/interior relation is clear, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "enclosure_region": string,
  "evidence_for": string,
  "evidence_against": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_ABOVE_SUPPORT_PROMPT = """Judge whether the vertical spatial statement is true in the image.

Use a vertical-geometry SRT process. In VSR, above/below are judged by visible relative height in the image, and support/contact does not cancel the relation.

S: Sequence
1. Parse the statement into subject, vertical relation, and reference object.
2. Locate the clearest subject-reference pair.
3. Compare their vertical placement in the image.

R: Relation
Use these rules:
- X above Y: true if the main mass or clearly relevant part of X is visually higher than Y in the image.
- X below Y: true if X is visually lower than Y in the image.
- If X rests on Y or touches Y from above, that can still count as above.
- If Y supports X from below, that does not make the statement false.
- Judge vertical relation directly from image height; do not reject it just because the two objects are in contact or because the sentence sounds unusual.
- Answer false only when X is not actually higher than Y, or when the opposite vertical order is clearer.

T: Timeline / consistency
Use vertical image geometry as the deciding cue.
If the higher/lower relation is clear, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "vertical_evidence": string,
  "contact_or_support_note": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_OPPOSITION_SHARED_SPACE_PROMPT = """Judge whether the opposition statement is true in the image.

Use a shared-space opposition SRT process. In VSR, opposite/across-from is often about occupying opposing sides of the same visible space, lane, curb, counter, or open region.

S: Sequence
1. Parse the statement into subject, opposition relation, and reference object.
2. Locate the subject and reference.
3. Identify the shared space between them, if any, such as a road, lane, walkway, counter gap, or open region.

R: Relation
Use these rules:
- X opposite to / across from Y: true if X and Y occupy opposing sides of the same visible space or gap.
- They do not need to physically face each other like humans.
- Large scale differences are allowed.
- A parking meter and a car can be opposite if they are on opposite sides of the street/parking space/open lane.
- Answer false when they are merely near each other on the same side, or when no shared dividing space is visible.

T: Timeline / consistency
Judge opposition from scene layout and side-of-space structure, not from object category expectations.
If the shared-space opposition is clear, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "shared_space": string,
  "subject_side": string,
  "reference_side": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_FACING_AWAY_BEARING_PROMPT = """Judge whether the facing-away statement is true in the image.

Use a bearing-based SRT process. "Facing away from" is determined by the subject's forward direction relative to the reference, not by simple left/right position.

S: Sequence
1. Parse the statement into subject, relation, and reference.
2. Locate the subject and reference.
3. Determine the subject's forward-facing direction from face, eyes, nose, chest, or functional front side.
4. Determine where the reference lies relative to that forward direction: in front, behind, or around the subject.

R: Relation
Use these rules:
- X facing away from Y: true when X's forward-facing side points away from Y, so Y lies mostly behind X or on X's back side.
- If X is looking into Y or toward Y, answer false.
- If Y surrounds or contains X, check the local part of Y that X is oriented toward. If X's head/front points into Y, answer false.
- Use head/face direction first for animals and people; use body/chest direction as backup.
- Do not answer false just because X is not moving away. Only the visible facing direction matters.

T: Timeline / consistency
Base the answer on orientation plus reference-relative bearing.
If the subject's front direction and the reference location are both visible, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "subject_forward_direction": string,
  "reference_relative_to_subject": string,
  "bearing_judgment": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_BEHIND_OCCLUSION_PROMPT = """Judge whether the depth statement is true in the image.

Use an occlusion-first SRT process for "behind". Do not reduce behind to up/down or left/right position in the image.

S: Sequence
1. Parse the statement into subject, relation, and reference.
2. Locate the clearest subject-reference pair.
3. First ask whether one object is visually in front of the other by occluding it, overlapping it, or occupying the foreground.
4. If occlusion is weak, use depth cues such as scale, focus, cropping, road/ground-plane layout, and perspective.

R: Relation
Use these rules:
- X behind Y: true if Y is visually in front of X, or X is farther away in scene depth than Y.
- Partial visibility still counts: if only part of X is visible because Y or the foreground hides it, X can still be behind Y.
- A large subject that appears only at the image edge or in the background can still be behind.
- If X is carrying Y, containing Y, or simply adjacent to Y on the same surface, do not call that behind.
- Answer false when X is clearly in front of Y, or when the objects appear on the same depth plane without strong evidence that X is farther back.

T: Timeline / consistency
Prefer occlusion and foreground-background evidence over commonsense expectations.
If front/back order is clear, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "occlusion_order": string,
  "depth_cues": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_ON_TOP_SURFACE_PROMPT = """Judge whether the support statement is true in the image.

Use an upper-surface SRT process for "on top of". The key question is whether the subject is resting on the reference's upper supporting surface or top rim, not merely touching or overlapping it.

S: Sequence
1. Parse the statement into subject, relation, and reference.
2. Locate the clearest subject-reference pair.
3. Identify the reference's upper surface, top panel, top rim, or top supporting boundary.

R: Relation
Use these rules:
- X on top of Y: true if X is resting on Y's upper surface or top rim, with visible support from below.
- Partial support still counts if a clear part of X rests on Y's top surface or top boundary.
- Broad surfaces such as beds, sink edges, toilet lids, suitcase tops, car roofs, or microwave tops count as valid top support surfaces.
- Answer false if X is on Y's side, front face, vertical wall, or merely inside/opening-adjacent without top support.
- Answer false if X is only near Y, overlapping Y in the image, or supported by something else.
- Do not require the whole subject to be fully above Y; local top support is enough.

T: Timeline / consistency
Judge from support geometry, not from category expectations.
If top support is visible, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "top_support_region": string,
  "support_evidence": string,
  "non_top_evidence": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_EDGE_RIM_PROMPT = """Judge whether the boundary statement is true in the image.

Use an edge/rim SRT process for "at the edge of". The key question is whether the subject lies near the outer boundary, rim, lip, or drop-off line of the reference rather than in its interior or center.

S: Sequence
1. Parse the statement into subject, relation, and reference.
2. Locate the clearest subject-reference pair.
3. Identify the reference's outer boundary, rim, edge line, or corner region.

R: Relation
Use these rules:
- X at the edge of Y: true if X is positioned very near Y's outer boundary, rim, lip, or end region.
- Beds: near the mattress edge or corner counts; near the middle does not.
- Sinks/toilets: on or immediately beside the rim/boundary counts; sitting in the deep interior basin does not.
- Cars/tables/counters: near the visible outer edge or corner counts; central placement does not.
- Visible overhang, immediate boundary contact, or closeness to a drop-off line supports true.
- Mere support by Y is not enough if X is clearly centered or interior.

T: Timeline / consistency
Judge from boundary proximity, not just contact or support.
If the outer-edge relation is visible, do not answer uncertain.

Return JSON with this schema:
{{
  "subject": string,
  "relation": string,
  "reference": string,
  "boundary_region": string,
  "edge_evidence": string,
  "interior_evidence": string,
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_VSR_RULES_GLM_PROMPT = """Use SRT-base to judge the statement.

S: Parse subject, relation, reference.
R: Judge only by visible spatial geometry, not commonsense plausibility.
- under/beneath: true if the subject is lower than the reference, or the reference rests on/supports over the subject.
- above/below: compare visible vertical position.
- behind/in front of: use depth cues, not just up/down.
- inside/in/within: check visible interior or opening.
- contains/surrounding: check enclosure or inclusion.
T: Final answer must follow the visual relation evidence only.

Return short JSON:
{{
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_ENCLOSURE_OPEN_GLM_PROMPT = """Use SRT-base to judge the statement.

S: Parse subject, relation, reference.
R: For surrounding/contains/inside, check whether the reference is visibly inside or enclosed by the subject's opening or boundary region.
- Full closure is not required.
- If the object is merely on top of or beside the subject, answer false.
T: Final answer must follow visible enclosure only.

Return short JSON:
{{
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_OPPOSITION_SHARED_SPACE_GLM_PROMPT = """Use SRT-base to judge the statement.

S: Parse subject, relation, reference.
R: For opposite to/across from, check whether the two objects occupy opposite sides of the same visible space, lane, curb gap, or open region.
- They do not need to face each other like people.
- If they are on the same side, answer false.
T: Final answer must follow the shared-space layout only.

Return short JSON:
{{
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_VSR_RULES_GLM_ULTRA_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: judge only by visible spatial geometry.
- under/beneath: lower than, or supports the reference
- above/below: vertical position
- behind/in front of: depth cues
- inside/in/within: visible interior/opening
- contains/surrounding: enclosure/inclusion
T: final answer must follow the visible relation only.

Return exactly one-line JSON and nothing else:
{{"answer":"true|false|uncertain","confidence":0.0}}

Statement: {question}
"""


SRT_SPATIAL_VSR_RULES_GLM_REFINED_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: judge only by visible spatial geometry.

Critical rules for under/beneath:
- true if the subject directly supports the reference, even when the reference is on top of the subject
- true if the subject is clearly below and aligned beneath the reference
- false if the subject is merely lower in the image but offset, separate, or just in front/behind
- false if both objects are simply on the same floor/plane without one being under the other
- if the reference is inside an open container, the container or its bottom can still be beneath/supporting the reference

Other rules:
- above/below: vertical position
- behind/in front of: depth cues
- inside/in/within: visible interior/opening
- contains/surrounding: enclosure/inclusion

T: final answer must follow the visible relation only.

Return exactly one-line JSON and nothing else:
{{"answer":"true|false|uncertain","confidence":0.0}}

Statement: {question}
"""


SRT_SPATIAL_VSR_RULES_GPT_REFINED_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: judge only by visible spatial geometry.

Critical rules for under/beneath:
- true if the subject directly supports the reference, even when the reference is on top of the subject
- true if the subject is clearly below and positioned under the reference
- false if the subject is only lower in the image but not actually under the reference
- false if both objects are simply on the same floor/plane without one being under the other
- if the reference is sitting inside an open container, the container or its bottom can still be under/beneath the reference

T: final answer must follow the visible relation only.

Return JSON:
{{
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_CONTAINMENT_GPT_REFINED_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: for inside / in / within / contains / surrounding, judge by visible enclosure or interior relation.

Critical rules for containment:
- true if the subject lies within the visible interior, opening, basin, cavity, or boundary region of the reference
- true if the reference visibly includes or encloses the subject, even when the container is open
- false if the subject is only on top of, beside, or touching the reference from outside
- false if the subject is merely near the boundary but not actually inside it
- do not require full closure; open containers still count when the interior relation is visually clear

T: final answer must follow the visible enclosure/interior evidence only.

Return JSON:
{{
  "answer": "true" | "false" | "uncertain",
  "confidence": number,
  "rationale": string
}}

Statement: {question}
"""


SRT_SPATIAL_VSR_RULES_MISTRAL_SHORT_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: judge only by visible spatial geometry.

Critical rules for under/beneath:
- true if the subject supports the reference, or is clearly below and under it
- false if the subject is only lower in the image but not actually under the reference
- false if both objects are just on the same floor/plane without an under relation

Other rules:
- above/below: vertical position
- behind/in front of: depth cues
- inside/in/within: visible interior/opening
- contains/surrounding: enclosure/inclusion

T: final answer must follow visible evidence only.

Return exactly one-line JSON and nothing else:
{{"answer":"true|false|uncertain","confidence":0.0}}

Statement: {question}
"""


SRT_SPATIAL_CONTAINMENT_MISTRAL_SHORT_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: for inside / in / within / contains / surrounding, judge only by visible interior or enclosure.

Critical rules:
- true if the subject is visibly inside the reference's interior, opening, cavity, or boundary region
- true if the reference visibly contains or encloses the subject, even if the container is open
- false if the subject is only on top of, beside, or touching the outside of the reference
- false if the subject is merely near the boundary but not actually inside

T: final answer must follow visible enclosure/interior evidence only.

Return exactly one-line JSON and nothing else:
{{"answer":"true|false|uncertain","confidence":0.0}}

Statement: {question}
"""


SRT_SPATIAL_ENCLOSURE_OPEN_GLM_ULTRA_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: for surrounding/contains/inside, check whether the reference is visibly inside or enclosed by the subject's opening or boundary.
- full closure is not required
- merely on top of / beside means false
T: final answer must follow visible enclosure only.

Return exactly one-line JSON and nothing else:
{{"answer":"true|false|uncertain","confidence":0.0}}

Statement: {question}
"""


SRT_SPATIAL_OPPOSITION_SHARED_SPACE_GLM_ULTRA_PROMPT = """Use SRT-base to judge the statement.

S: parse subject, relation, reference.
R: for opposite to/across from, check whether the two objects occupy opposite sides of the same visible space, lane, curb gap, or open region.
- if they are on the same side, answer false
T: final answer must follow the shared-space layout only.

Return exactly one-line JSON and nothing else:
{{"answer":"true|false|uncertain","confidence":0.0}}

Statement: {question}
"""


SPORTS_ACTION_DEFAULT_PROMPT = """Answer the visual question.

Choose the most likely phase of the visible sports action.

Use one label from this set:
- pre_action_setup
- active_execution
- follow_through_or_result
- resting_or_non_action

Return exactly one-line JSON and nothing else:
{{"action_phase":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


SPORTS_ACTION_GENERIC_SRT_PROMPT = """Use SRT-base to answer the visual question.

S: identify the sport, visible actor pose, equipment, ball/target if present, and any motion cues.
R: check whether the actor-equipment-ball/target relation supports a real sports action.
T: decide which action phase best matches the current frame.

Use one label from this set:
- pre_action_setup
- active_execution
- follow_through_or_result
- resting_or_non_action

Return exactly one-line JSON and nothing else:
{{"sport":string,"action_phase":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


SPORTS_ACTION_PHASE_PRIOR_PROMPT = """Use SRT-base with a sports action phase prior.

First inspect the whole scene, then decide the current action phase.

Phase definitions:

1. `pre_action_setup`
- the athlete is preparing before the main action
- examples: ready stance, holding bat/racket before swing, preparing to kick/throw/jump/ride

2. `active_execution`
- the main action is happening now
- examples: hitting, kicking, throwing, catching, jumping, riding, surfing, skiing, or body pose showing active motion

3. `follow_through_or_result`
- the main action just happened or the result is visible
- examples: completed swing, released ball/frisbee, landing after jump, post-action reaction

4. `resting_or_non_action`
- sports equipment is present, but no clear sports action phase is happening
- examples: standing near a bicycle, posing with equipment, parked bicycle, equipment lying nearby

Decision rules:
- do not choose `active_execution` just because sports equipment is visible
- choose `resting_or_non_action` when actor-equipment relation does not show an action
- choose `pre_action_setup` when the pose is preparatory and the main action has not started
- choose `active_execution` when pose, equipment, and ball/target relation show the main action underway
- choose `follow_through_or_result` when the action has already been completed or released

Use one label from this set:
- pre_action_setup
- active_execution
- follow_through_or_result
- resting_or_non_action

Return exactly one-line JSON and nothing else:
{{"sport":string,"action_phase":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


SPORTS_ACTION_VISIBLE_BOUNDARY_SRT_PROMPT = """Use SRT-base with a strict visible-boundary sports phase prior.

Classify only the phase that is visibly supported in the current frame. Do not infer a full event from sports context alone.

S: list the visible action cues:
- actor pose: ready/coiled, contact/propulsion, released/landed, or relaxed
- equipment relation: held, contacting, released/separated, or merely nearby
- ball/target relation if visible: before contact, at/near contact, already released/airborne, or absent

R: apply the boundary rules:
- `pre_action_setup`: ready stance, coiled body, raised bat/racket/arm, or aiming before visible contact/release.
- `active_execution`: the main action is visibly underway: contact, propulsion, riding/sliding/gliding, jump in progress, catch in progress, or body-equipment relation clearly doing the action now.
- `follow_through_or_result`: action just finished: ball/frisbee/object already released or airborne after a throw/hit, swing completed after contact, landing/reaction after jump, result of action visible.
- `resting_or_non_action`: equipment or sport setting is present but the actor is relaxed, posing, waiting, kneeling, standing around, or not visibly doing a sport action.

T: choose the earliest phase consistent with the visible evidence when the boundary is uncertain:
- raised racket/bat/leg with no visible contact -> `pre_action_setup`
- object already away from hand/foot/racket -> `follow_through_or_result`
- equipment nearby but no action posture -> `resting_or_non_action`
- moving on skis/surfboard/skateboard/bicycle with engaged body posture -> `active_execution`

Use one label from this set:
- pre_action_setup
- active_execution
- follow_through_or_result
- resting_or_non_action

Return exactly one-line JSON and nothing else:
{{"sport":string,"visible_cues":string,"action_phase":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


SPORTS_ACTION_BOUNDARY_VERIFY_PROMPT = """Answer the visual question, then verify the action-phase boundary.

Choose one label:
- pre_action_setup
- active_execution
- follow_through_or_result
- resting_or_non_action

Step 1: make the best visible-frame judgment.
Step 2: run these checks and revise only if the check clearly applies:
- Setup check: if the athlete is ready/coiled/aiming/raising equipment but there is no visible contact, release, jump, ride, slide, glide, or catch in progress, use `pre_action_setup`.
- Active check: if the athlete is visibly doing a continuous action now (skiing, surfing, skating, biking, riding) or is in contact/catch/propulsion during a discrete action, use `active_execution`.
- Follow-through check: if a thrown/hit/frisbee/ball/kite-like object is already separated from the actor after the release/contact, or the body is clearly after the swing/jump, use `follow_through_or_result`.
- Non-action check: if the scene only has equipment/context and no engaged sport posture, use `resting_or_non_action`.

Important:
- Do not change an initial answer unless a boundary check is visibly clear.
- Do not infer a hidden past action from context; use visible evidence.
- For continuous sports, engaged motion usually stays `active_execution`, not follow-through.

Return exactly one-line JSON and nothing else:
{{"initial_phase":string,"boundary_check":string,"action_phase":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


SPORTS_ACTION_SETUP_GUARD_PROMPT = """Answer the visual question with a short SRT guard.

Choose one label:
- pre_action_setup
- active_execution
- follow_through_or_result
- resting_or_non_action

Use these guard rules:

1. If the athlete is standing, crouching, coiled, holding equipment, aiming, or waiting, but there is no visible contact, release, swing extension, ride, slide, glide, jump, or catch in progress, choose `pre_action_setup`.
2. If equipment or uniform is visible but the actor is relaxed, standing around, seated, walking with equipment, or not using the equipment, choose `resting_or_non_action`.
3. Choose `active_execution` only when the main action is visibly underway in this frame: contact/propulsion, riding/sliding/gliding, catching, jumping, or a clearly engaged body-equipment relation.
4. Choose `follow_through_or_result` only when the action has visibly just completed: completed swing, object already released after throw/hit, landing, or post-action result.

Return exactly one-line JSON and nothing else:
{{"action_phase":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


SPORTS_ACTION_BOUNDARY_ROUTER_PROMPT = """Identify the sports action boundary family for this image.

Your job is only to decide what kind of process boundary the image tests. Do not answer the action phase.

Choose one boundary family:
- setup_vs_execution
- execution_vs_followthrough
- action_vs_nonaction
- equipment_context
- other

Family definitions:

1. `setup_vs_execution`
- the hard decision is whether the athlete is merely preparing or the main action is already happening
- cues: ready/coiled stance, raised bat/racket/leg/arm, aiming, waiting for contact/release

2. `execution_vs_followthrough`
- the hard decision is whether the main action is happening now or has just completed
- cues: ball/frisbee/object already airborne after release/contact, completed swing, landing/reaction after action

3. `action_vs_nonaction`
- the hard decision is whether this is a real sport action or just posing/resting/waiting
- cues: relaxed posture, people standing with equipment, no engaged motion or sport action

4. `equipment_context`
- sport equipment/context is visible, but the equipment relation itself is the main source of ambiguity
- cues: bicycle/skis/surfboard/skateboard/kite equipment, unclear whether it indicates active use or context only

5. `other`
- use only if none of the above is the main ambiguity.

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


TOOLUSE_DEFAULT_PROMPT = """Answer the visual question by choosing one operation-stage label.

Choose one label:
- preparing_tool
- active_tool_use
- result_or_finished
- unclear_or_non_process

Return exactly one-line JSON and nothing else:
{{"operation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


TOOLUSE_SRT_GENERIC_PROMPT = """Use SRT-base to answer the tool-use / object-operation question.

Choose one label:
- `preparing_tool`: a tool is present, held, or ready, but the operation is not visibly happening yet.
- `active_tool_use`: the tool is visibly being used on or with its target object now.
- `result_or_finished`: the result/output is visible, and active operation is no longer the central evidence.
- `unclear_or_non_process`: the image is not a clean operation-stage frame.

S: identify the ordered operation stage:
- setup/preparation -> active operation -> result/finished state

R: ground the visual relation:
- actor/person if visible
- tool
- target object
- hand/tool contact or attention relation
- visible state/result cues

T: choose the stage compatible with the visual timeline:
- tool merely present/held without contact -> `preparing_tool`
- tool contacting/manipulating target or clearly being used -> `active_tool_use`
- completed output/result dominates and active contact is absent -> `result_or_finished`
- if visual evidence does not support a clean stage -> `unclear_or_non_process`

Do not answer from the salient object alone. The final label must follow the actor-tool-object-state evidence.

Return exactly one-line JSON and nothing else:
{{"tool":string,"target_object":string,"visible_relation":string,"operation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


TOOLUSE_SRT_SCENARIO_PROMPT = """Use SRT-base with tool-use boundary rules.

Choose one label:
- `preparing_tool`
- `active_tool_use`
- `result_or_finished`
- `unclear_or_non_process`

S: identify the likely operation sequence:
- tool/object present -> actor prepares/holds tool -> actor uses tool on/with target -> result/display state

R: ground the visible relation:
- actor posture and attention
- tool in hand or near actor
- target object, body part, screen, food, or device
- contact, pointing, gesturing, display, or result cues

T: apply these boundary rules:

1. Setup vs active
- choose `preparing_tool` only when the tool is merely present, displayed, or held without visible use cues.
- choose `active_tool_use` when the actor is brushing/combing, talking on a phone, taking a photo, typing/working on a laptop, holding a game controller while gesturing/playing, or using a remote/controller even if the target screen is outside the frame.

2. Tool-object salience
- do not require the target object to be fully visible if the actor-tool pose conventionally implies active use, such as phone-to-ear, camera/phone aimed at a subject, hands on keyboard/laptop, or controller held during gameplay.
- if the tool/object is only a salient object in the scene and no actor-use relation is visible, do not call it active.

3. Result vs active
- choose `result_or_finished` for dining/served food, food trucks, plated food, objects displayed on a table, or a tool lying near a finished/result scene when no active tool contact is visible.
- choose `result_or_finished` when an object is being used as a resting/display surface rather than as an actively operated tool.

4. Unclear
- choose `unclear_or_non_process` only when neither active use nor result/display state can be visually grounded.

Return exactly one-line JSON and nothing else:
{{"tool":string,"target_object":string,"boundary_check":string,"operation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


TOOLUSE_SRT_SCENARIO_V2_PROMPT = """Use SRT-base with strict tool-use boundary rules.

Choose one label:
- `preparing_tool`
- `active_tool_use`
- `result_or_finished`
- `unclear_or_non_process`

First decide whether the image is a clean tool-use process frame.

Unclear guard:
- choose `unclear_or_non_process` if there is no real person/actor using a tool and no clear finished/result state.
- choose `unclear_or_non_process` for decorative displays, wax/dummy scenes, random objects, or scenes where the named tool is only incidental.
- do not force every object into a process stage.

S: identify the operation sequence:
- tool present -> actor prepares/holds tool -> actor actively uses tool -> result/display state

R: ground the actor-tool-target relation:
- actor visible or strongly implied
- tool in hand, near body, aimed, contacting, or displayed
- target object/body/screen/device/food
- visible action or result cues

T: apply the boundary rules:

1. `preparing_tool`
- tool is merely held, shown, or ready.
- no clear action, contact, aiming, gesturing, or target engagement.

2. `active_tool_use`
- actor is visibly using the tool now.
- examples: brushing/combing hair or teeth, phone at ear, person talking on phone, phone/camera aimed to take a photo, hands on laptop/keyboard while working, controller/remote held while playing or gesturing toward an off-frame screen, tool adjusting/contacting a target.
- for phones: choose active only when the person is using it for communication/photo/interaction, not merely when a phone screen is displayed.

3. `result_or_finished`
- the visible scene is a completed/result/display state rather than an ongoing tool action.
- examples: plated/dining food, food truck/service result, object displayed on phone screen, cat/object resting on laptop, tool lying near finished objects.

4. `unclear_or_non_process`
- no actor-tool-target relation and no clean result/display state.
- use this instead of guessing when the scene is a prop, dummy, decoration, or unrelated object collection.

Return exactly one-line JSON and nothing else:
{{"clean_process_frame":true,"tool":string,"target_object":string,"boundary_check":string,"operation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CLEANING_DEFAULT_PROMPT = """Answer the cleaning / household-task stage question from the image.

Choose one label:
- `pre_cleaning_setup`: cleaning tools, target objects, or setup are visible, but active cleaning is not clearly happening yet.
- `active_cleaning`: a person/tool/water/hand is visibly cleaning, washing, wiping, brushing, rinsing, or scrubbing a target now.
- `cleaned_or_finished`: the cleaned/result state is visible, and active cleaning is no longer central.
- `unclear_or_non_process`: this is not a clean cleaning-process frame.

Return exactly one-line JSON and nothing else:
{{"cleaning_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CLEANING_SRT_GENERIC_PROMPT = """Use SRT-base to answer the cleaning / household-task stage question.

Choose one label:
- `pre_cleaning_setup`
- `active_cleaning`
- `cleaned_or_finished`
- `unclear_or_non_process`

S: identify the ordered cleaning process:
- setup/dirty target/tools ready -> active cleaning contact -> cleaned/result state

R: ground the visual relation:
- actor/person if visible
- cleaning tool, water, soap, cloth, brush, hose, hand, or toothbrush
- target object/body/surface/dish/vehicle
- contact, wiping, washing, brushing, rinsing, scrubbing, or result cues

T: choose the stage compatible with the visual timeline:
- tool or target merely present -> `pre_cleaning_setup`
- visible contact/water/action on the target -> `active_cleaning`
- clean/result display without ongoing action -> `cleaned_or_finished`
- cleaning-related objects are only salient but no process is grounded -> `unclear_or_non_process`

Do not answer from a bathroom/sink/toothbrush/object cue alone. The final label must follow actor-tool-target-state evidence.

Return exactly one-line JSON and nothing else:
{{"tool_or_medium":string,"target":string,"visible_relation":string,"cleaning_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CLEANING_SRT_SCENARIO_PROMPT = """Use SRT-base with cleaning state-change boundary rules.

Choose one label:
- `pre_cleaning_setup`
- `active_cleaning`
- `cleaned_or_finished`
- `unclear_or_non_process`

S: identify the likely cleaning sequence:
- dirty/target area or object -> cleaning tool/medium prepared -> tool/water/hand contacts target -> cleaned or finished state

R: ground the visible process relation:
- actor posture and attention
- cleaning medium/tool: water, hose, soap, cloth, brush, toothbrush, hand, sponge, towel
- target: teeth/body, dish/tableware, vehicle, surface, bathroom fixture, clothing, object
- contact/action cues: wiping, brushing, washing, rinsing, scrubbing, spraying, soaking
- state cues: dirty target, wet target, foam/soap, clean result, arranged/finished object

T: apply these boundary rules:

1. Setup vs active
- choose `pre_cleaning_setup` when cleaning tools/targets are present but there is no visible contact, water flow, wiping, brushing, or scrubbing.
- choose `active_cleaning` when the actor/tool/water is contacting or clearly acting on the target.

2. Cleaning vs object salience
- do not call a bathroom, sink, toilet, toothbrush, dish, or vehicle a cleaning process just because it is visible.
- if the scene only contains cleaning-related objects without a visible cleaning relation, choose `unclear_or_non_process` or `pre_cleaning_setup` depending on whether setup is visually grounded.

3. Active vs result
- choose `cleaned_or_finished` when the main evidence is a completed clean/result/display state and no active contact is visible.
- if active water/tool contact is visible, prefer `active_cleaning` even if the target already looks partly clean.

4. Unclear
- choose `unclear_or_non_process` when neither setup, active cleaning, nor cleaned/result state can be visually grounded.

Return exactly one-line JSON and nothing else:
{{"tool_or_medium":string,"target":string,"boundary_check":string,"cleaning_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CLEANING_SRT_SCENARIO_V2_PROMPT = """Use SRT-base with result-aware cleaning boundary rules.

Choose one label:
- `pre_cleaning_setup`
- `active_cleaning`
- `cleaned_or_finished`
- `unclear_or_non_process`

The goal is not to guess that cleaning is happening. The goal is to decide which visible process stage is supported.

S: use this ordered cleaning timeline:
1. pre-cleaning setup: tool/target is ready, but no cleaning contact yet
2. active cleaning: water/tool/hand is currently acting on the target
3. cleaned/result: the target or scene looks like the finished clean/result state; active contact is absent
4. unclear/non-process: cleaning-related objects are present but no stage is visually grounded

R: ground the relation before choosing a label:
- actor: visible person, hand, or no actor
- medium/tool: water, hose, soap, cloth, brush, toothbrush, hand, sponge, towel, or none
- target: teeth/body, dish, vehicle, surface, bathroom fixture, room, object
- contact/action: touching, wiping, brushing, rinsing, spraying, scrubbing, or none
- result cue: clean vehicle/object/room, displayed clean surface, finished state, or none

T: apply the result-aware decision rules:

1. Active cleaning requires active evidence.
- choose `active_cleaning` only if there is visible contact/action: brushing teeth, washing with water, spraying/rinsing, wiping, scrubbing, or a hand/tool clearly cleaning a target.
- do not choose active merely because an object is clean, shiny, wet, in a bathroom, or near a sink.

2. Finished/result state is a real stage.
- choose `cleaned_or_finished` when the main evidence is a clean/result state and no active cleaning contact is visible.
- examples: a clean car/motorcycle/object shown as a result, a clean room/bathroom/sink/fixture, or a finished/display state after cleaning.

3. Setup vs unclear.
- choose `pre_cleaning_setup` when tools and targets are positioned for cleaning but the action has not started.
- choose `unclear_or_non_process` when cleaning-related objects are merely salient and no setup/action/result boundary is visually grounded.

Return exactly one-line JSON and nothing else:
{{"actor":string,"tool_or_medium":string,"target":string,"contact_action":string,"result_cue":string,"cleaning_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CLEANING_SRT_SALIENCE_GUARD_PROMPT = """Use a conservative cleaning salience guard.

Choose one label:
- `pre_cleaning_setup`
- `active_cleaning`
- `cleaned_or_finished`
- `unclear_or_non_process`

This item is mainly about distinguishing a real cleaning process from a cleaning-related object or scene.

Rules:
- Do not infer cleaning only because the image contains a bathroom, sink, toilet, toothbrush, dish, vehicle, clean object, or wet-looking surface.
- Choose `active_cleaning` only if a person/tool/water/hand is visibly cleaning, washing, wiping, brushing, rinsing, or scrubbing a target.
- Choose `pre_cleaning_setup` only if a cleaning tool/medium and target are arranged for cleaning but the action has not started.
- Choose `cleaned_or_finished` only if a cleaned/result state is clearly the main visual evidence.
- Otherwise choose `unclear_or_non_process`.

Return exactly one-line JSON and nothing else:
{{"salient_object":string,"grounded_cleaning_relation":string,"cleaning_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CRAFT_DEFAULT_PROMPT = """Answer the craft / art / creation-order stage question from the image.

Choose one label:
- `pre_creation_setup`: tools, materials, target surface/object, or setup are visible, but creation contact is not clearly happening yet.
- `active_creation`: a person/hand/tool is visibly making, writing, drawing, painting, cutting, decorating, or modifying a target now.
- `active_assembly`: parts/materials are being assembled or constructed into an artifact now.
- `finished_artifact`: the finished artwork, decorated object, cake, writing, or constructed artifact is visible, and active creation is not central.
- `unclear_or_non_creation`: this is not a grounded creation-process frame.

Return exactly one-line JSON and nothing else:
{{"creation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CRAFT_SRT_GENERIC_PROMPT = """Use SRT-base to answer the craft / art / creation-order stage question.

Choose one label:
- `pre_creation_setup`
- `active_creation`
- `active_assembly`
- `finished_artifact`
- `unclear_or_non_creation`

S: identify the ordered creation process:
- materials/tools ready -> active making/modifying -> assembled/decorated/finished artifact

R: ground the visual relation:
- actor/person/hand if visible
- tool or medium: pen, brush, pencil, marker, scissors, knife, icing/frosting, hands, parts, materials
- target: paper, canvas, book, surface, cake, fabric, object, constructed artifact
- contact/action cues: writing, drawing, painting, cutting, decorating, applying, assembling, constructing
- result cues: finished artwork, written text, decorated cake/object, completed artifact

T: choose the stage compatible with the visual timeline:
- materials/tool only, no contact -> `pre_creation_setup`
- visible tool/hand contact changing the target -> `active_creation`
- visible part-whole construction -> `active_assembly`
- completed artifact without active making -> `finished_artifact`
- creation-related objects are only salient but no process is grounded -> `unclear_or_non_creation`

Do not answer from object salience alone. A painting, drawing, cake, book, paper, or scissors can be a finished object or prop rather than an active creation process.

Return exactly one-line JSON and nothing else:
{{"tool_or_medium":string,"target":string,"visible_relation":string,"creation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CRAFT_SRT_SCENARIO_PROMPT = """Use SRT-base with craft/art creation boundary rules.

Choose one label:
- `pre_creation_setup`
- `active_creation`
- `active_assembly`
- `finished_artifact`
- `unclear_or_non_creation`

S: identify the likely creation sequence:
1. setup: materials/tools/surface are ready
2. active creation: tool/hand/medium contacts and changes the target
3. assembly: parts/materials are being put together
4. finished artifact: the created/modified object is shown as a result

R: ground the visible process relation:
- actor posture and attention
- tool/medium: pen, pencil, brush, marker, scissors, knife, icing, frosting, hands, parts, materials
- target: paper, canvas, book, fabric, cake, surface, object, artifact
- contact/action: writing, drawing, painting, cutting, decorating, applying, assembling, constructing
- state cues: blank vs marked surface, uncut vs cut material, undecorated vs decorated object, partial vs finished artifact

T: apply these boundary rules:

1. Setup vs active
- choose `pre_creation_setup` when tools/materials/targets are present but there is no visible contact or modification.
- choose `active_creation` when a hand/tool/medium is contacting or clearly modifying the target.

2. Assembly vs finished
- choose `active_assembly` when separate parts/materials are visibly being combined.
- choose `finished_artifact` when the artifact is already complete and no active construction is visible.

3. Active vs finished artifact
- do not call a displayed painting, drawing, decorated cake, written text, or artwork active creation unless the making action is visible.
- if the main evidence is the completed artifact, choose `finished_artifact`.

4. Unclear
- choose `unclear_or_non_creation` when creation-related objects are only salient and no setup/action/result boundary is visually grounded.

Return exactly one-line JSON and nothing else:
{{"tool_or_medium":string,"target":string,"boundary_check":string,"creation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CRAFT_SRT_SCENARIO_V2_PROMPT = """Use SRT-base with result-aware craft/art creation boundary rules.

Choose one label:
- `pre_creation_setup`
- `active_creation`
- `active_assembly`
- `finished_artifact`
- `unclear_or_non_creation`

The goal is not to guess that creation is happening. The goal is to decide which visible process stage is supported.

S: use this ordered creation timeline:
1. setup: materials/tools are ready, but no modification yet
2. active creation: hand/tool/medium is changing the target
3. active assembly: parts are being combined into an artifact
4. finished artifact: completed created/modified object is visible; active making is absent
5. unclear/non-creation: materials or artifacts are salient but no stage is grounded

R: ground the relation before choosing a label:
- actor: visible person/hand or no actor
- tool/medium: pen, pencil, brush, marker, scissors, knife, icing/frosting, hand, parts, or none
- target: paper, canvas, book, cake, fabric, surface, object, artifact
- contact/action: touching, writing, drawing, painting, cutting, applying, decorating, assembling, or none
- result cue: completed artwork/writing/decorated cake/cut object/assembled artifact or none

T: apply the result-aware decision rules:
- `active_creation` requires visible contact/action that changes the target.
- `active_assembly` requires visible part-whole construction, not just a building/object in the scene.
- `finished_artifact` is a real stage: use it for displayed paintings, drawings, writing, decorated cakes, or completed objects when no active making is visible.
- `pre_creation_setup` requires tools/materials arranged for creation but no action yet.
- `unclear_or_non_creation` is for object salience without a grounded process.

Return exactly one-line JSON and nothing else:
{{"actor":string,"tool_or_medium":string,"target":string,"contact_action":string,"result_cue":string,"creation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


CRAFT_SRT_SALIENCE_GUARD_PROMPT = """Use a conservative craft/art salience guard.

Choose one label:
- `pre_creation_setup`
- `active_creation`
- `active_assembly`
- `finished_artifact`
- `unclear_or_non_creation`

This item is mainly about distinguishing a real creation process from creation-related object salience.

Rules:
- Do not infer active creation only because the image contains a painting, drawing, writing, cake, paper, book, scissors, brush, or art object.
- Choose `active_creation` only if a person/hand/tool/medium is visibly making, writing, drawing, painting, cutting, decorating, or modifying a target.
- Choose `active_assembly` only if parts/materials are visibly being combined.
- Choose `pre_creation_setup` only if tools/materials and target are arranged for creation but the action has not started.
- Choose `finished_artifact` when the completed created/modified object is the main evidence and active creation is absent.
- Otherwise choose `unclear_or_non_creation`.

Return exactly one-line JSON and nothing else:
{{"salient_object":string,"grounded_creation_relation":string,"creation_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


MOBILITY_DEFAULT_PROMPT = """Answer the mobility / transport process-stage question from the image.

Choose one label:
- `pre_mobility_setup`: an actor and vehicle/device are staged for movement, but active movement is not clearly visible.
- `active_mobility`: a person, animal, or vehicle/device is visibly riding, driving, flying, skiing, surfing, skateboarding, or otherwise moving.
- `parked_or_finished`: the vehicle/device is stationary, parked, displayed, locked, stopped, or the mobility event is over.
- `unclear_or_non_mobility`: a vehicle/mobility object is salient, but no mobility-process stage is visually grounded.

Return exactly one-line JSON and nothing else:
{{"mobility_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


MOBILITY_SRT_GENERIC_PROMPT = """Use SRT-base to answer the mobility / transport process-stage question.

Choose one label:
- `pre_mobility_setup`
- `active_mobility`
- `parked_or_finished`
- `unclear_or_non_mobility`

S: identify the ordered mobility process:
- prepare/stage with vehicle or device -> active movement/control -> stopped/parked/finished state

R: ground the visual relation:
- actor/rider/driver/animal if visible
- vehicle/device: bicycle, motorcycle, skateboard, skis, snowboard, surfboard, horse, car, bus, train, airplane, boat
- relation: mounted/on/in/control of, standing beside, waiting near, parked/stationary, moving through scene
- state cues: posture, wheel/contact state, road/water/snow/air context, stationary display, parking/stop cues

T: choose the stage compatible with the visual timeline:
- staged near/on vehicle but no visible motion/control -> `pre_mobility_setup`
- visible riding/driving/flying/skiing/surfing/skateboarding/moving -> `active_mobility`
- vehicle/device is parked/stopped/displayed and movement is absent -> `parked_or_finished`
- vehicle is only salient but no process stage is grounded -> `unclear_or_non_mobility`

Do not infer active movement from object salience alone. A car, bike, skateboard, horse, or airplane can be a prop, parked object, or background object.

Return exactly one-line JSON and nothing else:
{{"actor":string,"vehicle":string,"visible_relation":string,"mobility_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


MOBILITY_SRT_SCENARIO_PROMPT = """Use SRT-base with mobility/transport boundary rules.

Choose one label:
- `pre_mobility_setup`
- `active_mobility`
- `parked_or_finished`
- `unclear_or_non_mobility`

S: identify the likely mobility sequence:
1. setup: actor/vehicle/device is positioned before motion
2. active mobility: rider/driver/vehicle/device is moving or being controlled
3. parked/finished: vehicle/device is stopped, parked, displayed, or no longer in motion
4. unclear/non-mobility: vehicle/device is salient but no process stage is grounded

R: ground the visible mobility relation:
- actor: visible rider, driver, passenger, animal, or none
- vehicle/device: bike, motorcycle, skateboard, skis, snowboard, surfboard, horse, car, bus, train, airplane, boat
- relation: mounted, seated in, steering, standing beside, waiting, parked, stopped, moving, airborne
- cues: rider posture, wheel/ski/board contact, road/water/snow/air context, parking position, absence of actor/control

T: apply these boundary rules:

1. Setup vs active
- choose `pre_mobility_setup` when the actor/vehicle is staged, waiting, standing beside, or mounted without clear motion/control.
- choose `active_mobility` when movement/control is visible: riding, driving, flying, skiing, surfing, skateboarding, or moving through the scene.

2. Active vs parked/finished
- choose `parked_or_finished` when the vehicle/device is stationary, parked, locked, displayed, stopped, or presented as an object.
- do not call a parked bike/car/motorcycle active just because it is a mobility object.

3. Mobility vs salience
- choose `unclear_or_non_mobility` when vehicle/mobility objects are only background or salient objects and no stage is grounded.

Return exactly one-line JSON and nothing else:
{{"actor":string,"vehicle":string,"boundary_check":string,"mobility_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


MOBILITY_SRT_SCENARIO_V2_PROMPT = """Use SRT-base with conservative motion-evidence rules.

Choose one label:
- `pre_mobility_setup`
- `active_mobility`
- `parked_or_finished`
- `unclear_or_non_mobility`

The goal is not to guess that movement is happening. The goal is to decide which visible process stage is supported.

S: use this ordered mobility timeline:
1. pre-mobility setup: actor/device positioned before movement
2. active mobility: visible control or movement
3. parked/finished: stopped, parked, displayed, or no longer active
4. unclear/non-mobility: vehicle/device salience without a grounded process

R: ground the relation before choosing:
- actor: rider/driver/person/animal/none
- vehicle: bike, motorcycle, skateboard, skis, snowboard, surfboard, horse, car, bus, train, airplane, boat
- contact/control: mounted, seated, steering, holding, standing beside, none
- motion evidence: moving posture, road/water/snow/air trajectory, wheels/board/skis in use, or none
- stationary evidence: parked position, lock/kickstand/display, stopped at curb/station, no rider/control

T: conservative decision rules:
- `active_mobility` requires visible motion or active control, not merely a vehicle in the image.
- `pre_mobility_setup` requires a plausible actor-vehicle setup relation before motion.
- `parked_or_finished` is a real stage: use it when stationary/parked/display evidence is strongest.
- `unclear_or_non_mobility` is for background/salient mobility objects without grounded process evidence.

Return exactly one-line JSON and nothing else:
{{"actor":string,"vehicle":string,"control_relation":string,"motion_evidence":string,"stationary_evidence":string,"mobility_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


MOBILITY_SRT_SALIENCE_GUARD_PROMPT = """Use a conservative mobility salience guard.

Choose one label:
- `pre_mobility_setup`
- `active_mobility`
- `parked_or_finished`
- `unclear_or_non_mobility`

This item is mainly about distinguishing a real mobility process from vehicle/object salience.

Rules:
- Do not infer active mobility only because the image contains a car, bus, train, airplane, bike, motorcycle, skateboard, horse, skis, snowboard, surfboard, or boat.
- Choose `active_mobility` only if movement/control is visibly grounded.
- Choose `pre_mobility_setup` only if actor and vehicle/device are staged before movement.
- Choose `parked_or_finished` when the vehicle/device is clearly stationary, parked, stopped, locked, or displayed.
- Otherwise choose `unclear_or_non_mobility`.

Return exactly one-line JSON and nothing else:
{{"salient_vehicle":string,"grounded_mobility_relation":string,"mobility_stage":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""


def _format_answer_space(labels: list[str]) -> str:
    return "\n".join(f"- `{label}`" for label in labels)


def _process_family_name(item: dict[str, Any]) -> str:
    return str(item.get("scenario_family") or item.get("task") or "process").replace("_", " ")


def _process_label_field(item: dict[str, Any]) -> str:
    family = str(item.get("scenario_family") or "")
    if "assembly" in family:
        return "assembly_stage"
    if "physical" in family:
        return "transition_stage"
    if "social" in family:
        return "event_stage"
    if "traffic" in family or "navigation" in family:
        return "navigation_stage"
    if "affordance" in family or "object_use" in family:
        return "use_stage"
    return "process_step"


def _process_domain_prior(item: dict[str, Any]) -> str:
    family = str(item.get("scenario_family") or "")
    if "assembly" in family:
        return """Domain process prior:
S: ordered process = materials/object ready -> active assembly/repair/setup -> assembled or finished result.
R: ground actor-object-tool relations: person/hand/tool contact, parts being joined, object merely present, or completed object.
T: choose setup before active contact, active assembly while the object is being changed, finished when the assembled object is only a result, and salience when no assembly stage is grounded."""
    if "physical" in family:
        return """Domain process prior:
S: ordered process = stable pre-state -> active transition/change -> post-transition result.
R: ground the physical relation: gravity, fluid, heat/fire, damage, tool contact, motion/contact cue, or result-state cue.
T: choose active transition only when change is happening now; choose result when the changed state is visible but the transition is over; choose stable when the object is pre-change; choose salience when no transition stage is grounded."""
    if "social" in family:
        return """Domain process prior:
S: ordered script = event setup -> active social/event participation -> after-event residue/result.
R: ground people, props, event space, participation roles, table/party/ceremony/public-event cues.
T: choose active event only when people are engaged in the event; setup when props/space are prepared before participation; post-event when remains/residue indicate it happened already; salience when event-like objects do not ground a script stage."""
    if "traffic" in family or "navigation" in family:
        return """Domain process prior:
S: ordered navigation process = waiting/pre-navigation -> active movement/navigation -> stopped/parked.
R: ground actor/vehicle/path/control relations: pedestrian crossing, bus/train waiting, vehicle moving, traffic-control stop, parked/stationary state.
T: choose active navigation only with visible movement through traffic space; waiting when positioned before movement; stopped/parked when motion is over or blocked; salience when traffic objects alone do not ground a process."""
    if "affordance" in family or "object_use" in family:
        return """Domain process prior:
S: ordered object-use process = ready-to-use setup -> active affordance use -> used/finished state.
R: ground actor-object affordance relation: holding/nearby, using according to function, post-use residue, or object salience.
T: choose active use only when the object's function is being executed; setup when the object is available but not clearly used; finished when post-use state is visible; salience when an affordance-capable object is merely present."""
    return """Domain process prior:
S: identify the plausible ordered process stages.
R: ground actor, object, relation, and state cues.
T: choose the stage supported by the current visible frame, not a hidden before/after story."""


def _process_boundary_rule(item: dict[str, Any]) -> str:
    boundary = str(item.get("boundary_family") or item.get("boundary_family_hint") or "unknown")
    neighbor = str(item.get("neighbor_label") or "neighbor stage")
    why_not = str(item.get("why_not_neighbor") or "Use visible evidence to reject the tempting neighboring stage.")
    return f"""Boundary router hint:
- Current boundary family: `{boundary}`
- Tempting neighbor stage: `{neighbor}`
- Visual reason to reject the neighbor when applicable: {why_not}

Apply only the rule needed for this boundary, then choose the current visible stage."""


def _process_wrong_boundary_rule(item: dict[str, Any]) -> str:
    boundary = str(item.get("boundary_family") or item.get("boundary_family_hint") or "unknown")
    wrong_map = {
        "stable_vs_active": "active_vs_result",
        "active_vs_result": "stable_vs_active",
        "setup_vs_active": "assembly_vs_finished",
        "assembly_vs_finished": "assembly_vs_salience",
        "assembly_vs_salience": "setup_vs_active",
    }
    wrong_boundary = wrong_map.get(boundary, "setup_vs_active" if boundary != "setup_vs_active" else "active_vs_result")
    return f"""Boundary router hint:
- Current boundary family: `{wrong_boundary}`
- This hint is intentionally selected by a boundary router and should be followed as the controlling process boundary.

Apply only the rule needed for this boundary, then choose the current visible stage."""


def _process_boundary_choices(item: dict[str, Any]) -> list[str]:
    family = str(item.get("scenario_family") or "")
    if "physical" in family:
        return ["stable_vs_active", "active_vs_result"]
    if "assembly" in family:
        return ["setup_vs_active", "assembly_vs_finished", "assembly_vs_salience"]
    if "traffic" in family or "navigation" in family:
        return ["waiting_vs_active", "active_vs_stopped", "navigation_vs_salience"]
    if "affordance" in family or "object_use" in family:
        return ["setup_vs_active_use", "active_use_vs_finished", "affordance_vs_salience"]
    return ["setup_vs_active", "active_vs_result", "process_vs_salience"]


def _process_label_only_choices(item: dict[str, Any]) -> list[str]:
    gold = str(item.get("gold_label") or item.get("expected_answer") or item.get("answer") or "")
    neighbor = str(item.get("neighbor_label") or "")
    labels = [str(label) for label in (item.get("answer_space") or [])]
    choices: list[str] = []
    for label in (gold, neighbor):
        if label and label not in choices:
            choices.append(label)
    if len(choices) < 2:
        for label in labels:
            if label and label != gold and label not in choices:
                choices.append(label)
            if len(choices) >= 2:
                break
    if not choices:
        choices = labels[:2]
    return choices or ["current_stage", "neighbor_stage"]


def make_process_prompt(mode: str, item: dict[str, Any]) -> str:
    question = str(item.get("underspecified_question") or item.get("question") or item.get("original_question") or "")
    labels = item.get("answer_space") or []
    if not isinstance(labels, list) or not labels:
        labels = ["pre_process_setup", "active_process", "finished_or_result", "unclear_or_non_process"]
    label_field = _process_label_field(item)
    answer_space = _format_answer_space([str(label) for label in labels])
    family_name = _process_family_name(item)
    domain_prior = _process_domain_prior(item)
    boundary_rule = _process_boundary_rule(item)
    wrong_boundary_rule = _process_wrong_boundary_rule(item)
    boundary_choices = _format_answer_space(_process_boundary_choices(item))
    label_only_choices = _format_answer_space(_process_label_only_choices(item))

    if mode == "process_default":
        return f"""Answer the {family_name} process-stage question from the image.

Choose one label:
{answer_space}

Return exactly one-line JSON and nothing else:
{{"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_visual_evidence_only":
        return f"""Inspect the image for the {family_name} process-stage question.

Choose one label:
{answer_space}

List only the visible evidence that may matter:
- actor or hand visibility
- object/tool visibility
- relation between actor, object, and tool
- object state or result cues
- any ambiguity or missing evidence

Do not use an explicit SRT decision rule. After listing the evidence, choose the best label directly from what is visible.

Return exactly one-line JSON and nothing else:
{{"visible_actor":string,"visible_object":string,"visible_relation":string,"state_cues":string,"ambiguity":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_plain_cot":
        return f"""Answer the {family_name} process-stage question from the image.

Choose one label:
{answer_space}

Think step by step about the image and the question, then choose the best label.
Do not use SRT terminology or a predefined SRT decision rule.

Return exactly one-line JSON and nothing else:
{{"reasoning":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_multimodal_cot":
        return f"""Answer the {family_name} process-stage question from the image using multimodal chain-of-thought.

Choose one label:
{answer_space}

First describe the relevant visual content. Then reason step by step from that description to the answer.
Do not use SRT terminology or a predefined SRT decision rule.

Return exactly one-line JSON and nothing else:
{{"visual_description":string,"reasoning":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_self_verification":
        return f"""Answer the {family_name} process-stage question from the image with self-verification.

Choose one label:
{answer_space}

First make an initial label choice. Then verify whether the choice is supported by visible image evidence.
If the initial choice is not supported, revise it. Do not use SRT terminology or a predefined SRT decision rule.

Return exactly one-line JSON and nothing else:
{{"initial_answer":string,"verification":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_llava_cot_style":
        return f"""Answer the {family_name} process-stage question from the image using a LLaVA-CoT-style four-stage reasoning format.

Choose one label:
{answer_space}

Follow these four stages:
1. Summary: summarize the task and the candidate process labels.
2. Visual Interpretation: describe only visible image evidence relevant to the current process stage.
3. Logical Reasoning: reason from the visible evidence to the most likely label.
4. Conclusion: give the final label.

Do not use SRT terminology or a predefined SRT decision rule.

Return exactly one-line JSON and nothing else:
{{"summary":string,"visual_interpretation":string,"logical_reasoning":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_grounded_cot":
        return f"""Answer the {family_name} process-stage question from the image using grounded visual chain-of-thought.

Choose one label:
{answer_space}

First identify the minimal visual evidence needed for the decision:
- decision target in the question
- visible actor/body/tool/object cues
- spatial contact or interaction cues
- state/result cues
- missing or ambiguous visual evidence

Then reason step by step using only the selected visual evidence and choose the best label.
Do not use SRT terminology or a predefined SRT decision rule.

Return exactly one-line JSON and nothing else:
{{"decision_target":string,"grounded_evidence":[string],"reasoning":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_policy_only":
        return f"""Answer the {family_name} process-stage question using only the SRT target-decision policy.

Choose one label:
{answer_space}

Policy:
- choose the current visible stage only
- do not infer a hidden earlier or later story
- do not replace the target label with a broad scene/activity label
- if the current stage is not visually verifiable, choose the unclear/non-process label when available

Do not enumerate visual evidence before answering.

Return exactly one-line JSON and nothing else:
{{"target_decision":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_s_t":
        return f"""Use a partial SRT process with State + Target only.

Choose one label:
{answer_space}

S: extract the minimal visible state needed for this decision: actor/hand, object/tool, object state, and result cues.
T: choose the current visible stage only. Do not infer hidden earlier/later steps or a broad activity label.

Do not perform a separate relevance/compatibility check.

Return exactly one-line JSON and nothing else:
{{"state":string,"target_decision":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_r_t":
        return f"""Use a partial SRT process with Relevance + Target only.

Choose one label:
{answer_space}

R: check whether the visible cues are relevant to the decision target: actor-object/tool contact, state-change cue, result cue, or mere object salience.
T: choose the current visible stage only. Do not infer hidden earlier/later steps or a broad activity label.

Do not first enumerate the full visible state.

Return exactly one-line JSON and nothing else:
{{"relevance_check":string,"target_decision":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_s_r_no_t":
        return f"""Use a partial SRT process with State + Relevance, but without a final SRT target-decision rule.

Choose one label:
{answer_space}

S: extract the minimal visible state needed for this process question.
R: check which visible cues are relevant to the process relation or boundary.

Do not apply the final SRT rule about current verifiable target, hidden earlier/later stories, or broad scene labels. Choose the best label after the State and Relevance notes.

Return exactly one-line JSON and nothing else:
{{"state":string,"relevance_check":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_generic":
        return f"""Use SRT-base to answer the {family_name} process-stage question.

Choose one label:
{answer_space}

S: identify the plausible sequence of stages in this process.
R: ground the visible actor, object/tool, relation, and state cues.
T: choose the stage supported by the current visible frame. Do not replace the visible stage with a hidden earlier/later story or a broad scene label.

Return exactly one-line JSON and nothing else:
{{"sequence":string,"relation":string,"timeline_check":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_scenario":
        return f"""Use SRT-base with a scenario-specific process prior.

Choose one label:
{answer_space}

{domain_prior}

Final rule:
- answer the current visible stage only
- do not infer an active process from object/context salience alone
- if evidence is insufficient, choose the unclear/non-process label from the answer space

Return exactly one-line JSON and nothing else:
{{"visible_actor":string,"visible_object":string,"visible_relation":string,"state_cues":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_boundary_router_oracle":
        return f"""Use SRT-base with an oracle process-boundary hint.

Choose one label:
{answer_space}

{domain_prior}

{boundary_rule}

Final rule:
- resolve the specified boundary using visible evidence
- choose the current visible stage, not the broad activity or a plausible hidden story
- do not infer active process from object/context salience alone

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"boundary_decision":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_boundary_label_only":
        boundary = str(item.get("boundary_family") or item.get("boundary_family_hint") or "unknown")
        return f"""Answer the {family_name} process-stage question from the image.

This is a label-only oracle control. The image has been assigned to this process boundary:
`{boundary}`

Choose one label only from this narrowed candidate set:
{label_only_choices}

Do not use SRT. Do not enumerate State, Relevance, or Target steps. Do not apply any boundary-specific verification rule. Use the image and the narrowed candidate set only.

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"candidate_set":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_wrong_boundary_router":
        return f"""Use SRT-base with a process-boundary hint.

Choose one label:
{answer_space}

{domain_prior}

{wrong_boundary_rule}

Final rule:
- resolve the specified boundary using visible evidence
- choose the current visible stage, not the broad activity or a plausible hidden story
- do not infer active process from object/context salience alone

Return exactly one-line JSON and nothing else:
{{"boundary_family":string,"boundary_decision":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    if mode == "process_srt_self_router":
        return f"""Use SRT-base with a non-oracle self-router.

Choose one label:
{answer_space}

{domain_prior}

First choose the process boundary family that best controls the current visual decision. Use only visible evidence, not the gold answer.

Choose one boundary family exactly from this set:
{boundary_choices}

Then apply the selected boundary to choose the current visible stage.

Final rule:
- choose the current visible stage, not the broad activity or a plausible hidden story
- do not infer active process from object/context salience alone
- if the selected boundary is uncertain, say so in `boundary_confidence`

Return exactly one-line JSON and nothing else:
{{"predicted_boundary_family":string,"boundary_confidence":0.0,"boundary_decision":string,"{label_field}":string,"confidence":0.0,"rationale":string}}

Question: {question}
"""
    raise ValueError(f"Unknown process prompt mode: {mode}")


def make_prompt(mode: str, question: str) -> str:
    if mode == "default":
        return DEFAULT_PROMPT.format(question=question)
    if mode == "direct":
        return DIRECT_PROMPT.format(question=question)
    if mode == "evidence":
        return EVIDENCE_PROMPT.format(question=question)
    if mode == "structured_prior":
        return STRUCTURED_PRIOR_PROMPT.format(question=question)
    if mode == "visual_sketchpad":
        return VISUAL_SKETCHPAD_PROMPT.format(question=question)
    if mode == "marked_no_prior":
        return MARKED_NO_PRIOR_PROMPT.format(question=question)
    if mode == "task3_evidence":
        return TASK3_EVIDENCE_PROMPT.format(question=question)
    if mode == "task3_marked_no_prior":
        return TASK3_MARKED_NO_PRIOR_PROMPT.format(question=question)
    if mode == "task3_visual_sketchpad":
        return TASK3_VISUAL_SKETCHPAD_PROMPT.format(question=question)
    if mode == "procedural_default":
        return PROCEDURAL_DEFAULT_PROMPT.format(question=question)
    if mode == "procedural_evidence":
        return PROCEDURAL_EVIDENCE_PROMPT.format(question=question)
    if mode == "procedural_marked_no_prior":
        return PROCEDURAL_MARKED_NO_PRIOR_PROMPT.format(question=question)
    if mode == "procedural_visual_sketchpad":
        return PROCEDURAL_VISUAL_SKETCHPAD_PROMPT.format(question=question)
    if mode == "procedural_srt_generic":
        return PROCEDURAL_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "procedural_srt_targeted_process":
        return PROCEDURAL_SRT_TARGETED_PROCESS_PROMPT.format(question=question)
    if mode == "procedural_srt_targeted_process_v2":
        return PROCEDURAL_SRT_TARGETED_PROCESS_V2_PROMPT.format(question=question)
    if mode == "procedural_srt_targeted_process_v3":
        return PROCEDURAL_SRT_TARGETED_PROCESS_V3_PROMPT.format(question=question)
    if mode == "procedural_srt_action_priority":
        return PROCEDURAL_SRT_ACTION_PRIORITY_PROMPT.format(question=question)
    if mode == "procedural_srt_auto_router_action":
        return PROCEDURAL_SRT_AUTO_ROUTER_ACTION_PROMPT.format(question=question)
    if mode == "procedural_boundary_classifier":
        return PROCEDURAL_BOUNDARY_CLASSIFIER_PROMPT.format(question=question)
    if mode == "procedural_srt_targeted_router":
        return PROCEDURAL_SRT_TARGETED_ROUTER_PROMPT.format(question=question)
    if mode == "procedural_srt_targeted_router_v2":
        return PROCEDURAL_SRT_TARGETED_ROUTER_V2_PROMPT.format(question=question)
    if mode == "procedural_srt_targeted_router_oracle":
        raise ValueError("Use make_prompt_with_item for oracle family prompts.")
    if mode == "procedural_srt_targeted_router_oracle_v2":
        raise ValueError("Use make_prompt_with_item for oracle family prompts.")
    if mode == "procedural_srt_gpt_verify":
        return PROCEDURAL_SRT_GPT_VERIFY_PROMPT.format(question=question)
    if mode == "procedural_srt_semantic_boundary":
        return PROCEDURAL_SRT_SEMANTIC_BOUNDARY_PROMPT.format(question=question)
    if mode == "procedural_srt_overload":
        return PROCEDURAL_SRT_OVERLOAD_PROMPT.format(question=question)
    if mode == "srt_spatial_default":
        return SRT_SPATIAL_DEFAULT_PROMPT.format(question=question)
    if mode == "srt_spatial_evidence":
        return SRT_SPATIAL_EVIDENCE_PROMPT.format(question=question)
    if mode == "srt_spatial":
        return SRT_SPATIAL_PROMPT.format(question=question)
    if mode == "srt_spatial_v2":
        return SRT_SPATIAL_V2_PROMPT.format(question=question)
    if mode == "srt_spatial_lite":
        return SRT_SPATIAL_LITE_PROMPT.format(question=question)
    if mode == "srt_spatial_minimal":
        return SRT_SPATIAL_MINIMAL_PROMPT.format(question=question)
    if mode == "srt_spatial_verify":
        return SRT_SPATIAL_VERIFY_PROMPT.format(question=question)
    if mode == "srt_spatial_adaptive":
        return SRT_SPATIAL_ADAPTIVE_PROMPT.format(question=question)
    if mode == "srt_spatial_contact_micro":
        return SRT_SPATIAL_CONTACT_MICRO_PROMPT.format(question=question)
    if mode == "srt_spatial_vsr_rules":
        return SRT_SPATIAL_VSR_RULES_PROMPT.format(question=question)
    if mode == "srt_spatial_orientation_pose":
        return SRT_SPATIAL_ORIENTATION_POSE_PROMPT.format(question=question)
    if mode == "srt_spatial_proximity_strict":
        return SRT_SPATIAL_PROXIMITY_STRICT_PROMPT.format(question=question)
    if mode == "srt_spatial_enclosure_open":
        return SRT_SPATIAL_ENCLOSURE_OPEN_PROMPT.format(question=question)
    if mode == "srt_spatial_above_support":
        return SRT_SPATIAL_ABOVE_SUPPORT_PROMPT.format(question=question)
    if mode == "srt_spatial_opposition_shared_space":
        return SRT_SPATIAL_OPPOSITION_SHARED_SPACE_PROMPT.format(question=question)
    if mode == "srt_spatial_facing_away_bearing":
        return SRT_SPATIAL_FACING_AWAY_BEARING_PROMPT.format(question=question)
    if mode == "srt_spatial_behind_occlusion":
        return SRT_SPATIAL_BEHIND_OCCLUSION_PROMPT.format(question=question)
    if mode == "srt_spatial_on_top_surface":
        return SRT_SPATIAL_ON_TOP_SURFACE_PROMPT.format(question=question)
    if mode == "srt_spatial_edge_rim":
        return SRT_SPATIAL_EDGE_RIM_PROMPT.format(question=question)
    if mode == "srt_spatial_vsr_rules_glm":
        return SRT_SPATIAL_VSR_RULES_GLM_PROMPT.format(question=question)
    if mode == "srt_spatial_enclosure_open_glm":
        return SRT_SPATIAL_ENCLOSURE_OPEN_GLM_PROMPT.format(question=question)
    if mode == "srt_spatial_opposition_shared_space_glm":
        return SRT_SPATIAL_OPPOSITION_SHARED_SPACE_GLM_PROMPT.format(question=question)
    if mode == "srt_spatial_vsr_rules_glm_ultra":
        return SRT_SPATIAL_VSR_RULES_GLM_ULTRA_PROMPT.format(question=question)
    if mode == "srt_spatial_vsr_rules_glm_refined":
        return SRT_SPATIAL_VSR_RULES_GLM_REFINED_PROMPT.format(question=question)
    if mode == "srt_spatial_vsr_rules_gpt_refined":
        return SRT_SPATIAL_VSR_RULES_GPT_REFINED_PROMPT.format(question=question)
    if mode == "srt_spatial_containment_gpt_refined":
        return SRT_SPATIAL_CONTAINMENT_GPT_REFINED_PROMPT.format(question=question)
    if mode == "srt_spatial_vsr_rules_mistral_short":
        return SRT_SPATIAL_VSR_RULES_MISTRAL_SHORT_PROMPT.format(question=question)
    if mode == "srt_spatial_containment_mistral_short":
        return SRT_SPATIAL_CONTAINMENT_MISTRAL_SHORT_PROMPT.format(question=question)
    if mode == "srt_spatial_enclosure_open_glm_ultra":
        return SRT_SPATIAL_ENCLOSURE_OPEN_GLM_ULTRA_PROMPT.format(question=question)
    if mode == "srt_spatial_opposition_shared_space_glm_ultra":
        return SRT_SPATIAL_OPPOSITION_SHARED_SPACE_GLM_ULTRA_PROMPT.format(question=question)
    if mode == "sports_action_default":
        return SPORTS_ACTION_DEFAULT_PROMPT.format(question=question)
    if mode == "sports_action_generic_srt":
        return SPORTS_ACTION_GENERIC_SRT_PROMPT.format(question=question)
    if mode == "sports_action_phase_prior":
        return SPORTS_ACTION_PHASE_PRIOR_PROMPT.format(question=question)
    if mode == "sports_action_visible_boundary_srt":
        return SPORTS_ACTION_VISIBLE_BOUNDARY_SRT_PROMPT.format(question=question)
    if mode == "sports_action_boundary_verify":
        return SPORTS_ACTION_BOUNDARY_VERIFY_PROMPT.format(question=question)
    if mode == "sports_action_setup_guard":
        return SPORTS_ACTION_SETUP_GUARD_PROMPT.format(question=question)
    if mode == "sports_action_boundary_router":
        return SPORTS_ACTION_BOUNDARY_ROUTER_PROMPT.format(question=question)
    if mode == "tooluse_default":
        return TOOLUSE_DEFAULT_PROMPT.format(question=question)
    if mode == "tooluse_srt_generic":
        return TOOLUSE_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "tooluse_srt_scenario":
        return TOOLUSE_SRT_SCENARIO_PROMPT.format(question=question)
    if mode == "tooluse_srt_scenario_v2":
        return TOOLUSE_SRT_SCENARIO_V2_PROMPT.format(question=question)
    if mode == "cleaning_default":
        return CLEANING_DEFAULT_PROMPT.format(question=question)
    if mode == "cleaning_srt_generic":
        return CLEANING_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "cleaning_srt_scenario":
        return CLEANING_SRT_SCENARIO_PROMPT.format(question=question)
    if mode == "cleaning_srt_scenario_v2":
        return CLEANING_SRT_SCENARIO_V2_PROMPT.format(question=question)
    if mode == "cleaning_srt_salience_guard":
        return CLEANING_SRT_SALIENCE_GUARD_PROMPT.format(question=question)
    if mode == "cleaning_srt_boundary_router_oracle":
        raise ValueError("Use make_prompt_with_item for cleaning boundary-router prompts.")
    if mode == "craft_default":
        return CRAFT_DEFAULT_PROMPT.format(question=question)
    if mode == "craft_srt_generic":
        return CRAFT_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "craft_srt_scenario":
        return CRAFT_SRT_SCENARIO_PROMPT.format(question=question)
    if mode == "craft_srt_scenario_v2":
        return CRAFT_SRT_SCENARIO_V2_PROMPT.format(question=question)
    if mode == "craft_srt_salience_guard":
        return CRAFT_SRT_SALIENCE_GUARD_PROMPT.format(question=question)
    if mode == "craft_srt_boundary_router_oracle":
        raise ValueError("Use make_prompt_with_item for craft boundary-router prompts.")
    if mode == "mobility_default":
        return MOBILITY_DEFAULT_PROMPT.format(question=question)
    if mode == "mobility_srt_generic":
        return MOBILITY_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "mobility_srt_scenario":
        return MOBILITY_SRT_SCENARIO_PROMPT.format(question=question)
    if mode == "mobility_srt_scenario_v2":
        return MOBILITY_SRT_SCENARIO_V2_PROMPT.format(question=question)
    if mode == "mobility_srt_salience_guard":
        return MOBILITY_SRT_SALIENCE_GUARD_PROMPT.format(question=question)
    if mode == "mobility_srt_boundary_router_oracle":
        raise ValueError("Use make_prompt_with_item for mobility boundary-router prompts.")
    if mode in {
        "process_default",
        "process_visual_evidence_only",
        "process_plain_cot",
        "process_multimodal_cot",
        "process_self_verification",
        "process_llava_cot_style",
        "process_grounded_cot",
        "process_srt_policy_only",
        "process_srt_s_t",
        "process_srt_r_t",
        "process_srt_s_r_no_t",
        "process_srt_generic",
        "process_srt_scenario",
        "process_srt_wrong_boundary_router",
        "process_srt_self_router",
        "process_srt_boundary_router_oracle",
        "process_boundary_label_only",
    }:
        raise ValueError("Use make_prompt_with_item for generic process prompts.")
    raise ValueError(f"Unknown prompt mode: {mode}")


def make_prompt_with_item(mode: str, item: dict[str, Any]) -> str:
    question = str(
        item.get("underspecified_question")
        or item.get("question")
        or item.get("original_question")
        or ""
    )
    if mode == "procedural_srt_targeted_router_oracle":
        family_hint = str(item.get("boundary_family_hint", "other"))
        return PROCEDURAL_SRT_TARGETED_ROUTER_ORACLE_PROMPT.format(
            question=question,
            family_hint=family_hint,
        )
    if mode == "procedural_srt_targeted_router_oracle_v2":
        family_hint = str(item.get("boundary_family_hint", "other"))
        return PROCEDURAL_SRT_TARGETED_ROUTER_ORACLE_V2_PROMPT.format(
            question=question,
            family_hint=family_hint,
        )
    if mode == "cleaning_srt_boundary_router_oracle":
        family_hint = str(item.get("boundary_family") or item.get("boundary_family_hint") or "other")
        if family_hint == "setup_vs_active":
            return CLEANING_SRT_SCENARIO_PROMPT.format(question=question)
        if family_hint == "active_vs_result":
            return CLEANING_SRT_SCENARIO_V2_PROMPT.format(question=question)
        if family_hint == "cleaning_vs_noncleaning_salience":
            return CLEANING_SRT_SALIENCE_GUARD_PROMPT.format(question=question)
        return CLEANING_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "craft_srt_boundary_router_oracle":
        family_hint = str(item.get("boundary_family") or item.get("boundary_family_hint") or "other")
        if family_hint == "setup_vs_active":
            return CRAFT_SRT_SCENARIO_PROMPT.format(question=question)
        if family_hint in {"active_vs_finished", "assembly_vs_finished"}:
            return CRAFT_SRT_SCENARIO_V2_PROMPT.format(question=question)
        if family_hint == "creation_vs_salience":
            return CRAFT_SRT_SALIENCE_GUARD_PROMPT.format(question=question)
        return CRAFT_SRT_GENERIC_PROMPT.format(question=question)
    if mode == "mobility_srt_boundary_router_oracle":
        family_hint = str(item.get("boundary_family") or item.get("boundary_family_hint") or "other")
        if family_hint == "setup_vs_active":
            return MOBILITY_SRT_SCENARIO_PROMPT.format(question=question)
        if family_hint == "active_vs_parked":
            return MOBILITY_SRT_SCENARIO_V2_PROMPT.format(question=question)
        if family_hint == "mobility_vs_salience":
            return MOBILITY_SRT_SALIENCE_GUARD_PROMPT.format(question=question)
        return MOBILITY_SRT_GENERIC_PROMPT.format(question=question)
    if mode in {
        "process_default",
        "process_visual_evidence_only",
        "process_plain_cot",
        "process_multimodal_cot",
        "process_self_verification",
        "process_llava_cot_style",
        "process_grounded_cot",
        "process_srt_policy_only",
        "process_srt_s_t",
        "process_srt_r_t",
        "process_srt_s_r_no_t",
        "process_srt_generic",
        "process_srt_scenario",
        "process_srt_wrong_boundary_router",
        "process_srt_self_router",
        "process_srt_boundary_router_oracle",
        "process_boundary_label_only",
    }:
        return make_process_prompt(mode, item)
    return make_prompt(mode, question)
