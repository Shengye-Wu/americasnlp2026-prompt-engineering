import argparse
import os
import re
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


LANG_INFO = {
    "wixarika": {
        "culture": "Wixárika",
        "language_name": "Wixárika",
        "iso": "hch",
    },
    "bribri": {
        "culture": "Bribri",
        "language_name": "Bribri",
        "iso": "bzd",
    },
    "guarani": {
        "culture": "Guaraní",
        "language_name": "Guaraní",
        "iso": "grn",
    },
    "nahuatl": {
        "culture": "Nahuatl / Nahua",
        "language_name": "Nahuatl",
        "iso": "nah",
    },
    "maya": {
        "culture": "Yucatec Maya",
        "language_name": "Yucatec Maya",
        "iso": "yua",
    },
}


def build_prompt(prompt_mode: str, culture: str, language_name: str) -> str:
    if prompt_mode == "p0_simple":
        return (
            "Describe this image in Spanish in one short sentence. "
            "Only describe what is visible. Output only the Spanish caption."
        )


    if prompt_mode == "p0_short_literal":
        return """
Describe this image in Spanish using one very short literal sentence.

Rules:
- Only describe what is visible.
- Do not add cultural interpretation.
- Do not mention rituals, symbolism, or history.
- Maximum 12 words.
- Output only the Spanish caption.
""".strip()

    if prompt_mode == "p0_medium_literal":
        return """
Describe this image in Spanish using one clear literal sentence.

Rules:
- Only describe what is visible.
- Include the main people, objects, animals, actions, and setting.
- Do not add cultural interpretation.
- Do not mention rituals, symbolism, or history.
- Maximum 25 words.
- Output only the Spanish caption.
""".strip()

    if prompt_mode == "p0_long_literal":
        return """
Describe this image in Spanish using one or two literal sentences.

Rules:
- Only describe what is visible.
- Include important visible details: people, objects, animals, actions, clothing, place, and background.
- Do not add cultural interpretation unless it is directly visible.
- Do not mention rituals, symbolism, or history.
- Maximum 45 words.
- Output only the Spanish caption.
""".strip()

    if prompt_mode == "p1_culture_aware":
        return f"""
You are helping with the AmericasNLP 2026 Shared Task on cultural image captioning.

The image belongs to the {culture} cultural context.
The final target language is {language_name}, but for this step write the caption in Spanish.

Write one concise Spanish caption.

Rules:
- Describe the visible content first.
- Mention culturally relevant objects, clothing, food, tools, buildings, or practices only if they are clearly supported by the image.
- Do not invent sacred, spiritual, ceremonial, or historical details.
- Use simple Spanish that can be translated easily.
- Maximum 2 sentences.
- Output only the Spanish caption.
""".strip()

    if prompt_mode == "p2_translation_friendly":
        return f"""
You are generating an intermediate Spanish caption for a machine translation system.

Culture: {culture}
Final target language: {language_name}

Write a short Spanish caption that will be translated into {language_name}.

Rules:
- Use simple sentence structure.
- Prefer concrete nouns and verbs.
- Avoid long explanations.
- Avoid metaphors.
- Avoid uncertain cultural claims.
- Mention cultural context only when visually supported.
- Do not invent names, rituals, sacred meanings, or uses.
- Maximum 1 sentence.
- Output only the Spanish caption.
""".strip()

    if prompt_mode == "p2b_translation_friendly_detailed":
        return f"""
You are generating an intermediate Spanish caption for a machine translation system.

Culture: {culture}
Final target language: {language_name}

Write one clear Spanish caption that will later be translated into {language_name}.

Rules:
- Use simple Spanish.
- Keep important visible details: people, objects, actions, animals, clothing, location, and culturally relevant terms.
- Do not make the caption too short if important information would be lost.
- Mention cultural context only when it is visible or strongly supported.
- Do not invent rituals, sacred meanings, historical claims, or invisible uses.
- Use 1 sentence, maximum 25 words.
- Output only the Spanish caption.
""".strip()

    if prompt_mode == "p3_object_action":
        return f"""
Describe the image in Spanish for a cultural image captioning task.

Culture: {culture}
Final target language: {language_name}

Write one caption using this structure when possible:
[main person/object] + [action or use] + [visible cultural context].

Rules:
- Only include details visible in the image or strongly supported by the cultural context.
- If the cultural use is uncertain, describe the object visually instead of guessing.
- Do not invent rituals or sacred meanings.
- Output only one Spanish sentence.
""".strip()

    if prompt_mode == "p4_direct_target":
        return f"""
You are helping with the AmericasNLP 2026 Shared Task.

Culture: {culture}
Target language: {language_name}

Write one image caption directly in {language_name}.
Do not write Spanish.
Do not write English.
Do not explain.
Output only the {language_name} caption.
""".strip()

    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def resolve_data_path(split: str, language: str) -> tuple[str, str]:
    if split == "pilot":
        data_dir = "data/pilot"
        jsonl_path = os.path.join(data_dir, f"{language}.jsonl")
    elif split == "dev":
        data_dir = os.path.join("data/dev", language)
        jsonl_path = os.path.join(data_dir, f"{language}.jsonl")
    else:
        raise ValueError("split must be 'pilot' or 'dev'")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Data file not found: {jsonl_path}")

    return data_dir, jsonl_path


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True, choices=list(LANG_INFO.keys()))
    parser.add_argument("--split", default="pilot", choices=["pilot", "dev"])
    parser.add_argument(
        "--prompt-mode",
        required=True,
        choices=[
            "p0_simple",
            "p0_short_literal",
            "p0_medium_literal",
            "p0_long_literal",
            "p1_culture_aware",
            "p2_translation_friendly",
            "p2b_translation_friendly_detailed",
            "p3_object_action",
            "p4_direct_target",
        ],
    )
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    info = LANG_INFO[args.language]
    prompt = build_prompt(args.prompt_mode, info["culture"], info["language_name"])

    data_dir, jsonl_path = resolve_data_path(args.split, args.language)
    df = pd.read_json(jsonl_path, lines=True)

    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    if args.language == "guarani":
        df["filename"] = df["filename"].str.split("data/guarani/").str[-1]

    df["filepath"] = df["filename"].apply(lambda x: os.path.join(data_dir, x))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Split: {args.split}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Samples: {len(df)}")
    print("Prompt:")
    print(prompt)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    generated = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.prompt_mode}"):
        image_path = row["filepath"]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Reduce image size to avoid CUDA out-of-memory on 8 GB GPUs.
        # This keeps the aspect ratio but limits the largest side.
        max_side = 1008
        image.thumbnail((max_side, max_side))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        generated.append(clean_text(response))

    df["generated_caption"] = generated
    df["prompt_mode"] = args.prompt_mode
    df["model"] = args.model
    df["split_used"] = args.split

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    jsonl_out = args.output_prefix + ".jsonl"
    txt_out = args.output_prefix + ".txt"

    df.to_json(jsonl_out, orient="records", lines=True, force_ascii=False)

    with open(txt_out, "w", encoding="utf-8") as f:
        for caption in generated:
            f.write(caption.replace("\n", " ") + "\n")

    print(f"Saved JSONL: {jsonl_out}")
    print(f"Saved TXT:   {txt_out}")


if __name__ == "__main__":
    main()
