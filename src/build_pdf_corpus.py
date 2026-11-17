"""Extract page text, figure/table captions and page renders from one PDF.

Output (under --out): pages.json, chunks.json, renders/page-NNN.png.
Chunks keep page number and source ID so evidence can be checked against the PDF.
"""
import argparse
import json
import re
from pathlib import Path

import pypdf
import pypdfium2 as pdfium

CAPTION = re.compile(r"^\s*(Bảng|Hình)\s+\d+\.\d+")
CHUNK_CHARS = 700
OVERLAP = 100


def split_text(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - OVERLAP
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/advanced"))
    parser.add_argument("--source-id", default="thesis")
    parser.add_argument("--render-scale", type=float, default=1.5)
    parser.add_argument("--figure-notes", type=Path, default=None,
                        help="JSON list of {page, text} written by a human reviewer from the rendered page")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    renders = args.out / "renders"
    renders.mkdir(exist_ok=True)

    reader = pypdf.PdfReader(str(args.pdf))
    pages = []
    chunks = []
    for index, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        pages.append({"page": index, "chars": len(text), "text": text})
        for part_no, piece in enumerate(split_text(text), start=1):
            chunks.append({
                "id": f"{args.source_id}-p{index:03d}-c{part_no}",
                "text": piece,
                "source_id": args.source_id,
                "source": args.pdf.name,
                "page": index,
                "metadata": {"kind": "text", "page_chars": len(text)},
            })
        for line in text.split("\n"):
            if CAPTION.match(line):
                chunks.append({
                    "id": f"{args.source_id}-p{index:03d}-cap{len(chunks)}",
                    "text": line.strip(),
                    "source_id": args.source_id,
                    "source": args.pdf.name,
                    "page": index,
                    "metadata": {"kind": "caption"},
                })

    if args.figure_notes:
        for note_no, note in enumerate(json.loads(args.figure_notes.read_text(encoding="utf-8")), start=1):
            chunks.append({
                "id": f"{args.source_id}-p{note['page']:03d}-note{note_no}",
                "text": re.sub(r"\s+", " ", note["text"]).strip(),
                "source_id": args.source_id,
                "source": args.pdf.name,
                "page": note["page"],
                "metadata": {"kind": "figure_note_manual"},
            })

    if not args.skip_render:
        pdf = pdfium.PdfDocument(str(args.pdf))
        for index in range(len(pdf)):
            image = pdf[index].render(scale=args.render_scale).to_pil()
            image.save(renders / f"page-{index + 1:03d}.png")

    (args.out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"pages={len(pages)} chunks={len(chunks)} out={args.out}")


if __name__ == "__main__":
    main()
