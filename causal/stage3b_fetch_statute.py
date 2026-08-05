"""Stage 3b — fetch the text of every provision the decisions actually cite.

Stage 3a established which variables cases differ on when a provision is cited. That is an
association and nothing more. To claim a provision makes a variable an INPUT, the
provision has to say so -- so this stage retrieves the statute itself.

The citation of record is the legislation, not a model's recollection of it. Nothing here
asks an LLM anything: it fetches primary-source HTML from legislation.nsw.gov.au, caches
it verbatim with a hash and a retrieval timestamp, and slices out the cited provisions.
Stage 3c then lets a model read ONLY what is in this cache.

Guards, because a wrong instrument identifier fetching the wrong statute would poison
everything downstream and look completely plausible:

  * the fetched page title must contain the expected instrument name, else the fetch is
    recorded as failed and no text is emitted;
  * the extracted chunk must begin with the section number that was asked for;
  * raw HTML is cached to disk with its sha256, so a re-run either hits the cache or can
    be diffed against it.

SOURCING, AND WHY IT IS MANUAL
------------------------------
Automated retrieval is not available for these instruments:

  * legislation.nsw.gov.au and austlii.edu.au both sit behind a Cloudflare bot challenge
    ("Just a moment... Enable JavaScript and cookies to continue"), returning 403 to any
    plain HTTP client;
  * austlii.edu.au/robots.txt carries an explicit `User-agent: ClaudeBot / Disallow: /`
    (alongside GPTBot, CCBot and Google-Extended).

Defeating either would mean spoofing a user agent past bot detection on a site that has
specifically opted out, so this script does not attempt it. Instead it reads primary
source that a human has saved locally:

    causal/provenance/statute_source/<anything>.html|.htm|.txt|.pdf

Drop the saved page for each instrument in that directory -- browser "Save page as", the
official PDF, or the provision text pasted into a .txt. Filenames are free-form; the
instrument is identified
from the CONTENT, by the same title guard used for a live fetch, so a mislabelled file
cannot be attributed to the wrong statute. Each file is hashed and its provenance recorded.

Instruments with no machine-readable source (the Motor Accident Guidelines are SIRA
instruments published by gazette, not on the legislation register) are recorded as
unavailable with a reason. They are not silently dropped and they are not guessed at.

Run:  python causal/stage3b_fetch_statute.py
Out:  causal/provenance/provision_text.json    (sliced provisions + source metadata)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pypdf
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
LINKS = HERE / "provenance" / "provision_links.json"
SOURCE_DIR = HERE / "provenance" / "statute_source"
OUT = HERE / "provenance" / "provision_text.json"

# Instrument -> the phrase a saved file must contain to be attributed to it. Files are
# identified by CONTENT, never by filename, so a mislabelled download cannot be filed
# against the wrong statute.
IDENTIFY = {
    "Motor Accident Injuries Act 2017": "Motor Accident Injuries Act 2017",
    "Motor Accidents Compensation Act 1999": "Motor Accidents Compensation Act 1999",
    "Motor Accident Injuries Regulation 2017": "Motor Accident Injuries Regulation 2017",
    "Personal Injury Commission Regulation 2020": "Personal Injury Commission Regulation 2020",
    "Personal Injury Commission Rules 2021": "Personal Injury Commission Rules 2021",
    "Motor Accident Guidelines": "Motor Accident Guidelines",
}
URL_VERIFIED = False   # constructed by hand; the 403 blocked confirmation
WHERE = {
    "Motor Accident Injuries Act 2017":
        "https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-2017-010",
    "Motor Accidents Compensation Act 1999":
        "https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-1999-041",
    "Motor Accident Injuries Regulation 2017":
        "https://legislation.nsw.gov.au/view/whole/html/inforce/current/sl-2017-0431",
    "Personal Injury Commission Regulation 2020":
        "https://legislation.nsw.gov.au/view/whole/html/inforce/current/sl-2020-0736",
    "Personal Injury Commission Rules 2021":
        "https://legislation.nsw.gov.au/view/whole/html/inforce/current/sl-2021-0138",
    "Motor Accident Guidelines":
        "https://www.sira.nsw.gov.au/ (SIRA instrument; gazetted, not on the register)",
}

# The workbook cites one instrument under several names.
ALIASES = {
    "Motor Accident Guidelines 2017": "Motor Accident Guidelines",
    "Motor Accident Injuries Guidelines": "Motor Accident Guidelines",
}


def canonical(act: str) -> str:
    return ALIASES.get(act, act)


def load_local() -> list[tuple[Path, str, dict]]:
    """Every file in the source directory, with its hash and mtime."""
    out = []
    if not SOURCE_DIR.exists():
        return out
    for path in sorted(SOURCE_DIR.iterdir()):
        if path.suffix.lower() not in (".html", ".htm", ".txt", ".xhtml", ".pdf"):
            continue
        blob = path.read_bytes()
        if path.suffix.lower() == ".pdf":
            try:
                pages = pypdf.PdfReader(str(path)).pages
                raw = "\n".join(pg.extract_text() or "" for pg in pages)
            except Exception as exc:                              # noqa: BLE001
                print(f"  !! {path.name}: PDF unreadable ({exc})")
                continue
            if len(raw.strip()) < 200:
                print(f"  !! {path.name}: PDF has no extractable text "
                      "(scanned image?) - needs OCR or a text source")
                continue
        else:
            # Saved pages are not always utf-8, and errors="replace" silently turns
            # em-dashes and curly apostrophes into U+FFFD -- which then goes to the model
            # as corrupted statutory text. Try the likely encodings in order instead.
            for enc in ("utf-8", "cp1252", "latin-1"):
                try:
                    raw = blob.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raw = blob.decode("utf-8", errors="replace")
            if "�" in raw:
                print(f"  !! {path.name}: contains U+FFFD after decode; text may be corrupt")
        out.append((path, raw, dict(
            source_file=path.name, bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
            file_modified=datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        )))
    return out


def to_text(raw: str) -> tuple[str, str]:
    if "<" not in raw[:2000]:                      # plain text (.txt or extracted PDF)
        first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        text = re.sub(r"[ \t\xa0]+", " ", raw)
        return first[:200], re.sub(r"\n{3,}", "\n\n", text).strip()
    soup = BeautifulSoup(raw, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


# A provision is read if it is cited at all, OR if its heading shows it governs quantum.
# Citation frequency tracks what gets ARGUED, not what governs the award -- s 4.6 MAIA
# caps loss-of-earnings damages and is cited 5 times in 540 decisions because it is never
# in dispute. Selecting on citations alone drops the rules the scheme actually runs on.
QUANTUM_HEADING = re.compile(
    r"damages|economic loss|earning|impairment|non-economic|interest|"
    r"contributory negligence|mitigation|apportion|discount rate|indexation|"
    r"care|threshold|gratuitous|assessment of claims",
    re.IGNORECASE)


# Restrict the heading sweep to the chapters that govern lump-sum damages. Without this
# the sweep also pulls in the statutory-benefits chapters -- weekly payments, medical
# expenses during the claim -- which share vocabulary ("damages", "care", "assessment")
# but do not set the award this dataset records.
DAMAGES_SCOPE = {
    # MAIA 2017 Chapter 4 is "Damages"; numbering is chapter.section.
    "Motor Accident Injuries Act 2017": lambda n: n.startswith("4."),
    # MACA 1999 Chapter 5 "Award of damages" runs ss 121-146 in flat numbering.
    "Motor Accidents Compensation Act 1999":
        lambda n: n.isdigit() and 121 <= int(n) <= 146,
}


def all_headings(text: str) -> list[tuple[str, str]]:
    """Every section heading in an instrument: (number, title)."""
    found: dict[str, str] = {}
    for num, title in re.findall(
            r"^\s*(\d+[A-Z]*(?:\.\d+)?)\s+([A-Z][^\n]{5,110})$", text, re.MULTILINE):
        found.setdefault(num, title.strip())
    return sorted(found.items(), key=lambda kv: (
        [int(x) for x in re.findall(r"\d+", kv[0])] or [0]))


def slice_section(text: str, section: str) -> str | None:
    """Take from the heading for `section` up to the next same-level heading."""
    esc = re.escape(section)
    # A heading is the number at line start, followed by a title (not a bare cross-reference).
    start = re.search(rf"^\s*{esc}\s+[A-Z(\"'][^\n]{{3,}}$", text, re.MULTILINE)
    if not start:
        return None
    rest = text[start.start():]
    nxt = re.search(r"^\s*(?:\d+[A-Z]*(?:\.\d+)?)\s+[A-Z(\"'][^\n]{3,}$",
                    rest[len(start.group(0)):], re.MULTILINE)
    body = rest[: len(start.group(0)) + nxt.start()] if nxt else rest[:12000]
    return body.strip()


def main() -> int:
    links = json.loads(LINKS.read_text(encoding="utf-8"))
    cite_counts: dict[tuple[str, str], int] = {}
    wanted: dict[str, set[str]] = {}
    for p in links.get("all_provisions", links["provisions_tested"]):
        section, act = p["provision"].split(" ", 1)
        act = canonical(act)
        wanted.setdefault(act, set()).add(section)
        cite_counts[(act, section)] = p["n_citing"]

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    files = load_local()
    print(f"source directory: {SOURCE_DIR}")
    print(f"files found: {len(files)}")

    # Identify each file by CONTENT. A file naming two instruments is ambiguous and is
    # rejected rather than guessed at.
    matched: dict[str, tuple[str, dict, str]] = {}
    unidentified = []
    for path, raw, meta in files:
        title, text = to_text(raw)
        # The TITLE decides. Scanning the body finds every instrument a statute happens to
        # mention -- MAIA 2017 refers to the Motor Accident Guidelines throughout, and MACA
        # 1999 refers to MAIA -- which made every file look ambiguous. Where a title matches
        # more than one configured phrase, the longest (most specific) wins.
        def pick(hay: str) -> list[str]:
            return [act for act, phrase in IDENTIFY.items() if phrase.lower() in hay.lower()]

        hits = pick(title)
        how = "title"
        if not hits:
            hits = pick(text[:5000])
            how = "body"
        if not hits:
            unidentified.append(dict(file=path.name, matched=[], title=title[:160],
                                     reason="no configured instrument matched"))
            print(f"  ?? {path.name}: unidentified - title was {title[:70]!r}")
            continue
        act = max(hits, key=lambda a: len(IDENTIFY[a]))
        if act in matched:
            unidentified.append(dict(file=path.name, matched=hits, title=title[:160],
                                     reason=f"duplicate source for {act}"))
            print(f"  ?? {path.name}: duplicate source for {act}")
            continue
        matched[act] = (text, dict(meta, title=title[:200], identified_by=how,
                                   also_matched=[h for h in hits if h != act],
                                   published_at=WHERE.get(act)), title)
        extra = f"  (also matched {[h for h in hits if h != act]})" if len(hits) > 1 else ""
        print(f"  ok {path.name[:52]:54} -> {act}{extra}")

    instruments, provisions, failures = {}, [], []
    for act, sections in sorted(wanted.items()):
        if act not in matched:
            reason = (f"no local source. Save {WHERE.get(act, 'the instrument')} "
                      f"into {SOURCE_DIR.name}/")
            instruments[act] = dict(available=False, reason=reason)
            failures.append(dict(act=act, reason=reason))
            for section in sorted(sections):
                provisions.append(dict(provision=f"{section} {act}", section=section,
                                       act=act, extracted=False, reason=reason,
                                       chars=0, text=None))
            continue

        text, meta, _ = matched[act]
        # Every quantum-governing section in this instrument, cited or not.
        headings = all_headings(text)
        titles = dict(headings)
        in_scope = DAMAGES_SCOPE.get(act, lambda n: False)
        added = {n for n, t in headings
                 if n not in sections and in_scope(n) and QUANTUM_HEADING.search(t)}
        if added:
            print(f"     + {len(added)} uncited quantum sections added from {act}")
        sections = set(sections) | added
        instruments[act] = dict(available=True, n_sections_read=len(sections), **meta)
        for section in sorted(sections):
            body = slice_section(text, section)
            ok = body is not None and body.lstrip().startswith(section)
            provisions.append(dict(
                provision=f"{section} {act}", section=section, act=act,
                heading=titles.get(section),
                n_citing=cite_counts.get((act, section), 0),
                selected_by=("cited" if cite_counts.get((act, section)) else "heading"),
                extracted=bool(ok),
                reason=None if ok else "section heading not found in the saved text",
                chars=len(body) if body else 0, text=body if ok else None,
                source_file=meta["source_file"], source_sha256=meta["sha256"],
                published_at=meta.get("published_at"),
            ))

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_mode="local primary source (see module docstring for why not fetched)",
        n_wanted=sum(len(v) for v in wanted.values()),
        n_extracted=sum(p["extracted"] for p in provisions),
        instruments=instruments, provisions=provisions,
        failures=failures, unidentified_files=unidentified,
        needed=[dict(act=a, sections=sorted(s), where=WHERE.get(a))
                for a, s in sorted(wanted.items()) if a not in matched],
        note=("Verbatim primary source only. No model has read or paraphrased any of this "
              "text at this stage. Stage 3c reads exclusively from the `text` fields here."),
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT}")
    print(f"provisions wanted={doc['n_wanted']}  extracted={doc['n_extracted']}")
    # Show the heading actually captured. A slice that silently grabbed the wrong section
    # is indistinguishable from a correct one in a count, and would poison stage 3c.
    for pr in provisions:
        head = ""
        if pr["extracted"]:
            head = " | " + " ".join(pr["text"].split()[:11])
        print(f"  {'ok' if pr['extracted'] else 'XX'}  {pr['provision']:44}"
              f"{pr['chars']:6d}ch{head[:74]}")
    if doc["needed"]:
        print("\nSTILL NEEDED - save each of these into "
              f"{SOURCE_DIR.relative_to(HERE.parent)}/ :")
        for n in doc["needed"]:
            print(f"  {n['act']}")
            print(f"      sections: {', '.join(n['sections'])}")
            print(f"      {n['where']}")
    return 0 if doc["n_extracted"] else 1


if __name__ == "__main__":
    sys.exit(main())
